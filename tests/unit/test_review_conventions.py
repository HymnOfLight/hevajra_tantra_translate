"""Conventions on review and gold sheets (collate.v2): columns, the three states, import
validation, the glossary check callback, legacy sheets and the way into the matrix."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hevajra_matrix.core.lexicon import StandingEntry, StandingRenderings
from hevajra_matrix.core.types import Alignment, Grade, Link, Relation, Status
from hevajra_matrix.matrix.build import build_cells
from hevajra_matrix.matrix.status import has_convention
from hevajra_matrix.review import gold_sheets, sheets
from hevajra_matrix.review.sampling import make_item
from hevajra_matrix.review.verdicts import VerdictError, load, save

from test_review_sheets import PARAMS, REFS, SEGMENTS, WITS, _blind_verdict, _filled_blind, machine, read, write_rows

REF_W, WIT_W = "bo_ref", "zh_wit"
GROUPS = [(("I.1",), ("pin1",))]


def accept(unit_id: str, wit_ids: tuple[str, ...]) -> str | None:
    return None


def refuse(unit_id: str, wit_ids: tuple[str, ...]) -> str | None:
    return f"not a listed rendering ({unit_id})"


def machine_with(flags_by_unit: dict[str, frozenset[str]]):
    """The fixture's machine matrix, with convention flags on some consensus links."""
    alignment, _ = machine()
    links = tuple(Link(lk.ref_id, lk.wit_ids, lk.relation, quotes=lk.quotes, source=lk.source,
                       flags=flags_by_unit.get(lk.ref_id or "", frozenset())) for lk in alignment.links)
    alignment = Alignment(alignment.source, REF_W, WIT_W, links)
    grades = {r.id: Grade.B for r in REFS} | {"+" + WITS[9].id: Grade.C}
    cells, orphans, _ = build_cells(REFS, alignment, grades, {}, (), (), WITS, GROUPS)
    return alignment, cells + orphans


# --------------------------------------------------------------------------- columns
def test_sheet_columns_carry_conventions_and_blind_sheets_stay_blind():
    assert sheets.BLIND_COLUMNS.index("blind_convention") == sheets.BLIND_COLUMNS.index("blind_wit_loci") + 1
    assert "blind_convention" in sheets.RESOLVE_COLUMNS
    reveal = sheets.REVEAL_COLUMNS
    assert reveal.index("machine_convention") == reveal.index("machine_polarity_flip") + 1
    assert reveal.index("final_convention") == reveal.index("final_wit_loci") + 1
    assert not [c for c in sheets.BLIND_COLUMNS if c.startswith("machine_")]
    ref = gold_sheets.GOLD_REF_COLUMNS
    assert ref.index("convention") == ref.index("relation") + 1 and "convention" not in gold_sheets.GOLD_WIT_COLUMNS


def test_the_glossary_is_written_beside_the_sheets(tmp_path: Path):
    _, cells = machine()
    entry = StandingEntry("hevajra", ("ka", "kha"), ("\u7a7a\u667a",))
    paths = sheets.export([make_item("verify", REFS[1].id, "s")], "verify", tmp_path, SEGMENTS, cells, batch="v1",
                          glossary=StandingRenderings((entry,)))
    assert tmp_path / sheets.GLOSSARY_FILE in paths
    header, rows = read(tmp_path / sheets.GLOSSARY_FILE)
    assert tuple(header) == sheets.GLOSSARY_COLUMNS
    assert rows == [{"entry": "hevajra", "reference_forms": "ka | kha", "witness_renderings": "\u7a7a\u667a"}]
    gold = sheets.export([make_item("gold", REFS[0].id)], "gold", tmp_path / "gold", SEGMENTS, [], batch="dev_r1",
                         witness_chapters=["pin1"], glossary=StandingRenderings.empty())
    header, rows = read(tmp_path / "gold" / sheets.GLOSSARY_FILE)
    assert tmp_path / "gold" / sheets.GLOSSARY_FILE in gold and rows == [] and tuple(header) == sheets.GLOSSARY_COLUMNS
    topics = sheets.export([make_item("topics", REFS[2].id)], "topics", tmp_path / "t", SEGMENTS, [], batch="t1",
                           glossary=StandingRenderings.empty())
    assert not (tmp_path / "t" / sheets.GLOSSARY_FILE).exists() and len(topics) == 1


# --------------------------------------------------------------------------- blind import
def test_blind_import_reads_the_convention_and_a_blank_cell_is_none(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[0]["blind_convention"] = "transcribed_name"
    write_rows(path, header, rows)
    unit, orphan = sheets.import_(path, "verify", params=PARAMS, date="2026-10-03", items=items)
    assert (unit.blind_convention, unit.final_convention) == ("transcribed_name", "")
    assert orphan.blind_convention == "none"                      # blank cell on a sheet that has the column
    assert "transcribed_name" not in unit.flags


def test_resolve_copies_the_blind_convention_to_the_final_one(tmp_path: Path):
    path, _ = _filled_blind(tmp_path, task="resolve")
    header, rows = read(path)
    rows[0]["blind_convention"] = "transcribed_name"
    write_rows(path, header, rows)
    v = sheets.import_(path, "resolve", params=PARAMS, date="2026-10-03")[0]
    assert (v.blind_convention, v.final_convention) == ("transcribed_name", "transcribed_name")


def test_a_deferred_blind_convention_imports_unannotated(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[0]["blind_convention"] = "?"                 # the glossary entry awaits verification
    write_rows(path, header, rows)
    unit = sheets.import_(path, "verify", params=PARAMS, items=items)[0]
    assert (unit.blind_convention, unit.final_convention) == ("", "")


def test_a_deferred_convention_is_refused_where_none_can_be_recorded(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[1].update(blind_relation="no_counterpart", blind_convention="?")
    rows[2]["blind_convention"] = "\uff1f"                                   # an orphan row, full-width
    write_rows(path, header, rows)
    with pytest.raises(VerdictError) as exc:
        sheets.import_(path, "verify", params=PARAMS, items=items)
    text = str(exc.value)
    assert "line 3: '?' defers a convention, but no_counterpart rows take none: leave the cell blank" in text
    assert "line 4: '?' defers a convention, but orphan rows take none" in text


def test_blind_import_refuses_bad_conventions_with_their_line(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[0]["blind_convention"] = "proper_name"
    rows[1].update(blind_relation="no_counterpart", blind_convention="transcribed_name")
    rows[2]["blind_convention"] = "transcribed_name"                         # an orphan row
    write_rows(path, header, rows)
    with pytest.raises(VerdictError) as exc:
        sheets.import_(path, "verify", params=PARAMS, items=items)
    text = str(exc.value)
    assert "line 2: convention 'proper_name' is not one of none, standing_rendering, transcribed_name" in text
    assert "line 3: convention transcribed_name needs a linked relation (not no_counterpart" in text
    assert "line 4: convention transcribed_name needs a linked relation (not addition" in text


def test_a_convention_or_a_machine_flag_among_the_human_flags_is_refused(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[0]["blind_flags"] = "standing_rendering;convention_unsupported"
    write_rows(path, header, rows)
    with pytest.raises(VerdictError) as exc:
        sheets.import_(path, "verify", params=PARAMS, items=items)
    assert "line 2: flag 'standing_rendering' is a convention" in str(exc.value)
    assert "line 2: flag 'convention_unsupported' is set by the code checks only" in str(exc.value)


def test_standing_rendering_needs_the_glossary_check(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[0]["blind_convention"] = "standing_rendering"
    write_rows(path, header, rows)
    with pytest.raises(VerdictError, match="line 2: standing_rendering is accepted only for a verified glossary"):
        sheets.import_(path, "verify", params=PARAMS, items=items)               # no check: the empty glossary
    with pytest.raises(VerdictError, match=f"line 2: not a listed rendering \\({REFS[4].id}\\)"):
        sheets.import_(path, "verify", params=PARAMS, items=items, check=refuse)
    v = sheets.import_(path, "verify", params=PARAMS, items=items, check=accept)[0]
    assert v.blind_convention == "standing_rendering"


def test_a_blind_sheet_of_before_the_column_imports_unannotated(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    old = [c for c in header if c != "blind_convention"]
    write_rows(path, old, [{k: r[k] for k in old} for r in rows])
    unit = sheets.import_(path, "verify", params=PARAMS, items=items)[0]
    assert unit.blind_convention == ""


# --------------------------------------------------------------------------- reveal
def test_reveal_sheet_shows_the_machine_convention(tmp_path: Path):
    alignment, cells = machine_with({REFS[1].id: frozenset({"transcribed_name"}),
                                     REFS[2].id: frozenset({"standing_rendering", "convention_unsupported"})})
    items = [make_item("verify", r.id, "s", 0.5) for r in REFS[:5]]
    blind = [_blind_verdict(i, "abridged", (WITS[n].id,)) for n, i in enumerate(items)]
    blind[1] = replace(blind[1], blind_relation="equivalent", blind_convention="transcribed_name")
    blind[3] = replace(blind[3], blind_relation="no_counterpart", blind_wit_ids=())
    path = sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", links=alignment.by_ref(),
                         blind=blind)[0]
    _, rows = read(path)
    assert [r["machine_convention"] for r in rows] == ["none", "transcribed_name", "standing_rendering:unsupported",
                                                       "none", "none"]
    assert (rows[1]["blind_convention"], rows[1]["final_convention"]) == ("transcribed_name", "transcribed_name")
    # a blind verdict of before v2 is not annotated: '?' where a convention can be recorded
    assert (rows[0]["blind_convention"], rows[0]["final_convention"]) == ("?", "?")
    assert (rows[3]["blind_convention"], rows[3]["final_convention"]) == ("", "")      # no_counterpart takes none
    out = sheets.import_(path, "reveal", params=PARAMS, blind=blind)
    assert [v.machine_convention for v in out] == ["none", "transcribed_name", "standing_rendering", "none", "none"]
    assert [v.final_convention for v in out] == ["", "transcribed_name", "", "none", ""]
    assert out[0].final_relation == "abridged", "an unedited legacy row is no revision"


def test_reveal_sheet_without_a_machine_cell_shows_no_convention(tmp_path: Path):
    alignment, cells = machine()
    cells = [c for c in cells if c.unit_id != REFS[1].id]
    unaligned = next(c for c in cells if c.unit_id == REFS[2].id)
    cells = [replace(c, status=Status.UNALIGNED, relation=None, wit_ids=(), reason="invalid") if c is unaligned else c
             for c in cells]
    items = [make_item("verify", REFS[1].id, "s", 0.5), make_item("verify", REFS[2].id, "s", 0.5)]
    blind = [_blind_verdict(i, "abridged", (WITS[1].id,)) for i in items]
    _, rows = read(sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", blind=blind)[0])
    assert [r["machine_convention"] for r in rows] == ["", ""]


def test_a_changed_final_convention_is_a_revision(tmp_path: Path):
    alignment, cells = machine()
    items = [make_item("verify", REFS[1].id, "s", 0.5)]
    blind = [replace(_blind_verdict(items[0], "abridged", (WITS[1].id,)), blind_convention="none")]
    path = sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", blind=blind)[0]
    header, rows = read(path)
    rows[0]["final_convention"] = "transcribed_name"
    write_rows(path, header, rows)
    with pytest.raises(VerdictError, match="line 2: the final decision differs from the blind one"):
        sheets.import_(path, "reveal", params=PARAMS, blind=blind)
    rows[0].update(revised_reason="a name read by sound", final_relation="no_counterpart", final_wit_loci="")
    write_rows(path, header, rows)
    with pytest.raises(VerdictError, match="needs a linked relation"):
        sheets.import_(path, "reveal", params=PARAMS, blind=blind)
    rows[0].update(final_relation="abridged", final_wit_loci=WITS[1].id)
    write_rows(path, header, rows)
    v = sheets.import_(path, "reveal", params=PARAMS, blind=blind)[0]
    assert (v.blind_convention, v.final_convention, v.note) == ("none", "transcribed_name",
                                                                "revised: a name read by sound")
    rows[0]["machine_convention"] = "gloss"
    write_rows(path, header, rows)
    with pytest.raises(VerdictError, match="machine_convention 'gloss'"):
        sheets.import_(path, "reveal", params=PARAMS, blind=blind)


def _reveal(tmp_path: Path, blind_convention: str):
    _, cells = machine()
    items = [make_item("verify", REFS[1].id, "s", 0.5)]
    blind = [replace(_blind_verdict(items[0], "abridged", (WITS[1].id,)), blind_convention=blind_convention)]
    path = sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", blind=blind)[0]
    header, rows = read(path)
    return path, header, rows, blind


def test_deferring_a_decided_blind_convention_at_reveal_is_a_revision(tmp_path: Path):
    path, header, rows, blind = _reveal(tmp_path, "transcribed_name")
    rows[0]["final_convention"] = "?"
    write_rows(path, header, rows)
    with pytest.raises(VerdictError, match="line 2: the final decision differs from the blind one"):
        sheets.import_(path, "reveal", params=PARAMS, blind=blind)
    rows[0]["revised_reason"] = "the entry is not verified yet"
    write_rows(path, header, rows)
    assert sheets.import_(path, "reveal", params=PARAMS, blind=blind)[0].final_convention == ""


def test_deciding_a_deferred_blind_convention_at_reveal_is_no_revision(tmp_path: Path):
    path, header, rows, blind = _reveal(tmp_path, "")
    assert (rows[0]["blind_convention"], rows[0]["final_convention"]) == ("?", "?")
    assert sheets.import_(path, "reveal", params=PARAMS, blind=blind)[0].final_convention == ""   # kept deferred
    rows[0]["final_convention"] = "transcribed_name"
    write_rows(path, header, rows)
    v = sheets.import_(path, "reveal", params=PARAMS, blind=blind)[0]
    assert (v.blind_convention, v.final_convention, v.note) == ("", "transcribed_name", "")


def test_deferring_a_blind_none_at_reveal_is_no_revision(tmp_path: Path):
    path, header, rows, blind = _reveal(tmp_path, "none")
    rows[0]["final_convention"] = "?"
    write_rows(path, header, rows)
    v = sheets.import_(path, "reveal", params=PARAMS, blind=blind)[0]
    assert (v.blind_convention, v.final_convention, v.note) == ("none", "", "")


def test_reveal_refuses_a_prefilled_question_mark_on_a_relation_without_a_convention(tmp_path: Path):
    path, header, rows, blind = _reveal(tmp_path, "")
    rows[0].update(final_relation="no_counterpart", final_wit_loci="", revised_reason="no counterpart after all")
    write_rows(path, header, rows)                                       # final_convention keeps its '?'
    with pytest.raises(VerdictError, match="line 2: '\\?' defers a convention, but no_counterpart rows take none"):
        sheets.import_(path, "reveal", params=PARAMS, blind=blind)
    rows[0]["final_convention"] = ""
    write_rows(path, header, rows)
    assert sheets.import_(path, "reveal", params=PARAMS, blind=blind)[0].final_convention == "none"


def test_a_deferred_blind_convention_survives_the_reveal(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[0]["blind_convention"] = "?"
    write_rows(path, header, rows)
    save(sheets.import_(path, "verify", params=PARAMS, date="2026-10-03", items=items),
         tmp_path / "committed" / "zh_wit" / "b7.csv")
    blind = load(tmp_path / "committed")
    _, cells = machine()
    revealed = [i for i in items if i.item_id in {v.item_id for v in blind}]
    reveal = sheets.export(revealed, "reveal", tmp_path / "reveal", SEGMENTS, cells, batch="b7", blind=blind)[0]
    _, rows = read(reveal)
    assert [(r["blind_convention"], r["final_convention"]) for r in rows] == [("?", "?"), ("none", "none")]
    out = sheets.import_(reveal, "reveal", params=PARAMS, blind=blind)         # unedited
    assert [(v.blind_convention, v.final_convention) for v in out] == [("", ""), ("none", "none")]


def test_legacy_and_half_legacy_reveal_sheets(tmp_path: Path):
    alignment, cells = machine()
    items = [make_item("verify", REFS[1].id, "s", 0.5)]
    blind = [_blind_verdict(items[0], "abridged", (WITS[1].id,))]
    path = sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", blind=blind)[0]
    header, rows = read(path)
    old = [c for c in header if c not in sheets.CONVENTION_SHEET_COLUMNS]
    write_rows(path, old, [{k: r[k] for k in old} for r in rows])
    v = sheets.import_(path, "reveal", params=PARAMS, blind=blind)[0]
    assert (v.machine_convention, v.final_convention, v.final_relation) == ("", "", "abridged")
    half = [c for c in header if c != "final_convention"]
    write_rows(path, half, [{k: r[k] for k in half} for r in rows])
    with pytest.raises(VerdictError, match="expected columns"):
        sheets.import_(path, "reveal", params=PARAMS, blind=blind)


# --------------------------------------------------------------------------- gold
def _gold_sheet(tmp_path: Path):
    items = [make_item("gold", r.id) for r in REFS[:3]]
    ref_path = sheets.export(items, "gold", tmp_path, SEGMENTS, [], batch="test_w01", witness_chapters=["pin1"])[0]
    header, rows = read(ref_path)
    rows[0].update(links="z0001", relation="equivalent", convention="transcribed_name")
    rows[1].update(links="z0002", relation="paraphrase", convention="standing_rendering")
    rows[2].update(links="", relation="no_counterpart")
    write_rows(ref_path, header, rows)
    return ref_path, header, rows


def test_gold_import_records_the_convention_and_checks_standing_renderings(tmp_path: Path):
    ref_path, header, rows = _gold_sheet(tmp_path)
    with pytest.raises(VerdictError, match="test_w01.ref.csv line 3: standing_rendering is accepted only"):
        sheets.import_(ref_path, "gold", params=PARAMS)
    seen = []
    out = sheets.import_(ref_path, "gold", params=PARAMS, check=lambda u, w: seen.append((u, w)))
    assert seen == [(REFS[1].id, (WITS[1].id,))]                     # handles resolved to segment ids
    assert [v.blind_convention for v in out] == ["transcribed_name", "standing_rendering", "none"]
    rows[2]["convention"] = "transcribed_name"
    write_rows(ref_path, header, rows)
    with pytest.raises(VerdictError, match="line 4: convention transcribed_name needs a linked relation"):
        sheets.import_(ref_path, "gold", params=PARAMS, check=accept)


def test_a_deferred_gold_convention_imports_unannotated(tmp_path: Path):
    ref_path, header, rows = _gold_sheet(tmp_path)
    rows[1]["convention"] = "?"           # the glossary entry awaits verification: no check, no value
    write_rows(ref_path, header, rows)
    out = sheets.import_(ref_path, "gold", params=PARAMS)
    assert [v.blind_convention for v in out] == ["transcribed_name", "", "none"]


def test_a_deferred_gold_convention_is_refused_on_a_no_counterpart_row(tmp_path: Path):
    ref_path, header, rows = _gold_sheet(tmp_path)
    rows[2]["convention"] = "?"
    write_rows(ref_path, header, rows)
    with pytest.raises(VerdictError, match="test_w01.ref.csv line 4: '\\?' defers a convention, but no_counterpart"):
        sheets.import_(ref_path, "gold", params=PARAMS, check=accept)


def test_a_gold_sheet_of_before_the_column_imports_unannotated(tmp_path: Path):
    ref_path, header, rows = _gold_sheet(tmp_path)
    old = [c for c in header if c != "convention"]
    write_rows(ref_path, old, [{k: r[k] for k in old} for r in rows])
    out = sheets.import_(ref_path, "gold", params=PARAMS)
    assert [v.blind_convention for v in out] == ["", "", ""]


# --------------------------------------------------------------------------- into the matrix
def test_a_human_convention_reaches_the_cell_and_keeps_grade_a(tmp_path: Path):
    path, _ = _filled_blind(tmp_path, task="resolve")
    header, rows = read(path)
    rows[0]["blind_convention"] = "transcribed_name"
    write_rows(path, header, rows)
    verdicts = sheets.import_(path, "resolve", params=PARAMS, date="2026-10-03")
    save(verdicts, tmp_path / "committed" / "zh_wit" / "b7.csv")
    assert load(tmp_path / "committed") == verdicts
    alignment, _ = machine_with({REFS[5].id: frozenset({"transcribed_name"})})
    grades = {r.id: Grade.B for r in REFS} | {"+" + WITS[9].id: Grade.C}
    cells, orphans, _ = build_cells(REFS, alignment, grades, {}, (), verdicts, WITS, GROUPS)
    human = next(c for c in cells if c.unit_id == REFS[4].id)
    assert (human.grade, human.source, human.relation) == (Grade.A, "verdict:b7", Relation.ABRIDGED.value)
    assert human.flags == {"transcribed_name", "unsure"} and has_convention(human.flags)
    assert has_convention(next(c for c in cells if c.unit_id == REFS[5].id).flags)        # machine cell untouched
    assert not any(has_convention(c.flags) for c in orphans)


def test_a_deferred_human_convention_adds_no_flag_and_drops_the_machine_one(tmp_path: Path):
    path, _ = _filled_blind(tmp_path, task="resolve")
    header, rows = read(path)
    rows[0]["blind_convention"] = "?"
    write_rows(path, header, rows)
    verdicts = sheets.import_(path, "resolve", params=PARAMS, date="2026-10-03")
    assert (verdicts[0].blind_convention, verdicts[0].final_convention) == ("", "")      # copied to final
    alignment, machine_cells = machine_with({REFS[4].id: frozenset({"transcribed_name"})})
    assert has_convention(next(c for c in machine_cells if c.unit_id == REFS[4].id).flags)
    grades = {r.id: Grade.B for r in REFS} | {"+" + WITS[9].id: Grade.C}
    cells, _, _ = build_cells(REFS, alignment, grades, {}, (), verdicts, WITS, GROUPS)
    human = next(c for c in cells if c.unit_id == REFS[4].id)
    assert human.grade is Grade.A and human.flags == {"unsure"}, "a remaining '?' counts as none in the matrix"
