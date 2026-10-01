"""review.sheets: exact sheet columns, blindness, reveal, import without text, quote truncation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hevajra_matrix.core.textnorm import fingerprint
from hevajra_matrix.core.types import Alignment, Grade, Link, Quote, Relation, Segment, Status, WitnessOnlyKind
from hevajra_matrix.matrix.build import build_cells
from hevajra_matrix.review import gold_sheets, sheets, topic_sheets
from hevajra_matrix.review.sampling import ReviewParams, make_item
from hevajra_matrix.review.verdicts import VerdictError, VerdictRecord, load, save
from hevajra_matrix.topics import TopicLabel, load_codebook
from hevajra_matrix.topics.prelabel import Prelabel

REF_W, WIT_W = "bo_ref", "zh_wit"
TSHEG = chr(0x0F0B)
PARAMS = ReviewParams(minutes={"verify": 2.5}, competence={"verify": ("bo", "zh")},
                      quote_max_chars={"zh": 30, "bo": 60})


def cjk(n: int, start: int = 0) -> str:
    """n distinct CJK ideographs (generated, so this file stays ASCII)."""
    return "".join(chr(0x4E00 + start + i) for i in range(n))


def tib(n: int) -> str:
    """n Tibetan syllables of two letters each, joined by tsheg."""
    return TSHEG.join(chr(0x0F40 + i % 20) + chr(0x0F63) for i in range(n))


def mk(sid: str, witness: str, lang: str, text: str, chapter=None, local=None) -> Segment:
    return Segment(sid, witness, lang, text, sid.split(":")[1].rsplit(".", 1)[0], "", "verse", local, chapter,
                   fingerprint(text, lang))


REFS = [mk(f"D417:1a.{i}.1", REF_W, "bo", tib(3 + i), chapter="I.1") for i in range(1, 9)]
WITS = [mk(f"T0892:0587a{10 + i}.1", WIT_W, "zh", cjk(8, 10 * i), local="pin1") for i in range(10)]
SEGMENTS = REFS + WITS


def machine():
    links = [Link(r.id, (WITS[i].id,), Relation.EQUIVALENT, quotes=(Quote("ref", "rq"), Quote("wit", "wq")),
                  source="claude:consensus") for i, r in enumerate(REFS)]
    links[4] = Link(REFS[4].id, (), Relation.NO_COUNTERPART, source="claude:consensus")
    links.append(Link(None, (WITS[9].id,), WitnessOnlyKind.ADDITION, source="claude:consensus"))
    alignment = Alignment("claude:consensus", REF_W, WIT_W, tuple(links))
    grades = {r.id: Grade.B for r in REFS} | {"+" + WITS[9].id: Grade.C}
    cells, orphans, _ = build_cells(REFS, alignment, grades, {}, (), (), WITS, [(("I.1",), ("pin1",))])
    return alignment, cells + orphans


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "sheets are UTF-8 with BOM"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or ()), list(reader)


def write_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- export
def test_blind_sheet_columns_context_and_no_machine_output(tmp_path: Path):
    _, cells = machine()
    items = [make_item("verify", REFS[4].id, "pos:absent:other"), make_item("verify", "+" + WITS[9].id, "pos:witness_only")]
    paths = sheets.export(items, "verify", tmp_path, SEGMENTS, cells, batch="v1")
    assert paths[0].name == "v1.blind.csv"
    header, rows = read(paths[0])
    assert tuple(header) == sheets.BLIND_COLUMNS
    assert not [c for c in header if c.startswith("machine_")]
    unit_row, orphan_row = rows
    # Units 1-3 and 5-7 (0-based) link WITS[1..3] and WITS[5..7]: the span is WITS[1..7].
    assert unit_row["zh_context_loci"].split() == [w.id for w in WITS[1:8]]
    assert unit_row["zh_context"].splitlines()[0] == f"[{WITS[1].id}] {WITS[1].text}"
    assert unit_row["ref_text"] == REFS[4].text and unit_row["fingerprint"] == REFS[4].fingerprint
    assert unit_row["blind_relation"] == "" and unit_row["task"] == "verify"
    assert orphan_row["ref_text"] == "" and WITS[9].id in orphan_row["zh_context_loci"].split()
    assert (tmp_path / "witness_text" / "pin1.txt").read_text(encoding="utf-8").startswith(f"{WITS[0].id}\tverse\t")


def test_blind_sheet_rejects_items_of_another_task(tmp_path: Path):
    _, cells = machine()
    with pytest.raises(ValueError, match="task audit"):
        sheets.export([make_item("verify", REFS[0].id)], "audit", tmp_path, SEGMENTS, cells)


def test_gold_sheets_are_blind(tmp_path: Path):
    _, cells = machine()
    items = [make_item("gold", r.id) for r in REFS[:3]]
    with pytest.raises(ValueError, match="witness_chapters"):
        sheets.export(items, "gold", tmp_path, SEGMENTS, cells, batch="test_w01")
    ref_path, wit_path = sheets.export(items, "gold", tmp_path, SEGMENTS, [], batch="test_w01",
                                       witness_chapters=["pin1"])[:2]          # gold needs no machine cells
    header, rows = read(ref_path)
    assert tuple(header) == gold_sheets.GOLD_REF_COLUMNS and [r["row"] for r in rows] == ["1", "2", "3"]
    assert all(r["links"] == r["relation"] == "" for r in rows)
    header, rows = read(wit_path)
    assert tuple(header) == gold_sheets.GOLD_WIT_COLUMNS
    assert [r["handle"] for r in rows[:2]] == ["z0001", "z0002"] and len(rows) == 10


def test_topics_sheets_first_with_and_second_without_prelabels(tmp_path: Path):
    items = [make_item("topics", REFS[2].id)]
    pre = {REFS[2].id: Prelabel(REFS[2].id, frozenset({"harm"}), cues=(("harm", "cue"),))}
    first = sheets.export(items, "topics", tmp_path, SEGMENTS, [], batch="t1", prelabels=pre)[0]
    header, rows = read(first)
    assert tuple(header) == topic_sheets.TOPIC_COLUMNS
    assert rows[0]["prelabel_topics"] == "harm" and rows[0]["prelabel_cue"] == "harm: cue"
    assert rows[0]["context_before"] == f"{REFS[0].text} / {REFS[1].text}"
    second = sheets.export(items, "topics_second", tmp_path, SEGMENTS, [], batch="t1", prelabels=pre)[0]
    header, _ = read(second)
    assert tuple(header) == topic_sheets.TOPIC_SECOND_COLUMNS
    assert not [c for c in header if "prelabel" in c]
    for path in (first, second):                      # blind to the witness: no Chinese at all
        assert not any(0x4E00 <= ord(ch) <= 0x9FFF for ch in path.read_text(encoding="utf-8-sig"))


def test_reveal_sheet(tmp_path: Path):
    alignment, cells = machine()
    items = [make_item("verify", REFS[1].id, "pos:x:other")]
    blind = [_blind_verdict(items[0], "abridged", (WITS[1].id,))]
    path = sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", links=alignment.by_ref(),
                         blind=blind)[0]
    header, rows = read(path)
    assert tuple(header) == sheets.REVEAL_COLUMNS
    row = rows[0]
    assert (row["blind_relation"], row["final_relation"], row["final_wit_loci"]) == ("abridged", "abridged", WITS[1].id)
    assert (row["machine_relation"], row["machine_grade"], row["machine_wit_loci"]) == ("equivalent", "B", WITS[1].id)
    assert row["machine_quotes"] == "ref: rq | wit: wq"
    assert row["machine_polarity_flip"] == "0"
    with pytest.raises(ValueError, match="no reveal"):
        sheets.export([make_item("resolve", REFS[1].id)], "reveal", tmp_path, SEGMENTS, cells, blind=blind)
    with pytest.raises(ValueError, match="no blind verdict"):
        sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, blind=[])


def _blind_verdict(item, value, ids):
    return VerdictRecord(batch_id="v1", item_id=item.item_id, task=item.task, unit_id=item.unit_id,
                         fingerprint=REFS[1].fingerprint, stratum=item.stratum, inclusion_prob=0.5,
                         blind_relation=value, blind_wit_ids=ids, blind_date="2026-10-01", annotator="ab")


# --------------------------------------------------------------------------- truncation
def test_truncate_quote():
    assert sheets.truncate_quote(cjk(40), "zh", PARAMS) == cjk(30)
    assert sheets.truncate_quote(" a  b ", "zh", PARAMS) == "a b"
    long_bo = tib(30)                                   # 3 characters per syllable incl. tsheg
    cut = sheets.truncate_quote(long_bo, "bo", PARAMS)
    assert len(cut) <= 60 and long_bo.startswith(cut) and cut.endswith(TSHEG)
    wide = ReviewParams(quote_max_chars={"zh": 500})
    assert len(sheets.truncate_quote(cjk(100), "zh", wide)) == 60          # licence cap wins


# --------------------------------------------------------------------------- import
def _filled_blind(tmp_path: Path, task: str = "verify"):
    _, cells = machine()
    items = [make_item(task, REFS[4].id, "pos:absent:other", 1.0), make_item(task, REFS[5].id, "pos:absent:other", 1.0),
             make_item(task, "+" + WITS[9].id, "pos:witness_only", 1.0)]
    path = sheets.export(items, task, tmp_path, SEGMENTS, cells, batch="b7")[0]
    header, rows = read(path)
    rows[0].update(blind_relation="abridged", blind_wit_loci=f"{WITS[4].id}", blind_flags="unsure", note="hmm")
    rows[2].update(blind_relation="addition", blind_wit_loci=WITS[9].id)
    rows[0]["ref_text"] = REFS[4].text + tib(40)
    write_rows(path, header, rows)
    return path, items


def test_import_blind_strips_text_and_truncates_quotes(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    verdicts = sheets.import_(path, "verify", params=PARAMS, annotator="ab", date="2026-10-03", items=items)
    assert [v.unit_id for v in verdicts] == [REFS[4].id, "+" + WITS[9].id]          # unfilled row skipped
    v = verdicts[0]
    assert (v.batch_id, v.stratum, v.inclusion_prob, v.blind_relation, v.blind_wit_ids) == (
        "b7", "pos:absent:other", 1.0, "abridged", (WITS[4].id,))
    assert v.final_relation == "" and v.flags == {"unsure"} and v.note == "hmm"
    assert len(v.quote_ref) <= 60 and v.quote_zh == WITS[4].text                   # zh quote from the context
    save(verdicts, tmp_path / "committed" / "zh_wit" / "b7.csv")
    committed = (tmp_path / "committed" / "zh_wit" / "b7.csv").read_text(encoding="utf-8")
    assert REFS[4].text + tib(40) not in committed and WITS[3].text not in committed
    assert load(tmp_path / "committed") == verdicts


def test_import_resolve_copies_blind_to_final(tmp_path: Path):
    path, items = _filled_blind(tmp_path, task="resolve")
    v = sheets.import_(path, "resolve", params=PARAMS, date="2026-10-03")[0]
    assert (v.final_relation, v.final_wit_ids, v.final_date) == ("abridged", (WITS[4].id,), "2026-10-03")


def test_import_reports_every_problem(tmp_path: Path):
    path, items = _filled_blind(tmp_path)
    header, rows = read(path)
    rows[0]["blind_relation"] = "omitted"
    rows[1].update(blind_relation="equivalent", blind_flags="odd")
    rows[2]["task"] = "audit"
    write_rows(path, header, rows)
    with pytest.raises(VerdictError) as exc:
        sheets.import_(path, "verify", params=PARAMS, items=items[:2])
    text = str(exc.value)
    for expected in ("'omitted' is not one of", "needs witness segment ids", "unknown flag", "task 'audit'",
                     "not in the plan"):
        assert expected in text


def test_import_refuses_a_reveal_sheet_as_blind(tmp_path: Path):
    alignment, cells = machine()
    items = [make_item("verify", REFS[1].id, "s", 0.5)]
    path = sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", blind=[_blind_verdict(items[0], "abridged", (WITS[1].id,))])[0]
    with pytest.raises(VerdictError, match="expected columns"):
        sheets.import_(path, "verify", params=PARAMS, items=items)


def test_import_reveal(tmp_path: Path):
    alignment, cells = machine()
    items = [make_item("verify", REFS[1].id, "pos:x:other", 0.5), make_item("verify", REFS[2].id, "pos:x:other", 0.5)]
    blind = [_blind_verdict(items[0], "abridged", (WITS[1].id,)),
             _blind_verdict(items[1], "abridged", (WITS[2].id,))]
    path = sheets.export(items, "reveal", tmp_path, SEGMENTS, cells, batch="v1", links=alignment.by_ref(), blind=blind)[0]
    header, rows = read(path)
    rows[0].update(final_relation="equivalent", revised_reason="missed the second clause")
    write_rows(path, header, rows)
    not_on_sheet = _blind_verdict(make_item("verify", REFS[3].id, "pos:x:other", 0.5), "equivalent", (WITS[3].id,))
    out = sheets.import_(path, "reveal", params=PARAMS, date="2026-10-09", blind=blind + [not_on_sheet],
                         instrument_digest="d" * 64)
    changed, kept, untouched = out
    assert untouched == not_on_sheet                       # the whole batch comes back, in order
    assert (changed.blind_relation, changed.final_relation, changed.final_date) == ("abridged", "equivalent", "2026-10-09")
    assert changed.note == "revised: missed the second clause"
    assert (changed.machine_relation, changed.machine_status, changed.instrument_digest) == ("equivalent", "PRESENT", "d" * 64)
    assert changed.machine_polarity_flip is False
    rows[1]["machine_polarity_flip"] = "1"                  # the machine's polarity call survives the round trip
    write_rows(path, header, rows)
    flipped = sheets.import_(path, "reveal", params=PARAMS, blind=blind)[1]
    assert flipped.machine_polarity_flip is True
    rows[1]["machine_polarity_flip"] = "0"
    assert (kept.final_relation, kept.blind_date, kept.stratum) == ("abridged", "2026-10-01", "pos:x:other")
    rows[0]["revised_reason"] = ""
    write_rows(path, header, rows)
    with pytest.raises(VerdictError, match="revised_reason"):
        sheets.import_(path, "reveal", params=PARAMS, blind=blind)


def test_import_gold(tmp_path: Path):
    _, cells = machine()
    items = [make_item("gold", r.id) for r in REFS[:2]]
    ref_path, wit_path = sheets.export(items, "gold", tmp_path, SEGMENTS, cells, batch="test_w01",
                                       witness_chapters=["pin1"])[:2]
    header, rows = read(ref_path)
    rows[0].update(links="z0001 z0002", relation="abridged")
    rows[1].update(links="", relation="no_counterpart", flags="unsure")
    write_rows(ref_path, header, rows)
    wheader, wrows = read(wit_path)
    wrows[5]["witness_only"] = "addition"
    write_rows(wit_path, wheader, wrows)
    out = sheets.import_(ref_path, "gold", params=PARAMS, annotator="ab", date="2026-10-02")
    assert [(v.unit_id, v.blind_relation, v.blind_wit_ids) for v in out] == [
        (REFS[0].id, "abridged", (WITS[0].id, WITS[1].id)), (REFS[1].id, "no_counterpart", ()),
        ("+" + WITS[5].id, "addition", (WITS[5].id,))]
    assert all(v.task == "gold" and v.batch_id == "test_w01" and v.final_relation == "" for v in out)
    assert out[2].fingerprint == WITS[5].fingerprint and out[0].quote_zh == WITS[0].text + " " + WITS[1].text[:21]
    rows[1]["relation"] = ""
    rows[0]["links"] = "z0099"
    write_rows(ref_path, header, rows)
    with pytest.raises(VerdictError) as exc:
        sheets.import_(ref_path, "gold", params=PARAMS)
    assert "unknown handle" in str(exc.value) and "gold windows must be complete" in str(exc.value)


def test_imported_verdicts_flow_into_the_matrix(tmp_path: Path):
    path, items = _filled_blind(tmp_path, task="resolve")
    verdicts = sheets.import_(path, "resolve", params=PARAMS, date="2026-10-03")
    alignment, _ = machine()
    grades = {r.id: Grade.B for r in REFS} | {"+" + WITS[9].id: Grade.C}
    cells, orphans, stale = build_cells(REFS, alignment, grades, {}, (), verdicts, WITS, [(("I.1",), ("pin1",))])
    c = next(c for c in cells if c.unit_id == REFS[4].id)
    assert (c.status, c.grade, c.source) == (Status.PARTIAL, Grade.A, "verdict:b7")
    assert orphans[0].grade is Grade.A and stale == []


# --------------------------------------------------------------------------- topics import
def test_import_topics_and_merge(tmp_path: Path, repo_root: Path):
    codebook = load_codebook(repo_root / "data" / "codebook" / "topics.yaml")
    items = [make_item("topics", REFS[2].id), make_item("topics", REFS[3].id)]
    pre = {REFS[2].id: Prelabel(REFS[2].id, frozenset({"harm"}), cues=(("harm", "c"),))}
    first = sheets.export(items, "topics", tmp_path, SEGMENTS, [], batch="t1", prelabels=pre)[0]
    header, rows = read(first)
    rows[0]["topics"] = "harm;ritual"
    write_rows(first, header, rows)
    labels = topic_sheets.import_topics(first, codebook, coder="ab", date="2026-10-01", prelabels=pre)
    assert labels == [TopicLabel(REFS[2].id, REFS[2].fingerprint, topics=frozenset({"harm", "ritual"}),
                                 prelabel_topics=frozenset({"harm"}), coder="ab", date="2026-10-01")]
    second = sheets.export(items, "topics_second", tmp_path, SEGMENTS, [], batch="t1")[0]
    header, rows = read(second)
    rows[0]["topics"] = "harm"
    write_rows(second, header, rows)
    seconds = topic_sheets.import_topics(second, codebook, coder="cd", date="2026-10-02")
    merged = topic_sheets.merge_topic_labels(topic_sheets.merge_topic_labels({}, labels), seconds)
    m = merged[REFS[2].id]
    assert (m.topics, m.second_topics, m.coder, m.second_coder) == ({"harm", "ritual"}, {"harm"}, "ab", "cd")
    with pytest.raises(VerdictError, match="without a first label"):
        topic_sheets.merge_topic_labels({}, seconds)
    rows[0]["topics"] = "gossip"
    write_rows(second, header, rows)
    with pytest.raises(VerdictError, match="unknown topic"):
        topic_sheets.import_topics(second, codebook, coder="cd", date="2026-10-02")
