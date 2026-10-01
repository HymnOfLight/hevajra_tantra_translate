"""The 8 sentinel check kinds on synthetic texts (evaluation.sentinels)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hevajra_matrix.core.types import Alignment, Cell, Grade, Link, Relation, Segment, Status, WitnessOnlyKind
from hevajra_matrix.evaluation import sentinels as S

DATA = Path(__file__).resolve().parents[2] / "data"


def seg(uid: str, text: str, kind: str = "verse_line", local: str | None = None, **extra) -> Segment:
    line = uid.split(":", 1)[1].rsplit(".", 1)[0]
    return Segment(id=uid, witness="w", lang="en", text=text, start=line, end=line, kind=kind,
                   local_chapter=local, extra=extra)


REF = [
    seg("D417:1a.1.1", "alpha beta"),
    seg("D417:1a.1.2", "gamma delta"),          # shares line 1a.1: quotes pick the segment
    seg("D417:1a.2.1", "epsilon"),
    seg("D417:1a.3.1", "zeta"),
    seg("D417:1a.4.1", "colophon one", kind="paratext"),
    seg("D417:1a.4.2", "colophon two", kind="paratext"),
]
WIT = [
    seg("T0892:0001a01.1", "one", kind="prose", local="pin1"),
    seg("T0892:0001a01.n1", "sanskrit lacks this", kind="note", local="pin1", note_class="vorlage_statement"),
    seg("T0892:0001a02.1", "two", kind="prose", local="pin1"),
    seg("T0892:0001a03.1", "three", kind="prose", local="pin2"),
    seg("T0892:0001a04.1", "four", kind="prose", local="pin2"),
]
SEGMENTS = REF + WIT
EQ = Relation.EQUIVALENT

ALIGNMENT = Alignment("claude:consensus", "ref", "wit", (
    Link("D417:1a.1.1", ("T0892:0001a01.1",), EQ),
    Link("D417:1a.1.2", ("T0892:0001a02.1",), Relation.REVERSAL, polarity_flip=True, flags=frozenset({"reordered"})),
    Link("D417:1a.2.1", ("T0892:0001a03.1",), Relation.SUBSTITUTION),
    Link("D417:1a.3.1", (), Relation.NO_COUNTERPART),
    Link(None, ("T0892:0001a04.1",), WitnessOnlyKind.ADDITION),
))


def sentinel(check: str, loci: dict, expect: dict, quotes: dict | None = None, stage: str = "ingest",
             status: str = "verified") -> S.Sentinel:
    return S.Sentinel(id=f"X_{check}", status=status, stage=stage, check=check, loci=loci, quotes=quotes or {},
                      expect=expect, verified_by="test" if status == "verified" else None, source="test")


def run(s: S.Sentinel, stage: str | None = None, **kw) -> S.SentinelResult:
    (result,) = S.check([s], stage or s.stage, SEGMENTS, **kw)
    return result


# --------------------------------------------------------------------------- ingest-stage kinds
def test_note_kind():
    s = sentinel("note_kind", {"note": "T0892:0001a01"}, {"note_class": "vorlage_statement"}, {"note": "lacks"})
    assert run(s).passed
    assert not run(sentinel("note_kind", {"note": "T0892:0001a01"}, {"note_class": "title"})).passed


def test_paratext():
    s = sentinel("paratext", {"from": "D417:1a.4", "to": "D417:1a.4"}, {"count": 2})
    assert run(s).passed
    assert not run(sentinel("paratext", {"from": "D417:1a.3", "to": "D417:1a.4"}, {"count": 3})).passed
    assert not run(sentinel("paratext", {"from": "D417:1a.4", "to": "D417:1a.4"}, {"count": 3})).passed
    # per the schema only CONTENT segments may not follow: a shorter paratext range still passes
    assert run(sentinel("paratext", {"from": "D417:1a.4", "to": "D417:1a.4"}, {"count": 1},
                        {"to": "colophon one"})).passed


def test_paratext_must_end_the_text():
    segments = SEGMENTS + [seg("D417:1a.5.1", "late verse")]
    (result,) = S.check([sentinel("paratext", {"from": "D417:1a.4", "to": "D417:1a.4"}, {"count": 2})],
                        "ingest", segments)
    assert not result.passed and "1 content segment(s) after" in result.detail


def test_adjacent_counts_content_segments_between():
    s = sentinel("adjacent", {"first": "D417:1a.1", "second": "D417:1a.3"}, {"max_between": 1},
                 {"first": "gamma"})
    assert run(s).passed                       # 1a.1.2 -> 1a.3.1: only 1a.2.1 between
    s = sentinel("adjacent", {"first": "D417:1a.1", "second": "D417:1a.3"}, {"max_between": 1})
    result = run(s)                            # 1a.1.1 -> 1a.3.1: two between
    assert not result.passed and "2 content segment(s)" in result.detail


# --------------------------------------------------------------------------- alignment kinds
def test_relation_in_with_flags_polarity_and_witness_range():
    loci = {"ref_from": "D417:1a.1", "ref_to": "D417:1a.1", "wit_from": "T0892:0001a02", "wit_to": "T0892:0001a02"}
    quotes = {"ref_from": "gamma", "ref_to": "gamma"}
    ok = sentinel("relation_in", loci, {"relations": ["reversal"], "polarity_flip": True, "flags": ["reordered"]},
                  quotes, stage="proposal")
    assert run(ok, alignment=ALIGNMENT).passed
    no_flag = sentinel("relation_in", loci, {"relations": ["reversal"], "flags": ["scope_list"]}, quotes,
                       stage="proposal")
    assert not run(no_flag, alignment=ALIGNMENT).passed
    outside = dict(loci, wit_from="T0892:0001a03", wit_to="T0892:0001a03")
    assert not run(sentinel("relation_in", outside, {"relations": ["reversal"]}, quotes, stage="proposal"),
                   alignment=ALIGNMENT).passed
    both = sentinel("relation_in", {"ref_from": "D417:1a.1", "ref_to": "D417:1a.1"}, {"relations": ["reversal"]},
                    stage="proposal")
    result = run(both, alignment=ALIGNMENT)    # without quotes the range is the whole line: 1a.1.1 fails
    assert not result.passed and "D417:1a.1.1=equivalent" in result.detail


def test_status_in_and_none_status():
    loci = {"ref_from": "D417:1a.1", "ref_to": "D417:1a.2"}
    assert run(sentinel("status_in", loci, {"statuses": ["PRESENT"]}, stage="proposal"), alignment=ALIGNMENT).passed
    loci3 = {"ref_from": "D417:1a.1", "ref_to": "D417:1a.3"}
    assert not run(sentinel("status_in", loci3, {"statuses": ["PRESENT"]}, stage="proposal"),
                   alignment=ALIGNMENT).passed
    assert run(sentinel("none_status", loci, {"status": "ABSENT"}, stage="proposal"), alignment=ALIGNMENT).passed
    assert not run(sentinel("none_status", loci3, {"status": "ABSENT"}, stage="proposal"),
                   alignment=ALIGNMENT).passed


def test_unresolved_unit_is_unaligned():
    partial = Alignment("claude:consensus", "ref", "wit", ALIGNMENT.links[:3])
    loci = {"ref_from": "D417:1a.3", "ref_to": "D417:1a.3"}
    assert run(sentinel("none_status", loci, {"status": "ABSENT"}, stage="proposal"), alignment=partial).passed
    result = run(sentinel("status_in", loci, {"statuses": ["ABSENT"]}, stage="proposal"), alignment=partial)
    assert not result.passed and "UNALIGNED" in result.detail


def test_linked_local():
    loci = {"ref_from": "D417:1a.1", "ref_to": "D417:1a.1"}
    assert run(sentinel("linked_local", loci, {"local": "pin1"}, stage="proposal"), alignment=ALIGNMENT).passed
    assert not run(sentinel("linked_local", {"ref_from": "D417:1a.1", "ref_to": "D417:1a.2"}, {"local": "pin1"},
                            stage="proposal"), alignment=ALIGNMENT).passed
    # an unlinked unit fails
    assert not run(sentinel("linked_local", {"ref_from": "D417:1a.3", "ref_to": "D417:1a.3"}, {"local": "pin2"},
                            stage="proposal"), alignment=ALIGNMENT).passed


def test_witness_only():
    s = sentinel("witness_only", {"wit_from": "T0892:0001a04", "wit_to": "T0892:0001a04"}, {"kinds": ["addition"]},
                 stage="proposal")
    assert run(s, alignment=ALIGNMENT).passed
    linked = sentinel("witness_only", {"wit_from": "T0892:0001a03", "wit_to": "T0892:0001a04"},
                      {"kinds": ["addition"]}, stage="proposal")
    result = run(linked, alignment=ALIGNMENT)
    assert not result.passed and "T0892:0001a03.1=linked" in result.detail
    wrong_kind = sentinel("witness_only", {"wit_from": "T0892:0001a04", "wit_to": "T0892:0001a04"},
                          {"kinds": ["translator_note"]}, stage="proposal")
    assert not run(wrong_kind, alignment=ALIGNMENT).passed


# --------------------------------------------------------------------------- final stage reads cells
def test_final_stage_reads_cells_including_orphans():
    cells = [
        Cell("D417:1a.1.1", "wit", Status.PRESENT, Grade.A, relation="equivalent", wit_ids=("T0892:0001a01.1",)),
        Cell("D417:1a.3.1", "wit", Status.PRESENT, Grade.A, relation="paraphrase", wit_ids=("T0892:0001a02.1",)),
        Cell("+T0892:0001a04.1", "wit", Status.NA, Grade.B, relation="addition", wit_ids=("T0892:0001a04.1",)),
    ]
    absent = sentinel("none_status", {"ref_from": "D417:1a.3", "ref_to": "D417:1a.3"}, {"status": "ABSENT"},
                      stage="proposal")
    assert not run(absent, alignment=ALIGNMENT).passed           # the machine said ABSENT
    assert run(absent, stage="final", cells=cells).passed         # a verdict made it PRESENT
    only = sentinel("witness_only", {"wit_from": "T0892:0001a04", "wit_to": "T0892:0001a04"},
                    {"kinds": ["addition"]}, stage="final")
    assert run(only, cells=cells).passed


# --------------------------------------------------------------------------- stages, status, failures
def test_stage_filtering_and_retired():
    ingest = sentinel("paratext", {"from": "D417:1a.4", "to": "D417:1a.4"}, {"count": 2})
    final = sentinel("none_status", {"ref_from": "D417:1a.1", "ref_to": "D417:1a.1"}, {"status": "ABSENT"},
                     stage="final")
    retired = S.Sentinel("R", "retired", "ingest", "paratext", {"from": "D417:1a.4", "to": "D417:1a.4"},
                         expect={"count": 9})
    assert [r.sentinel_id for r in S.check([ingest, final, retired], "ingest", SEGMENTS)] == ["X_paratext"]
    results = S.check([ingest, final], "final", SEGMENTS, cells=[])
    assert [(r.sentinel_id, r.stage, r.sentinel_stage) for r in results] == [
        ("X_paratext", "final", "ingest"), ("X_none_status", "final", "final")]
    with pytest.raises(ValueError):
        S.check([ingest], "review", SEGMENTS)


def test_missing_input_and_unresolved_locus_fail():
    s = sentinel("status_in", {"ref_from": "D417:1a.1", "ref_to": "D417:1a.1"}, {"statuses": ["PRESENT"]},
                 stage="proposal")
    result = run(s)
    assert not result.passed and "no alignment" in result.detail
    ghost = sentinel("note_kind", {"note": "T0892:0009z99"}, {"note_class": "title"})
    result = run(ghost)
    assert not result.passed and "unresolved locus" in result.detail
    quoted = sentinel("note_kind", {"note": "T0892:0001a01"}, {"note_class": "vorlage_statement"},
                      {"note": "not in the note"})
    assert "with its quote" in run(quoted).detail


def test_only_verified_sentinels_block():
    failing = sentinel("note_kind", {"note": "T0892:0001a01"}, {"note_class": "title"})
    proposed = sentinel("note_kind", {"note": "T0892:0001a01"}, {"note_class": "title"}, status="proposed")
    results = S.check([failing, proposed], "ingest", SEGMENTS)
    assert [r.passed for r in results] == [False, False]
    assert [r.status for r in S.blocking_failures(results)] == ["verified"]


# --------------------------------------------------------------------------- the data file
def test_load_the_committed_sentinels():
    loaded = S.load(DATA / "sentinels" / "sentinels.yaml")
    assert len(loaded) == 18 and {s.check for s in loaded} == set(S.CHECKS)
    assert sum(s.blocking for s in loaded) == 11


def test_load_rejects_malformed_files(tmp_path: Path):
    path = tmp_path / "s.yaml"
    path.write_text("schema_version: 1\nsentinels: []\n", encoding="utf-8")
    with pytest.raises(S.SentinelError, match="schema_version"):
        S.load(path)
    path.write_text("schema_version: 2\nsentinels:\n  - {id: a, status: verified, verified_by: null, source: s, "
                    "stage: ingest, check: adjacent, loci: {}, quotes: {}, expect: {}}\n", encoding="utf-8")
    with pytest.raises(S.SentinelError, match="names who verified"):
        S.load(path)


def test_alignment_sentinels_apply_only_to_their_text_pair():
    """A fact stated for Derge -> Chinese is not checked against Sanskrit -> Chinese or
    Sanskrit -> Derge; segment-only checks always apply."""
    rel = sentinel("relation_in", {"ref_from": "D417:1a.1", "ref_to": "D417:1a.1", "wit_from": "T0892:0001a01",
                                   "wit_to": "T0892:0001a01"}, {"relations": ["equivalent"]}, stage="proposal")
    only = sentinel("witness_only", {"wit_from": "T0892:0001a04", "wit_to": "T0892:0001a04"},
                    {"kinds": ["addition"]}, stage="proposal")
    note = sentinel("note_kind", {"note": "T0892:0001a01"}, {"note_class": "vorlage_statement"})
    every = [rel, only, note]
    derge, chinese = S.text_prefixes(REF), S.text_prefixes(WIT)
    sanskrit = S.text_prefixes([Segment("I.1.1", "sa", "sa", "x", "I.1.1", "I.1.1", "verse")])
    assert (derge, chinese, sanskrit) == ({"D417"}, {"T0892"}, frozenset())
    assert S.applicable(every, derge, chinese) == every
    assert S.applicable(every, sanskrit, chinese) == [only, note]
    assert S.applicable(every, sanskrit, derge) == [note]
