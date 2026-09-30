"""matrix.build: status table, precedence, stale verdicts, d_* dimensions and orphan rows."""

from __future__ import annotations

import math

import pytest

from hevajra_matrix.core.textnorm import fingerprint
from hevajra_matrix.core.types import (
    Alignment,
    Grade,
    Link,
    OutcomeClass,
    Relation,
    Segment,
    Status,
    WitnessOnlyKind,
)
from hevajra_matrix.matrix.build import (
    REASON_HUMAN_UNRESOLVED,
    STALE_FINGERPRINT,
    STALE_MISSING,
    attribute_chapter,
    build_cells,
    order_displacement,
)
from hevajra_matrix.matrix.status import cell_outcome
from hevajra_matrix.review.verdicts import HAS_COUNTERPART, VerdictRecord

REF, WIT = "bo_ref", "zh_wit"


def seg(sid: str, text: str, *, witness: str = REF, lang: str = "bo", kind: str = "verse",
        chapter: str | None = None, local: str | None = None) -> Segment:
    return Segment(id=sid, witness=witness, lang=lang, text=text, start=sid.split(":")[1].rsplit(".", 1)[0],
                   end="", kind=kind, local_chapter=local, chapter=chapter, fingerprint=fingerprint(text, lang))


def wseg(sid: str, text: str, local: str = "pin1", kind: str = "verse") -> Segment:
    return seg(sid, text, witness=WIT, lang="zh", kind=kind, local=local)


# Reference chapter I.1: four units; witness chapter pin1: five segments.
REFS = [seg(f"D417:1a.1.{i}", f"ref unit number {i} text", chapter="I.1") for i in range(1, 5)]
WITS = [wseg(f"T0892:0587a0{i}.1", f"wit seg {i}") for i in range(1, 6)]
GROUPS = [(("I.1",), ("pin1",))]


def link(ref_id: str | None, wit_ids: tuple[str, ...], relation, flip: bool = False, flags=frozenset()) -> Link:
    return Link(ref_id, wit_ids, relation, polarity_flip=flip, flags=frozenset(flags), source="claude:consensus")


def alignment(*links: Link) -> Alignment:
    return Alignment("claude:consensus", REF, WIT, tuple(links))


def verdict(unit: Segment | str, value: str, wit_ids=(), *, task="verify", batch="b1", date="2026-10-01",
            fp: str | None = None, final=True, quote_ref="") -> VerdictRecord:
    uid = unit if isinstance(unit, str) else unit.id
    fpr = fp if fp is not None else ("" if isinstance(unit, str) else unit.fingerprint)
    cols = dict(final_relation=value, final_wit_ids=tuple(wit_ids), final_date=date) if final else {}
    return VerdictRecord(batch_id=batch, item_id=f"{task}:{uid}", task=task, unit_id=uid, fingerprint=fpr,
                         blind_relation=value, blind_wit_ids=tuple(wit_ids), blind_date=date,
                         polarity_flip=value == "reversal", quote_ref=quote_ref, **cols)


def build(links=(), grades=None, reasons=None, gold=(), verdicts=(), refs=REFS, wits=WITS, groups=GROUPS):
    grades = grades if grades is not None else {l.ref_id: Grade.B for l in links if l.ref_id}
    return build_cells(refs, alignment(*links), grades, reasons or {}, gold, verdicts, wits, groups)


def cell_of(cells, uid):
    return next(c for c in cells if c.unit_id == uid)


# --------------------------------------------------------------------------- status table (4.1)
@pytest.mark.parametrize("relation, flip, kind, status, outcome", [
    ("equivalent", False, "verse", Status.PRESENT, OutcomeClass.NONDEV),
    ("paraphrase", False, "verse", Status.PRESENT, OutcomeClass.NONDEV),
    ("expanded", False, "verse", Status.PRESENT, OutcomeClass.NONDEV),
    ("transliterated", False, "mantra", Status.PRESENT, OutcomeClass.NONDEV),
    ("transliterated", False, "prose", Status.PRESENT, OutcomeClass.DEV_PRESENT),
    ("generalised", False, "verse", Status.PRESENT, OutcomeClass.DEV_PRESENT),
    ("substitution", False, "verse", Status.PRESENT, OutcomeClass.DEV_PRESENT),
    ("category_name_omitted", False, "verse", Status.PRESENT, OutcomeClass.DEV_PRESENT),
    ("reversal", True, "verse", Status.PRESENT, OutcomeClass.DEV_PRESENT),
    ("abridged", False, "verse", Status.PARTIAL, OutcomeClass.PARTIAL),
    ("no_counterpart", False, "verse", Status.ABSENT, OutcomeClass.ABSENT),
])
def test_status_table_in_built_cells(relation, flip, kind, status, outcome):
    unit = seg("D417:1a.1.1", "ref", chapter="I.1", kind=kind)
    wit_ids = () if relation == "no_counterpart" else (WITS[0].id,)
    cells, _, _ = build([link(unit.id, wit_ids, Relation(relation), flip)], refs=[unit])
    c = cells[0]
    assert (c.status, c.relation, c.grade) == (status, relation, Grade.B)
    assert cell_outcome(c, kind) is outcome


def test_length_never_decides_status():
    long_ref = seg("D417:1a.1.1", "a " * 60, chapter="I.1")
    cells, _, _ = build([link(long_ref.id, (WITS[0].id,), Relation.EQUIVALENT)], refs=[long_ref])
    assert cells[0].status is Status.PRESENT and cells[0].d_len < -2


def test_unaligned_cells_always_carry_a_reason():
    cells, _, _ = build([link(REFS[0].id, (WITS[0].id,), Relation.EQUIVALENT)], reasons={REFS[1].id: "refused:bio"})
    assert cell_of(cells, REFS[1].id).reason == "refused:bio"
    assert cell_of(cells, REFS[2].id).reason == "unassessed"
    for c in cells[1:]:
        assert c.status is Status.UNALIGNED and c.grade is Grade.X and c.reason


def test_consensus_link_without_grade_is_an_error():
    with pytest.raises(ValueError, match="without a grade"):
        build([link(REFS[0].id, (WITS[0].id,), Relation.EQUIVALENT)], grades={})


def test_baselines_never_populate_the_matrix():
    control = Alignment("dp:anchor", REF, WIT, (link(REFS[0].id, (WITS[0].id,), Relation.EQUIVALENT),))
    with pytest.raises(ValueError, match="control alignment"):
        build_cells(REFS, control, {REFS[0].id: Grade.B}, {}, (), (), WITS, GROUPS)


# --------------------------------------------------------------------------- precedence (4.4)
def test_precedence_gold_over_verdict_over_consensus():
    u0, u1, u2 = REFS[:3]
    links = [link(u.id, (WITS[i].id,), Relation.EQUIVALENT) for i, u in enumerate((u0, u1, u2))]
    gold = [verdict(u0, "no_counterpart", task="gold", batch="test_w01", final=False)]
    verdicts = [verdict(u0, "abridged", (WITS[0].id,)), verdict(u1, "substitution", (WITS[1].id,))]
    cells, _, stale = build(links, gold=gold, verdicts=verdicts)
    c0, c1, c2 = cells[:3]
    assert (c0.status, c0.grade, c0.source) == (Status.ABSENT, Grade.A, "gold:test_w01")
    assert (c1.relation, c1.grade, c1.source) == ("substitution", Grade.A, "verdict:b1")
    assert (c2.relation, c2.grade, c2.source) == ("equivalent", Grade.B, "claude:consensus")
    assert stale == []


def test_verdict_without_final_columns_is_not_applied():
    links = [link(REFS[0].id, (WITS[0].id,), Relation.EQUIVALENT)]
    cells, _, _ = build(links, verdicts=[verdict(REFS[0], "abridged", (WITS[0].id,), final=False)])
    assert cells[0].grade is Grade.B and cells[0].relation == "equivalent"


def test_latest_final_verdict_wins():
    older = verdict(REFS[0], "abridged", (WITS[0].id,), batch="old", date="2026-10-01")
    newer = verdict(REFS[0], "generalised", (WITS[0].id,), batch="new", date="2026-10-05")
    cells, _, _ = build(verdicts=[newer, older])
    assert (cells[0].relation, cells[0].source) == ("generalised", "verdict:new")


def test_stale_verdict_is_listed_and_not_applied():
    links = [link(REFS[0].id, (WITS[0].id,), Relation.EQUIVALENT)]
    stale_v = verdict(REFS[0], "no_counterpart", fp="000000000000", quote_ref="old words")
    ghost = verdict("D417:9a.1.1", "no_counterpart", fp="111111111111")
    cells, _, stale = build(links, verdicts=[stale_v, ghost])
    assert cells[0].relation == "equivalent" and cells[0].grade is Grade.B
    assert [(s.unit_id, s.reason) for s in stale] == [(REFS[0].id, STALE_FINGERPRINT), ("D417:9a.1.1", STALE_MISSING)]
    assert stale[0].current_fingerprint == REFS[0].fingerprint and stale[0].quote_ref == "old words"


def test_stale_gold_is_listed_too():
    _, _, stale = build(gold=[verdict(REFS[0], "equivalent", (WITS[0].id,), task="gold", fp="0" * 12, final=False)])
    assert [s.task for s in stale] == ["gold"]


def test_human_lacuna_and_unresolved():
    cells, _, _ = build(verdicts=[verdict(REFS[0], "lacuna"), verdict(REFS[1], "unresolved", task="resolve")])
    assert (cells[0].status, cells[0].grade) == (Status.LACUNA, Grade.A)
    assert (cells[1].status, cells[1].reason, cells[1].grade) == (Status.UNALIGNED, REASON_HUMAN_UNRESOLVED, Grade.A)


def test_invalid_human_value_is_an_error():
    with pytest.raises(ValueError, match="not a relation"):
        build(verdicts=[verdict(REFS[0], "addition", (WITS[0].id,))])


# --------------------------------------------------------------------------- d_len, d_lit
def test_d_len_is_log_length_ratio():
    ref = seg("D417:1a.1.1", "ka kha ga nga", chapter="I.1")             # 4 syllables
    wit = wseg("T0892:0587a01.1", "ab")                                   # 2 characters
    cells, _, _ = build([link(ref.id, (wit.id,), Relation.EQUIVALENT)], refs=[ref], wits=[wit])
    assert math.isclose(cells[0].d_len, math.log(2 / 4))
    assert cells[0].d_ord == 0.0


def test_d_lit_uses_the_witness_mantra_charset():
    mantra = wseg("T0892:0587a01.1", "xyz", kind="mantra")
    prose = wseg("T0892:0587a02.1", "xxab")
    ref = seg("D417:1a.1.1", "ref", chapter="I.1")
    cells, _, _ = build([link(ref.id, (prose.id,), Relation.TRANSLITERATED)], refs=[ref], wits=[mantra, prose])
    assert cells[0].d_lit == pytest.approx(0.5)


def test_no_counterpart_has_no_dimensions():
    cells, _, _ = build([link(REFS[0].id, (), Relation.NO_COUNTERPART)])
    assert (cells[0].d_len, cells[0].d_lit, cells[0].d_ord) == (None, None, None)


# --------------------------------------------------------------------------- d_ord (#6)
MERGED_REFS = ([seg(f"D417:30a.{i}.1", f"late I.11 unit {i}", chapter="I.11") for i in range(1, 5)]
               + [seg(f"D418:1a.{i}.1", f"early II.1 unit {i}", chapter="II.1") for i in range(1, 5)])
MERGED_WITS = [wseg(f"T0892:0595a{10 + i}.1", f"merged seg {i}", local="pin11") for i in range(10)]
MERGED_GROUPS = [(("I.11", "II.1"), ("pin11",))]


def test_d_ord_is_zero_for_an_in_order_alignment_of_a_merged_group():
    # 8 units over 10 segments, in order, with two witness-only segments in between.
    targets = [0, 1, 2, 3, 5, 6, 8, 9]
    links = [link(u.id, (MERGED_WITS[t].id,), Relation.EQUIVALENT) for u, t in zip(MERGED_REFS, targets)]
    cells, _, _ = build(links, refs=MERGED_REFS, wits=MERGED_WITS, groups=MERGED_GROUPS)
    assert [c.d_ord for c in cells] == [0.0] * 8


def test_d_ord_measures_a_transposition_and_ignores_n_to_1_ties():
    targets = [0, 0, 2, 1, 4, 5, 6, 7]         # units 0-1 share a segment; units 2 and 3 swapped
    links = [link(u.id, (MERGED_WITS[t].id,), Relation.EQUIVALENT) for u, t in zip(MERGED_REFS, targets)]
    cells, _, _ = build(links, refs=MERGED_REFS, wits=MERGED_WITS, groups=MERGED_GROUPS)
    assert [round(c.d_ord * 7) for c in cells] == [0, 0, 1, -1, 0, 0, 0, 0]


def test_d_ord_is_none_for_a_link_outside_the_group():
    other = wseg("T0892:0599a01.1", "elsewhere", local="pin18")
    links = [link(MERGED_REFS[0].id, (other.id,), Relation.EQUIVALENT),
             link(MERGED_REFS[1].id, (MERGED_WITS[0].id,), Relation.EQUIVALENT)]
    cells, _, _ = build(links, refs=MERGED_REFS, wits=MERGED_WITS + [other],
                        groups=MERGED_GROUPS + [(("II.9",), ("pin18",))])
    assert cells[0].d_ord is None and cells[1].d_ord == 0.0


def test_order_displacement_range():
    d = order_displacement(["a", "b", "c"], {"a": 9, "b": 1, "c": 5})
    assert d == {"a": 1.0, "b": -0.5, "c": -0.5}
    assert order_displacement(["a"], {"a": 3}) == {"a": 0.0}


def test_groups_must_not_share_keys():
    with pytest.raises(ValueError, match="more than one group"):
        build(groups=[(("I.1",), ("pin1",)), (("I.2",), ("pin1",))])


# --------------------------------------------------------------------------- orphans (#14, B15)
def test_two_orphans_on_one_line_give_two_rows():
    a, b = wseg("T0892:0601c01.1", "gatha one", local="pin20"), wseg("T0892:0601c01.2", "gatha two", local="pin20")
    ref = seg("D418:30a.1.1", "last unit", chapter="II.12")
    links = [link(ref.id, (), Relation.NO_COUNTERPART),
             link(None, (a.id,), WitnessOnlyKind.ADDITION), link(None, (b.id,), WitnessOnlyKind.ADDITION)]
    grades = {ref.id: Grade.B, "+" + a.id: Grade.B, "+" + b.id: Grade.C}
    _, orphans, _ = build(links, grades=grades, refs=[ref], wits=[a, b], groups=[(("II.11", "II.12"), ("pin20",))])
    assert [o.unit_id for o in orphans] == ["+T0892:0601c01.1", "+T0892:0601c01.2"]
    assert [o.grade for o in orphans] == [Grade.B, Grade.C]
    assert all(o.status is Status.NA and o.relation == "addition" and o.wit_ids for o in orphans)
    assert len({o.unit_id for o in orphans}) == 2


def test_orphan_in_merged_group_is_attributed_to_the_nearest_linked_neighbour():
    # pin11 = I.11 + II.1; the orphan sits between a segment linked to I.11 (two before) and
    # one linked to II.1 (one after): the nearest neighbour decides.
    w = MERGED_WITS
    links = [link(MERGED_REFS[3].id, (w[2].id,), Relation.EQUIVALENT),
             link(MERGED_REFS[4].id, (w[5].id,), Relation.EQUIVALENT),
             link(None, (w[4].id,), WitnessOnlyKind.ADDITION),
             link(None, (w[3].id,), WitnessOnlyKind.TRANSLATOR_NOTE)]
    grades = {MERGED_REFS[3].id: Grade.B, MERGED_REFS[4].id: Grade.B, "+" + w[4].id: Grade.B, "+" + w[3].id: Grade.B}
    cells, orphans, _ = build(links, grades=grades, refs=MERGED_REFS, wits=w, groups=MERGED_GROUPS)
    chapters = {o.unit_id: o.chapter for o in orphans}
    assert chapters == {"+" + w[3].id: "I.11", "+" + w[4].id: "II.1"}   # tie at w[3] goes to the preceding side
    assert attribute_chapter(w[4].id, w, cells, MERGED_GROUPS) == "II.1"


def test_orphan_in_single_chapter_group_takes_that_chapter():
    links = [link(None, (WITS[0].id,), WitnessOnlyKind.ADDITION)]
    _, orphans, _ = build(links, grades={"+" + WITS[0].id: Grade.B})
    assert orphans[0].chapter == "I.1"


def test_human_verdicts_on_orphan_rows():
    links = [link(None, (WITS[0].id,), WitnessOnlyKind.ADDITION), link(None, (WITS[1].id,), WitnessOnlyKind.ADDITION)]
    grades = {"+" + WITS[0].id: Grade.B, "+" + WITS[1].id: Grade.B}
    orphan_fp = {w.id: w.fingerprint for w in WITS}
    verdicts = [
        verdict("+" + WITS[0].id, HAS_COUNTERPART, fp=orphan_fp[WITS[0].id]),              # claim rejected
        verdict("+" + WITS[1].id, "translator_note", (WITS[1].id,), fp=orphan_fp[WITS[1].id]),
        verdict("+" + WITS[2].id, "paratext", (WITS[2].id,), fp=orphan_fp[WITS[2].id]),     # new human claim
    ]
    _, orphans, stale = build(links, grades=grades, verdicts=verdicts)
    assert [(o.unit_id, o.relation, o.grade) for o in orphans] == [
        ("+" + WITS[1].id, "translator_note", Grade.A), ("+" + WITS[2].id, "paratext", Grade.A)]
    assert stale == []


def test_segment_linked_by_a_verdict_is_no_longer_an_orphan():
    links = [link(None, (WITS[0].id,), WitnessOnlyKind.ADDITION)]
    _, orphans, _ = build(links, grades={"+" + WITS[0].id: Grade.B},
                          verdicts=[verdict(REFS[0], "equivalent", (WITS[0].id,))])
    assert orphans == []
