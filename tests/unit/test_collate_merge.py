"""collate.merge: V9 witness exhaustiveness, V10 overlap agreement, chunk and text merging."""

from __future__ import annotations

import pytest

from hevajra_matrix.collate.collator import Unresolved
from hevajra_matrix.collate.merge import OverlapReport, compare_overlaps, merge_chapter, merge_text, verify_all
from hevajra_matrix.collate.verify import FLAG_OVERLAP_DISAGREEMENT, verify
from hevajra_matrix.core.types import Relation, WitnessOnlyKind

from test_collate_support import SOURCE, gold_answer, lexicon, raw, unit, windows

Z_ADDITION = "ZT:0001a08.1"
PIN2 = ("ZT:0001b01.1", "ZT:0001b02.1", "ZT:0001b03.1", "ZT:0001b04.1", "ZT:0001b05.1")


def chunks():
    """I.1 in two chunks of 4 units sharing 2: units 0-3 and 2-5."""
    ws = [w for w in windows(max_ref_units=4, overlap=2) if w.chapter == "I.1"]
    assert [len(w.units) for w in ws] == [4, 4] and ws[0].overlap_next == 2
    return ws


def verified(w, answer):
    return verify(raw(w, answer), w, lexicon(), SOURCE)


def gold_results(ws):
    return [verified(w, gold_answer(w)) for w in ws]


def replace_unit(w, answer, uid, **kwargs):
    answer["units"] = [unit(w, uid, **kwargs) if u["ref"] == w.handle_of[uid] else u for u in answer["units"]]
    return answer


# --------------------------------------------------------------------------- V10 overlap
def test_v10_agreeing_overlap() -> None:
    ws = chunks()
    report = compare_overlaps(ws, gold_results(ws))
    assert report == OverlapReport(compared=2, agreed=2, disagreeing=())
    assert report.rate == 1.0


def test_v10_disagreeing_overlap_is_counted_flagged_and_queued() -> None:
    ws = chunks()
    uid = ws[1].units[0].id                                    # first shared unit
    second = replace_unit(ws[1], gold_answer(ws[1]), uid, wit=[], relation=Relation.NO_COUNTERPART)
    results = [verified(ws[0], gold_answer(ws[0])), verified(ws[1], second)]
    report = compare_overlaps(ws, results)
    assert (report.compared, report.agreed, report.disagreeing) == (2, 1, (uid,))
    assert report.rate == 0.5
    merged = merge_chapter(ws, results)
    assert FLAG_OVERLAP_DISAGREEMENT in merged.alignment.by_ref()[uid].flags
    assert any(d.kind == FLAG_OVERLAP_DISAGREEMENT and d.ref_ids == (uid,) for d in merged.diagnostics)


def test_v10_same_class_but_other_segments_disagree() -> None:
    ws = chunks()
    uid = ws[1].units[1].id
    other = replace_unit(ws[1], gold_answer(ws[1]), uid, wit=[Z_ADDITION], relation=Relation.EQUIVALENT)
    report = compare_overlaps(ws, [verified(ws[0], gold_answer(ws[0])), verified(ws[1], other)])
    assert report.disagreeing == (uid,)


def test_v10_unresolved_readings_are_not_compared() -> None:
    ws = chunks()
    results = [verified(ws[0], gold_answer(ws[0])), verify(Unresolved(ws[1].key, "truncated"), ws[1], lexicon(),
                                                           SOURCE)]
    assert compare_overlaps(ws, results) == OverlapReport()
    assert OverlapReport().rate is None


def test_overlap_reports_add_up() -> None:
    a, b = OverlapReport(2, 1, ("x",)), OverlapReport(3, 3, ())
    assert a + b == OverlapReport(5, 4, ("x",))


# --------------------------------------------------------------------------- merge_chapter
def test_merge_chapter_reads_each_overlap_unit_from_the_chunk_where_it_is_central() -> None:
    ws = chunks()
    u2, u3 = ws[1].units[0].id, ws[1].units[1].id           # third and fourth unit of the chapter
    first = replace_unit(ws[0], gold_answer(ws[0]), u3, wit=[], relation=Relation.NO_COUNTERPART)
    second = replace_unit(ws[1], gold_answer(ws[1]), u2, wit=[], relation=Relation.NO_COUNTERPART)
    merged = merge_chapter(ws, [verified(ws[0], first), verified(ws[1], second)])
    by_ref = merged.alignment.by_ref()
    assert by_ref[u2].relation is not Relation.NO_COUNTERPART   # from chunk 1 (central there)
    assert by_ref[u3].relation is not Relation.NO_COUNTERPART   # from chunk 2 (central there)
    assert list(by_ref) == [u.id for u in ws[0].units] + [u.id for u in ws[1].units[2:]]


def test_merge_chapter_falls_back_to_the_other_chunk_when_one_is_unresolved() -> None:
    ws = chunks()
    results = [verify(Unresolved(ws[0].key, "refused:bio"), ws[0], lexicon(), SOURCE),
               verified(ws[1], gold_answer(ws[1]))]
    merged = merge_chapter(ws, results)
    shared = {u.id for u in ws[1].units[:2]}
    assert shared <= set(merged.alignment.by_ref())
    assert set(merged.unresolved) == {u.id for u in ws[0].units[:2]}
    assert set(merged.unresolved.values()) == {"refused:bio"}


def test_v9_unaccounted_core_segment_is_a_diagnostic() -> None:
    (w, _) = windows()
    answer = gold_answer(w)
    answer["witness_only"] = [x for x in answer["witness_only"] if x["wit"] != [w.handle_of[Z_ADDITION]]]
    merged = merge_chapter([w], [verified(w, answer)])
    diags = [d for d in merged.diagnostics if d.kind == "unaccounted"]
    assert len(diags) == 1 and diags[0].wit_ids == (Z_ADDITION,)


def test_v9_is_checked_after_merging_all_chunks() -> None:
    ws = chunks()
    # each chunk accounts only for what it links plus the witness-only material
    answers = []
    for w in ws:
        a = gold_answer(w)
        linked = {h for u in a["units"] for h in u["wit"]}
        a["witness_only"] = [x for x in a["witness_only"] if x["kind"] != "belongs_elsewhere" or
                             not set(x["wit"]) - linked]
        answers.append(a)
    merged = merge_chapter(ws, [verified(w, a) for w, a in zip(ws, answers)])
    assert not [d for d in merged.diagnostics if d.kind == "unaccounted"]


def test_v9_whole_chapter_unaccounted_when_the_window_failed() -> None:
    (w, _) = windows()
    merged = merge_chapter([w], [verify(Unresolved(w.key, "truncated"), w, lexicon(), SOURCE)])
    (diag,) = [d for d in merged.diagnostics if d.kind == "unaccounted"]
    assert set(diag.wit_ids) == set(w.core)


def test_merge_chapter_rejects_inconsistent_inputs() -> None:
    ws = chunks()
    with pytest.raises(ValueError):
        merge_chapter(ws, gold_results(ws)[:1])
    with pytest.raises(ValueError):
        merge_chapter(ws[::-1], gold_results(ws[::-1]))
    w1, w2 = windows()
    with pytest.raises(ValueError):
        merge_chapter([w1, w2], gold_results([w1, w2]))


# --------------------------------------------------------------------------- merge_text
def test_merge_text_reconciles_belongs_elsewhere_across_chapters() -> None:
    ws = windows(neighbours=1)                     # I.1 sees pin1+pin2, I.2 sees pin1..pin3
    results = [verified(w, gold_answer(w)) for w in ws]
    i1 = results[0].alignment.witness_only()
    assert {x.wit_ids[0] for x in i1 if x.relation is WitnessOnlyKind.BELONGS_ELSEWHERE} >= set(PIN2[1:])
    text = merge_text([merge_chapter([w], [r]) for w, r in zip(ws, results)])
    wo = {i: x.relation for x in text.alignment.witness_only() for i in x.wit_ids}
    assert not set(PIN2[1:]) & set(wo), "linked by I.2, so no longer witness-only"
    # pin3 is declared belongs_elsewhere by I.2 and linked by nothing: V9 over the text
    diag = next(d for d in text.diagnostics if d.kind == "unaccounted" and "belongs_elsewhere" in d.detail)
    assert set(diag.wit_ids) == {"ZT:0001c01.1", "ZT:0001c02.1", "ZT:0001c03.1"}
    assert len(text.alignment.by_ref()) == 10


def test_merge_text_rejects_mixed_sources_and_repeated_units() -> None:
    ws = windows()
    merged = [merge_chapter([w], [verified(w, gold_answer(w))]) for w in ws]
    with pytest.raises(ValueError):
        merge_text([merged[0], merged[0]])
    with pytest.raises(ValueError):
        merge_text([])
    other = merge_chapter([ws[1]], [verify(raw(ws[1], gold_answer(ws[1])), ws[1], lexicon(), "other")])
    with pytest.raises(ValueError):
        merge_text([merged[0], other])


# --------------------------------------------------------------------------- verify_all
def test_verify_all_one_replicate_end_to_end() -> None:
    ws = windows(max_ref_units=4, overlap=2)
    parsed = [raw(w, gold_answer(w)) for w in ws]
    result, overlap = verify_all(ws, parsed, lexicon(), SOURCE)
    assert overlap == OverlapReport(2, 2, ())
    assert len(result.alignment.by_ref()) == 10 and dict(result.unresolved) == {}
    assert [link.ref_id for link in result.alignment.links if link.ref_id] == [
        "BT:1a.1.1", "BT:1a.1.2", "BT:1a.2.1", "BT:1a.2.2", "BT:1a.3.1", "BT:1a.3.2",
        "BT:1b.1.1", "BT:1b.1.2", "BT:1b.2.1", "BT:1b.2.2"]
    with pytest.raises(ValueError):
        verify_all(ws, parsed[:-1], lexicon(), SOURCE)
