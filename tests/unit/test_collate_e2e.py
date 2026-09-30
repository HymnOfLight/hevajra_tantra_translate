"""End to end on the fixture texts with the FakeClient echo collator:
plan -> build requests -> collate (k replicates) -> verify and merge -> consensus,
plus offline replay from the response cache and the two G2 perturbations."""

from __future__ import annotations

from pathlib import Path

import pytest

from hevajra_matrix.collate import (
    agreement,
    collate,
    consensus,
    replicate_tags,
    source_name,
    verify_all,
)
from hevajra_matrix.collate.perturb import deletion_recall, delete_segments, false_link_rate, wrong_window
from hevajra_matrix.core.types import Grade, Relation, WitnessOnlyKind
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import CacheMiss, LLMRequest
from hevajra_matrix.llm.fake import FakeClient, ok_response, refusal_response

from test_collate_support import (
    SETTINGS,
    echo_script,
    examples,
    gold,
    gold_witness_only,
    lexicon,
    reference,
    template,
    windows,
)

REF_KINDS = {s.id: s.kind for s in reference() if s.kind != "colophon"}


def run_pipeline(client, ws, k: int = 3):
    parsed = collate(ws, client, SETTINGS, replicate_tags(k), examples(), template(), workers=3)
    replicates, overlaps = [], []
    for tag, results in parsed.items():
        verified, overlap = verify_all(ws, results, lexicon(), source_name(SETTINGS.model, tag))
        replicates.append(verified)
        overlaps.append(overlap)
    return replicates, overlaps, consensus(replicates, REF_KINDS)


def test_echo_collator_reproduces_the_gold_with_grade_b() -> None:
    ws = windows(max_ref_units=4, overlap=2, neighbours=1)
    replicates, overlaps, (alignment, grades, reasons) = run_pipeline(FakeClient(echo_script), ws)
    assert reasons == {}
    g = gold()
    assert {uid: (link.wit_ids, link.relation) for uid, link in alignment.by_ref().items()} == g
    assert all(grades[uid] is Grade.B for uid in g)
    assert alignment.by_ref()["BT:1a.3.2"].flags == frozenset({"corroborated"})
    wo = {x.wit_ids[0]: x.relation for x in alignment.witness_only()}
    expected = {sid: kind for sid, kind in gold_witness_only().items()}
    assert {sid: wo[sid] for sid in expected} == expected
    # pin3 is declared belongs_elsewhere (never linked): it stays, and V9 reports it
    assert {sid for sid, kind in wo.items() if kind is WitnessOnlyKind.BELONGS_ELSEWHERE} == {
        "ZT:0001c01.1", "ZT:0001c02.1", "ZT:0001c03.1"}
    assert all(o.rate == 1.0 for o in overlaps)
    stats = agreement(replicates, REF_KINDS)
    assert stats.mean_link_jaccard == 1.0 and stats.units == len(g)


def test_disagreeing_and_refusing_replicates() -> None:
    ws = windows()
    horses = next(s for s in reference() if s.id == "BT:1a.3.2")
    first_of_i2 = next(s for s in reference() if s.id == "BT:1b.1.1")

    def script(request: LLMRequest):
        answer = echo_script(request)
        if request.replicate == "r2":
            for u in answer["units"]:
                if u["ref_quote"] == horses.text:
                    u.update(wit=[], relation=Relation.NO_COUNTERPART.value, polarity_flip=False, wit_quote="")
        if request.replicate == "r3" and first_of_i2.text in request.body:
            return refusal_response(request, category="other")
        return ok_response(request, answer)

    replicates, _, (alignment, grades, reasons) = run_pipeline(FakeClient(script), ws)
    assert alignment.by_ref()["BT:1a.3.2"].relation is Relation.REVERSAL
    assert grades["BT:1a.3.2"] is Grade.C, "2 of 3"
    i2 = [u for u, kind in REF_KINDS.items() if u.startswith("BT:1b")]
    assert all(grades[u] is Grade.C for u in i2), "one replicate refused the I.2 window"
    assert set(replicates[2].unresolved.values()) == {"refused:other"}
    assert reasons == {}


def test_offline_replay_from_the_cache(tmp_path: Path) -> None:
    ws = windows()
    online = CachedClient(FakeClient(echo_script), tmp_path)
    first = run_pipeline(online, ws)[2]

    def unreachable(request: LLMRequest):
        raise AssertionError("offline replay must not call the model")

    offline = CachedClient(FakeClient(unreachable), tmp_path, offline=True)
    assert run_pipeline(offline, ws)[2] == first
    with pytest.raises(CacheMiss):
        run_pipeline(offline, ws, k=4)


def test_wrong_window_perturbation_with_a_faithful_reader() -> None:
    w = next(w for w in windows() if w.chapter == "I.1")
    perturbed, truth = wrong_window(w, ["pin3"])
    replicates, _, (alignment, _, _) = run_pipeline(FakeClient(echo_script), [perturbed], k=1)
    assert false_link_rate(truth, alignment).value == 0.0
    diags = [d for d in replicates[0].diagnostics if d.kind == "relocation"]
    assert len(diags) == len(w.units), "every true link now lies outside the (wrong) core window"


def test_deletion_perturbation_with_a_faithful_reader() -> None:
    w = next(w for w in windows() if w.chapter == "I.1")
    _, _, (expected, _, _) = run_pipeline(FakeClient(echo_script), [w], k=1)
    perturbed, truth = delete_segments(w, 0.3, seed=1)
    _, _, (result, _, _) = run_pipeline(FakeClient(echo_script), [perturbed], k=1)
    recall = deletion_recall(truth, expected, result)
    assert recall.n >= 1 and recall.value == 1.0
