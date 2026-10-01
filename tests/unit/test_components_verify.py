"""collate.components.verify: the code checks C1-C5 on T2 answers, one test per rule.

Answers are built from the correct answers in data/fixtures/components_texts.yaml and
then broken one rule at a time; every response goes through ``llm.fake``.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate import components as C
from hevajra_matrix.collate.verify import (
    FLAG_CORROBORATED,
    FLAG_POLARITY_UNCORROBORATED,
    FLAG_QUOTE_VARIANT,
    CheckLexicon,
)
from hevajra_matrix.core.types import Segment
from hevajra_matrix.llm.fake import (
    FakeClient,
    invalid_response,
    ok_response,
    refusal_response,
    substituted_response,
    truncated_response,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURE = yaml.safe_load((DATA / "fixtures" / "components_texts.yaml").read_text(encoding="utf-8"))
Q = FIXTURE["quotes"]


def _segments(records, witness: str, lang: str) -> list[Segment]:
    return [Segment(r["id"], witness, lang, r["text"], r["id"], r["id"], r["kind"]) for r in records]


REF = _segments(FIXTURE["reference_units"], FIXTURE["reference_witness"], "bo")
WIT = _segments(FIXTURE["witness_units"], FIXTURE["witness"], "zh")
STORE = {s.id: s for s in REF + WIT}
R_LAMP, R_YOGIN, R_SKULL, R_HERUKA, R_LONG = (s.id for s in REF[:5])
W_LAMP, W_YOGIN, W_SKULL, W_SALUTE, W_NAME, W_LONG = (s.id for s in WIT[:6])

PAIRS = [
    C.Pair(R_LAMP, (W_LAMP,), "reversal", True, frozenset({C.SELECT_DEVIATION})),
    C.Pair(R_YOGIN, (W_YOGIN,), "abridged", False, frozenset({C.SELECT_DEVIATION})),
    C.Pair(R_SKULL, (W_SKULL,), "equivalent", False, frozenset({C.SELECT_SENSITIVE})),
    C.Pair(R_HERUKA, (W_SALUTE, W_NAME), "transliterated", False, frozenset({C.SELECT_DEVIATION})),
]
BATCH = C.plan_batches(PAIRS, STORE, 12)[0]
REQUEST = C.build_request(BATCH, C.ComponentTaskSettings(), "system")
H_LAMP, H_YOGIN, H_SKULL, H_HERUKA = BATCH.handles()


@pytest.fixture(scope="module")
def lexicon() -> CheckLexicon:
    return CheckLexicon.load(DATA, WIT)


def _answer() -> dict:
    return {"pairs": [{"pair": h, **copy.deepcopy(FIXTURE["answers"][i.pair.ref_id])}
                      for h, i in BATCH.handles().items()]}


def _entry(answer: dict, handle: str) -> dict:
    return next(e for e in answer["pairs"] if e["pair"] == handle)


def _slot(answer: dict, handle: str, slot: str) -> dict:
    return next(s for s in _entry(answer, handle)["slots"] if s["slot"] == slot)


def _run(answer: dict, lexicon: CheckLexicon, response=ok_response):
    return C.verify(response(REQUEST, answer), BATCH, lexicon)


def _by_ref(codes) -> dict[str, list[C.SlotCode]]:
    out: dict[str, list[C.SlotCode]] = {}
    for c in codes:
        out.setdefault(c.ref_id, []).append(c)
    return out


def _failed(diagnostics, ref_id: str) -> str:
    (d,) = [d for d in diagnostics if d.kind == "verification_failed" and d.ref_ids == (ref_id,)]
    return d.detail


def test_a_correct_answer_is_kept_in_full(lexicon):
    codes, diagnostics = _run(_answer(), lexicon)
    assert diagnostics == []
    by_ref = _by_ref(codes)
    assert list(by_ref) == [R_LAMP, R_YOGIN, R_SKULL, R_HERUKA]
    assert [(c.slot, c.code) for c in by_ref[R_YOGIN]] == [
        ("agent", "gen"), ("place", "ret"), ("quantity", "om"), ("action", "ret")]
    assert all(c.usable and c.wit_ids == (W_SALUTE, W_NAME) for c in by_ref[R_HERUKA])
    assert all(c.polarity_flip for c in by_ref[R_LAMP])


# --------------------------------------------------------------------------- C1 pair set
def test_c1_unknown_handle_is_ignored(lexicon):
    answer = _answer()
    answer["pairs"].append({**copy.deepcopy(_entry(answer, H_YOGIN)), "pair": "p99"})
    codes, diagnostics = _run(answer, lexicon)
    assert [d.kind for d in diagnostics] == ["unknown_handle"]
    assert len(_by_ref(codes)) == 4


def test_c1_repeated_handle_keeps_the_first_answer(lexicon):
    answer = _answer()
    second = copy.deepcopy(_entry(answer, H_SKULL))
    second["slots"] = [{"slot": "agent", "code": "add", "ref_quote": "", "wit_quote": "x"}]
    answer["pairs"].append(second)
    codes, diagnostics = _run(answer, lexicon)
    assert [(d.kind, d.ref_ids) for d in diagnostics] == [("duplicate_pair", (R_SKULL,))]
    assert [c.slot for c in _by_ref(codes)[R_SKULL]] == ["place", "patient", "action"]


def test_c1_missing_pair_is_unassessed(lexicon):
    answer = _answer()
    answer["pairs"] = [e for e in answer["pairs"] if e["pair"] != H_SKULL]
    codes, diagnostics = _run(answer, lexicon)
    assert [(d.kind, d.ref_ids, d.wit_ids) for d in diagnostics] == [("unassessed", (R_SKULL,), (W_SKULL,))]
    assert R_SKULL not in _by_ref(codes)


# --------------------------------------------------------------------------- C2 quote shape
@pytest.mark.parametrize("slot, code, ref_empty, wit_empty", [
    ("quantity", "om", False, False),     # om with a witness quote
    ("quantity", "om", True, True),       # om without a reference quote
    ("place", "add", False, False),       # add with a reference quote
    ("place", "ret", False, True),        # ret without a witness quote
    ("place", "gen", True, False),        # gen without a reference quote
])
def test_c2_quote_shape_is_enforced(lexicon, slot, code, ref_empty, wit_empty):
    answer = _answer()
    s = _slot(answer, H_YOGIN, slot)
    s["code"] = code
    if code == "om" and not wit_empty:
        s["wit_quote"] = FIXTURE["answers"][R_YOGIN]["slots"][0]["wit_quote"]
    if ref_empty:
        s["ref_quote"] = ""
    if wit_empty:
        s["wit_quote"] = ""
    codes, diagnostics = _run(answer, lexicon)
    assert f"C2 {slot}/{code}" in _failed(diagnostics, R_YOGIN)
    assert R_YOGIN not in _by_ref(codes) and len(_by_ref(codes)) == 3   # the other pairs stand


def test_c2_add_with_only_a_witness_quote_is_valid(lexicon):
    answer = _answer()
    _entry(answer, H_SKULL)["slots"].append({"slot": "instrument", "code": "add", "ref_quote": "",
                                             "wit_quote": _slot(answer, H_SKULL, "patient")["wit_quote"]})
    codes, diagnostics = _run(answer, lexicon)
    assert diagnostics == [] and ("instrument", "add") in [(c.slot, c.code) for c in _by_ref(codes)[R_SKULL]]


# --------------------------------------------------------------------------- C3 verbatim
def test_c3_reference_quote_must_cover_whole_syllables(lexicon):
    answer = _answer()
    _slot(answer, H_SKULL, "patient")["ref_quote"] = Q["bo_inside_syllable"]
    codes, diagnostics = _run(answer, lexicon)
    assert "C3 patient/ret ref_quote" in _failed(diagnostics, R_SKULL)
    assert R_SKULL not in _by_ref(codes)


def test_c3_witness_quote_must_come_from_the_pairs_own_segments(lexicon):
    assert Q["zh_elsewhere"] in STORE[W_LONG].text
    answer = _answer()
    _slot(answer, H_SKULL, "action")["wit_quote"] = Q["zh_elsewhere"]
    _, diagnostics = _run(answer, lexicon)
    assert "C3 action/ret wit_quote" in _failed(diagnostics, R_SKULL)


def test_c3_witness_quote_may_span_the_linked_segments(lexicon):
    answer = _answer()
    salute, name = STORE[W_SALUTE].text, STORE[W_NAME].text
    _slot(answer, H_HERUKA, "action")["wit_quote"] = salute[-1] + name[0]
    codes, diagnostics = _run(answer, lexicon)
    assert diagnostics == [] and len(_by_ref(codes)[R_HERUKA]) == 2


def test_c3_variant_form_is_kept_with_a_flag(lexicon):
    answer = _answer()
    _slot(answer, H_SKULL, "patient")["wit_quote"] = Q["zh_skull_simplified"]
    codes, diagnostics = _run(answer, lexicon)
    assert diagnostics == []
    assert all(FLAG_QUOTE_VARIANT in c.flags for c in _by_ref(codes)[R_SKULL])
    assert not any(FLAG_QUOTE_VARIANT in c.flags for c in _by_ref(codes)[R_YOGIN])


def test_all_broken_rules_are_named(lexicon):
    answer = _answer()
    _slot(answer, H_SKULL, "patient")["ref_quote"] = Q["bo_inside_syllable"]
    _slot(answer, H_SKULL, "action")["wit_quote"] = Q["zh_elsewhere"]
    detail = _failed(_run(answer, lexicon)[1], R_SKULL)
    assert "patient/ret ref_quote" in detail and "action/ret wit_quote" in detail


# --------------------------------------------------------------------------- C4 lit
def test_c4_lit_needs_a_transcription(lexicon):
    answer = _answer()
    s = _slot(answer, H_HERUKA, "action")
    s["code"] = "lit"
    assert s["wit_quote"] == Q["zh_not_transcription"]
    codes, diagnostics = _run(answer, lexicon)
    assert "C4 action/lit" in _failed(diagnostics, R_HERUKA)
    assert R_HERUKA not in _by_ref(codes)


def test_c4_uses_the_witness_mantra_charset(lexicon):
    no_mantras = dataclasses.replace(lexicon, translit_charset=frozenset())
    _, diagnostics = _run(_answer(), no_mantras)
    assert "C4 patient/lit" in _failed(diagnostics, R_HERUKA)
    assert _run(_answer(), lexicon)[1] == []


# --------------------------------------------------------------------------- C5 polarity
def test_c5_flip_without_a_negation_slot_fails(lexicon):
    answer = _answer()
    entry = _entry(answer, H_LAMP)
    entry["slots"] = [s for s in entry["slots"] if s["slot"] != "negation_modality"]
    _, diagnostics = _run(answer, lexicon)
    assert "C5" in _failed(diagnostics, R_LAMP)


def test_c5_flip_with_a_retained_negation_slot_fails(lexicon):
    answer = _answer()
    _slot(answer, H_LAMP, "negation_modality")["code"] = "ret"
    _, diagnostics = _run(answer, lexicon)
    assert "C5" in _failed(diagnostics, R_LAMP)


@pytest.mark.parametrize("code", ["om", "add"])
def test_c5_flip_accepts_an_omitted_or_added_negation(lexicon, code):
    answer = _answer()
    s = _slot(answer, H_LAMP, "negation_modality")
    s["code"] = code
    s["ref_quote" if code == "add" else "wit_quote"] = ""
    codes, diagnostics = _run(answer, lexicon)
    assert diagnostics == []
    flags = {f for c in _by_ref(codes)[R_LAMP] for f in c.flags}
    # add keeps the witness negator; om keeps only the reference obligative (no negator)
    assert flags == ({FLAG_CORROBORATED} if code == "add" else {FLAG_POLARITY_UNCORROBORATED})


def test_negation_slot_without_flip_is_allowed_and_disagreement_is_flagged(lexicon):
    answer = _answer()
    _entry(answer, H_LAMP)["polarity_flip"] = False
    codes, diagnostics = _run(answer, lexicon)
    assert diagnostics == []
    assert all(c.flags == {C.FLAG_POLARITY_DISAGREES} and not c.polarity_flip for c in _by_ref(codes)[R_LAMP])


def test_flip_is_corroborated_only_by_a_negator_on_one_side(lexicon):
    codes, _ = _run(_answer(), lexicon)
    assert all(c.flags == {FLAG_CORROBORATED} for c in _by_ref(codes)[R_LAMP])
    answer = _answer()
    _slot(answer, H_LAMP, "negation_modality")["wit_quote"] = Q["zh_affirmative"]
    codes, diagnostics = _run(answer, lexicon)
    assert diagnostics == []
    assert all(c.flags == {FLAG_POLARITY_UNCORROBORATED} for c in _by_ref(codes)[R_LAMP])


def test_identical_slot_entries_are_merged_with_a_flag(lexicon):
    answer = _answer()
    entry = _entry(answer, H_SKULL)
    entry["slots"].append(copy.deepcopy(entry["slots"][0]))
    codes, diagnostics = _run(answer, lexicon)
    assert diagnostics == []
    assert len(_by_ref(codes)[R_SKULL]) == 3
    assert all(C.FLAG_DUPLICATE_SLOT in c.flags for c in _by_ref(codes)[R_SKULL])


# --------------------------------------------------------------------------- response status
@pytest.mark.parametrize("response, reason", [
    (lambda r, a: refusal_response(r, category="bio"), "refused:bio"),
    (lambda r, a: refusal_response(r), "refused:unspecified"),     # as collate (T1) and topics (T3)
    (lambda r, a: truncated_response(r), "truncated"),
    (lambda r, a: invalid_response(r), "invalid"),
    (lambda r, a: dataclasses.replace(ok_response(r, a), data={"pairs": [{"pair": "p01"}]}), "invalid"),
])
def test_unusable_responses_code_nothing_and_name_the_reason(lexicon, response, reason):
    codes, diagnostics = _run(_answer(), lexicon, response)
    assert codes == []
    assert [(d.kind, d.ref_ids) for d in diagnostics] == [(reason, (p.ref_id,)) for p in PAIRS]


def test_substituted_model_answer_is_a_hint_only(lexicon):
    codes, diagnostics = _run(_answer(), lexicon, substituted_response)
    assert len(codes) == sum(len(FIXTURE["answers"][p.ref_id]["slots"]) for p in PAIRS)
    assert all(c.reason == "substituted_model" and not c.usable for c in codes)
    assert [d.kind for d in diagnostics] == ["substituted_model"]
    assert diagnostics[0].ref_ids == tuple(p.ref_id for p in PAIRS)


def test_substituted_model_answer_is_still_verified(lexicon):
    answer = _answer()
    _slot(answer, H_SKULL, "patient")["ref_quote"] = Q["bo_inside_syllable"]
    codes, diagnostics = _run(answer, lexicon, substituted_response)
    assert R_SKULL not in _by_ref(codes)
    assert {d.kind for d in diagnostics} == {"verification_failed", "substituted_model"}


def test_fake_client_round_trip(lexicon):
    client = FakeClient(lambda request: _answer())
    codes, diagnostics = C.verify(client.complete(REQUEST), BATCH, lexicon)
    assert diagnostics == [] and len(_by_ref(codes)) == 4
    assert STORE[W_NAME].text in client.requests[0].body
