"""collate.verify: one or more tests per code check V1-V8 and V11 on fake payloads
(V9 and V10 need several chunks and are tested in test_collate_merge.py)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hevajra_matrix.collate.collator import Unresolved
from hevajra_matrix.collate.verify import (
    FLAG_CORROBORATED,
    FLAG_CROSSING,
    FLAG_DUPLICATE,
    FLAG_HINT,
    FLAG_INSTRUCTION_AS_MANTRA,
    FLAG_POLARITY_MISMATCH,
    FLAG_POLARITY_UNCORROBORATED,
    FLAG_QUOTE_VARIANT,
    FLAG_RELOCATION,
    FLAG_TRANSLITERATION_UNSUPPORTED,
    FLAG_UNKNOWN_HANDLE,
    FLAG_WITNESS_ONLY_TRIMMED,
    Collation,
    quote_match,
    reconcile_witness_only,
    verify,
)
from hevajra_matrix.core.types import Link, Relation, WitnessOnlyKind
from hevajra_matrix.registry import load_concordance

from test_collate_support import (
    DATA,
    SOURCE,
    concordance,
    gold_answer,
    lexicon,
    quote,
    raw,
    reference,
    unit,
    window,
    witness,
    witness_only,
)

# Fixture ids (see data/fixtures/collate_texts.yaml)
U_SAID, U_TREES, U_FISH, U_MANTRA, U_FLOWERS, U_HORSES = (
    "BT:1a.1.1", "BT:1a.1.2", "BT:1a.2.1", "BT:1a.2.2", "BT:1a.3.1", "BT:1a.3.2")
Z_SAID, Z_TREES, Z_FISH, Z_MANTRA, Z_NOTE, Z_FLOWERS, Z_HORSES, Z_ADDITION = (
    "ZT:0001a03.1", "ZT:0001a03.2", "ZT:0001a04.1", "ZT:0001a05.1", "ZT:0001a05.n1", "ZT:0001a06.1",
    "ZT:0001a07.1", "ZT:0001a08.1")
Z_PIN2 = "ZT:0001b02.1"
Z_PIN3 = "ZT:0001c02.1"


def run(answer, w=None) -> Collation:
    w = w or window()
    return verify(raw(w, answer), w, lexicon(), SOURCE)


def with_unit(record: dict, w=None) -> dict:
    """The gold answer of the I.1 window with one unit record replaced."""
    w = w or window()
    answer = gold_answer(w)
    answer["units"] = [record if u["ref"] == record["ref"] else u for u in answer["units"]]
    return answer


def link(result: Collation, uid: str) -> Link:
    return result.alignment.by_ref()[uid]


# --------------------------------------------------------------------------- the gold passes
def test_gold_answer_passes_with_only_benign_flags() -> None:
    w = window()
    result = run(gold_answer(w), w)
    assert dict(result.unresolved) == {}
    assert len(result.alignment.by_ref()) == len(w.units)
    assert link(result, U_HORSES).flags == frozenset({FLAG_CORROBORATED})
    assert all(link(result, u.id).flags == frozenset() for u in w.units if u.id != U_HORSES)
    assert result.diagnostics == ()
    assert {x.wit_ids[0]: x.relation for x in result.alignment.witness_only()} == {
        "ZT:0001a02.1": WitnessOnlyKind.PARATEXT, Z_NOTE: WitnessOnlyKind.TRANSLATOR_NOTE,
        Z_ADDITION: WitnessOnlyKind.ADDITION}
    first = link(result, U_SAID)
    assert first.source == SOURCE and first.wit_ids == (Z_SAID,) and first.confidence == "high"
    assert {q.side for q in first.quotes} == {"ref", "wit"}


# --------------------------------------------------------------------------- V1 coverage
def test_v1_duplicate_record_keeps_the_first_and_flags() -> None:
    w = window()
    answer = gold_answer(w)
    dup = unit(w, U_SAID, [Z_TREES], Relation.SUBSTITUTION, ref_quote=w.segment[U_SAID].text)
    answer["units"].append(dup)
    result = run(answer, w)
    assert link(result, U_SAID).wit_ids == (Z_SAID,)
    assert FLAG_DUPLICATE in link(result, U_SAID).flags


def test_v1_missing_unit_in_a_large_window_is_unassessed() -> None:
    # 2% of 60 units may be missing before the window fails: build a 60-unit window
    w = window()
    many = tuple(replace(w.units[i % 6], id=f"BT:9a.{i}.1", text=f"{w.units[i % 6].text} {i}")
                 for i in range(60))
    big = replace(w, units=many)
    answer = {"units": [unit(big, u.id, [Z_SAID], ref_quote="", wit_quote=w.segment[Z_SAID].text)
                        for u in many[1:]], "witness_only": []}
    result = run(answer, big)
    assert result.unresolved == {many[0].id: "unassessed"}
    assert any(d.kind == "unassessed" for d in result.diagnostics)


def test_v1_more_than_two_percent_missing_invalidates_the_window() -> None:
    w = window()
    answer = gold_answer(w)
    answer["units"] = answer["units"][1:]
    result = run(answer, w)
    assert result.alignment.links == ()
    assert set(result.unresolved.values()) == {"invalid"} and len(result.unresolved) == len(w.units)
    assert result.diagnostics[-1].kind == "window_invalid"


def test_v1_unknown_reference_handle_is_reported() -> None:
    w = window()
    answer = gold_answer(w)
    answer["units"].append({**answer["units"][0], "ref": "r999"})
    result = run(answer, w)
    assert any(d.kind == "unknown_handle" and "r999" in d.detail for d in result.diagnostics)
    assert dict(result.unresolved) == {}


# --------------------------------------------------------------------------- V2 existence
def test_v2_unknown_witness_handle_is_dropped_and_flagged() -> None:
    w = window()
    record = unit(w, U_SAID, [Z_SAID])
    record["wit"].append("z9999")
    result = run(with_unit(record, w), w)
    assert link(result, U_SAID).wit_ids == (Z_SAID,)
    assert FLAG_UNKNOWN_HANDLE in link(result, U_SAID).flags


def test_v2_nothing_left_is_verification_failed() -> None:
    w = window()
    record = unit(w, U_SAID, [Z_SAID])
    record["wit"] = ["z9999"]
    result = run(with_unit(record, w), w)
    assert result.unresolved == {U_SAID: "verification_failed"}
    assert U_SAID not in result.alignment.by_ref()


def test_v2_repeated_handles_are_collapsed_in_text_order() -> None:
    w = window()
    record = unit(w, U_TREES, [Z_FISH, Z_TREES, Z_FISH], Relation.EXPANDED, wit_quote=w.segment[Z_TREES].text)
    result = run(with_unit(record, w), w)
    assert link(result, U_TREES).wit_ids == (Z_TREES, Z_FISH)


# --------------------------------------------------------------------------- V3 consistency
def test_v3_no_counterpart_with_segments_fails() -> None:
    w = window()
    record = unit(w, U_TREES, [Z_TREES], Relation.NO_COUNTERPART)
    result = run(with_unit(record, w), w)
    assert result.unresolved == {U_TREES: "verification_failed"}
    assert "V3" in next(d.detail for d in result.diagnostics if d.ref_ids == (U_TREES,))


def test_v3_other_relation_without_segments_fails() -> None:
    w = window()
    record = unit(w, U_TREES, [], Relation.EQUIVALENT)
    assert run(with_unit(record, w), w).unresolved == {U_TREES: "verification_failed"}


def test_v3_no_counterpart_with_empty_wit_is_kept() -> None:
    w = window()
    record = unit(w, U_TREES, [], Relation.NO_COUNTERPART)
    result = run(with_unit(record, w), w)
    assert link(result, U_TREES).wit_ids == () and link(result, U_TREES).relation is Relation.NO_COUNTERPART


# --------------------------------------------------------------------------- V4 verbatim quotes
def test_v4_quote_not_in_the_linked_segments_fails() -> None:
    w = window()
    record = unit(w, U_SAID, [Z_SAID], wit_quote=quote("not_verbatim"))
    assert run(with_unit(record, w), w).unresolved == {U_SAID: "verification_failed"}


def test_v4_quote_from_an_unlinked_segment_fails() -> None:
    w = window()
    record = unit(w, U_SAID, [Z_SAID], wit_quote=w.segment[Z_TREES].text)
    assert run(with_unit(record, w), w).unresolved == {U_SAID: "verification_failed"}


def test_v4_reference_quote_must_be_whole_syllables_of_the_unit() -> None:
    w = window()
    bad = unit(w, U_SAID, [Z_SAID], Relation.PARAPHRASE, ref_quote=quote("bo_partial_syllable"))
    assert run(with_unit(bad, w), w).unresolved == {U_SAID: "verification_failed"}
    other_unit = unit(w, U_SAID, [Z_SAID], Relation.PARAPHRASE, ref_quote=w.segment[U_TREES].text)
    assert run(with_unit(other_unit, w), w).unresolved == {U_SAID: "verification_failed"}


def test_v4_quote_spanning_two_linked_segments_passes() -> None:
    w = window()
    both = w.segment[Z_TREES].text + w.segment[Z_FISH].text
    record = unit(w, U_TREES, [Z_TREES, Z_FISH], Relation.EXPANDED, wit_quote=both)
    assert U_TREES in run(with_unit(record, w), w).alignment.by_ref()


def test_v4_variant_form_match_is_kept_with_a_flag() -> None:
    w = window()
    record = unit(w, U_SAID, [Z_SAID], wit_quote=quote("variant_form"))
    result = run(with_unit(record, w), w)
    assert FLAG_QUOTE_VARIANT in link(result, U_SAID).flags
    assert quote_match(quote("variant_form"), w.segment[Z_SAID].text, "zh", {}) is None
    assert quote_match(w.segment[Z_SAID].text, w.segment[Z_SAID].text, "zh", {}) == "exact"


def test_v4_witness_only_quote_is_checked_too() -> None:
    w = window()
    answer = gold_answer(w)
    answer["witness_only"] = [witness_only(w, [Z_ADDITION], "addition", wit_quote=quote("not_verbatim"))]
    result = run(answer, w)
    assert result.alignment.witness_only() == ()
    assert any(d.kind == "verification_failed" and d.wit_ids == (Z_ADDITION,) for d in result.diagnostics)


# --------------------------------------------------------------------------- V5 required quotes
@pytest.mark.parametrize("relation", [r for r in Relation if r not in (
    Relation.EQUIVALENT, Relation.PARAPHRASE, Relation.EXPANDED, Relation.NO_COUNTERPART)])
def test_v5_ref_quote_required(relation: Relation) -> None:
    w = window()
    record = unit(w, U_FLOWERS, [Z_FLOWERS], relation, polarity_flip=relation is Relation.REVERSAL, ref_quote="")
    assert run(with_unit(record, w), w).unresolved == {U_FLOWERS: "verification_failed"}


def test_v5_ref_quote_required_for_no_counterpart() -> None:
    w = window()
    record = unit(w, U_FLOWERS, [], Relation.NO_COUNTERPART, ref_quote="")
    assert run(with_unit(record, w), w).unresolved == {U_FLOWERS: "verification_failed"}


@pytest.mark.parametrize("relation", [Relation.EQUIVALENT, Relation.PARAPHRASE, Relation.EXPANDED])
def test_v5_ref_quote_optional(relation: Relation) -> None:
    w = window()
    record = unit(w, U_SAID, [Z_SAID], relation, ref_quote="")
    result = run(with_unit(record, w), w)
    assert link(result, U_SAID).relation is relation
    assert [q.side for q in link(result, U_SAID).quotes] == ["wit"]


def test_v5_wit_quote_required_when_linked() -> None:
    w = window()
    record = unit(w, U_SAID, [Z_SAID], wit_quote="  ")
    assert run(with_unit(record, w), w).unresolved == {U_SAID: "verification_failed"}


def test_v5_witness_only_needs_a_quote() -> None:
    w = window()
    answer = gold_answer(w)
    answer["witness_only"] = [witness_only(w, [Z_ADDITION], "addition", wit_quote="")]
    assert run(answer, w).alignment.witness_only() == ()


# --------------------------------------------------------------------------- V6 polarity
def test_v6_reversal_with_negator_on_one_side_is_corroborated() -> None:
    result = run(gold_answer(window()))
    assert FLAG_CORROBORATED in link(result, U_HORSES).flags


def test_v6_reversal_with_negators_on_both_sides_is_uncorroborated() -> None:
    w = window()
    record = unit(w, U_FISH, [Z_FISH], Relation.REVERSAL, polarity_flip=True)
    flags = link(run(with_unit(record, w), w), U_FISH).flags
    assert FLAG_POLARITY_UNCORROBORATED in flags and FLAG_CORROBORATED not in flags


def test_v6_reversal_without_the_flag_is_a_mismatch() -> None:
    w = window()
    record = unit(w, U_HORSES, [Z_HORSES], Relation.REVERSAL, polarity_flip=False)
    flags = link(run(with_unit(record, w), w), U_HORSES).flags
    assert FLAG_POLARITY_MISMATCH in flags and FLAG_CORROBORATED in flags


def test_v6_flip_on_another_relation_is_a_mismatch_and_checked_for_negators() -> None:
    w = window()
    record = unit(w, U_TREES, [Z_TREES], Relation.EQUIVALENT, polarity_flip=True)
    flags = link(run(with_unit(record, w), w), U_TREES).flags
    assert {FLAG_POLARITY_MISMATCH, FLAG_POLARITY_UNCORROBORATED} <= flags


def test_v6_negator_in_an_exclusion_context_does_not_count() -> None:
    from hevajra_matrix.collate.verify import _has_negator
    negators = lexicon().negators
    exclusion = negators.exclusions_for("bo")[0]
    assert not _has_negator(exclusion, "bo", negators)
    assert _has_negator(negators.for_lang("bo")[0], "bo", negators)


# --------------------------------------------------------------------------- V7 locality
def test_v7_link_outside_the_core_window_is_a_relocation() -> None:
    w = window()
    record = unit(w, U_TREES, [Z_PIN3], Relation.SUBSTITUTION, ref_quote=w.segment[U_TREES].text)
    result = run(with_unit(record, w), w)
    assert FLAG_RELOCATION in link(result, U_TREES).flags
    diag = next(d for d in result.diagnostics if d.kind == "relocation")
    assert diag.ref_ids == (U_TREES,) and diag.wit_ids == (Z_PIN3,) and "pin3" in diag.detail


def _as_real_witnesses(conc_mapping=None):
    """The fixture renamed onto the real concordance's witnesses and chapter keys:
    reference chapter I.1 -> II.9, witness pin1/pin2/pin3 -> pin18/pin19/pin20."""
    renamed = {"pin1": "pin18", "pin2": "pin19", "pin3": "pin20"}
    ref = [replace(s, witness="bo_derge_D417_418", chapter="II.9") for s in reference() if s.chapter == "I.1"]
    wit = [replace(s, witness="zh_T0892_song", local_chapter=renamed.get(s.local_chapter, s.local_chapter))
           for s in witness()]
    return ref, wit


def test_v7_link_from_ii9_to_chinese_18_outside_the_window_is_a_relocation() -> None:
    """The v0.2 situation: II.9 mapped to Chinese 19 only; a link to 18 must surface."""
    from hevajra_matrix.collate.windows import WindowParams, plan
    ref, wit = _as_real_witnesses()
    locals_ = tuple(f"pin{i}" for i in range(1, 21))
    conc = concordance({"II.9": ["pin19"]}, witness_id="zh_T0892_song", locals_=locals_)
    (w,) = plan(ref, wit, conc, WindowParams(150, 10, 0))
    assert w.core_locals == frozenset({"pin19"})
    record = unit(w, U_TREES, [Z_TREES], Relation.EQUIVALENT)
    answer = {"units": [record if u.id == U_TREES else unit(w, u.id, [Z_PIN2], ref_quote="",
                                                              wit_quote=w.segment[Z_PIN2].text)
                        for u in w.units], "witness_only": []}
    result = run(answer, w)
    diag = next(d for d in result.diagnostics if d.kind == "relocation")
    assert diag.ref_ids == (U_TREES,) and "pin18" in diag.detail
    assert FLAG_RELOCATION in link(result, U_TREES).flags


def test_v7_concordance_v2_puts_chinese_18_inside_the_ii9_window() -> None:
    from hevajra_matrix.collate.windows import WindowParams, plan
    ref, wit = _as_real_witnesses()
    conc = load_concordance(DATA / "registry" / "concordance.yaml")
    (w,) = plan(ref, wit, conc, WindowParams(150, 10, 1))
    assert "pin18" in w.core_locals
    record = unit(w, U_TREES, [Z_TREES], Relation.EQUIVALENT)
    assert FLAG_RELOCATION not in link(run(with_unit(record, w), w), U_TREES).flags


# --------------------------------------------------------------------------- V8 order
def test_v8_crossing_links_are_flagged_never_removed() -> None:
    w = window()
    answer = gold_answer(w)
    answer["units"][0] = unit(w, U_SAID, [Z_TREES], Relation.SUBSTITUTION)
    answer["units"][1] = unit(w, U_TREES, [Z_SAID], Relation.SUBSTITUTION)
    result = run(answer, w)
    assert len(result.alignment.by_ref()) == len(w.units), "crossing links are kept"
    crossing = {uid for uid, lk in result.alignment.by_ref().items() if FLAG_CROSSING in lk.flags}
    assert len(crossing) == 1 and crossing <= {U_SAID, U_TREES}
    diag = next(d for d in result.diagnostics if d.kind == "crossing")
    assert "1 crossing pair" in diag.detail


def test_v8_monotone_links_have_no_crossing() -> None:
    result = run(gold_answer(window()))
    assert not any(FLAG_CROSSING in lk.flags for lk in result.alignment.links)


def test_v8_longest_chain_keeps_the_majority_in_order() -> None:
    from hevajra_matrix.collate.verify import _longest_non_decreasing
    assert _longest_non_decreasing([0, 1, 9, 2, 3, 4]) == [0, 1, 3, 4, 5]
    assert _longest_non_decreasing([]) == []
    assert _longest_non_decreasing([2, 2, 1]) == [0, 1]


# --------------------------------------------------------------------------- V11 transliteration
def test_v11_transliteration_supported_by_the_charset() -> None:
    result = run(gold_answer(window()))
    lk = link(result, U_MANTRA)
    assert lk.relation is Relation.TRANSLITERATED and FLAG_TRANSLITERATION_UNSUPPORTED not in lk.flags


def test_v11_transliteration_of_ordinary_words_is_flagged() -> None:
    w = window()
    record = unit(w, U_TREES, [Z_TREES], Relation.TRANSLITERATED)
    assert FLAG_TRANSLITERATION_UNSUPPORTED in link(run(with_unit(record, w), w), U_TREES).flags


def test_v11_prose_reference_linked_to_a_mantra_is_an_instruction_as_mantra_candidate() -> None:
    w = window()
    record = unit(w, U_FLOWERS, [Z_MANTRA], Relation.TRANSLITERATED)
    flags = link(run(with_unit(record, w), w), U_FLOWERS).flags
    assert FLAG_INSTRUCTION_AS_MANTRA in flags and FLAG_TRANSLITERATION_UNSUPPORTED not in flags
    # a mantra unit linked to a mantra is not a candidate
    assert FLAG_INSTRUCTION_AS_MANTRA not in link(run(gold_answer(w), w), U_MANTRA).flags


# --------------------------------------------------------------------------- witness-only records
def test_segment_linked_by_a_unit_leaves_the_witness_only_record() -> None:
    w = window()
    answer = gold_answer(w)
    answer["witness_only"] = [witness_only(w, [Z_SAID, Z_ADDITION], "addition")]
    result = run(answer, w)
    (record,) = result.alignment.witness_only()
    assert record.wit_ids == (Z_ADDITION,) and FLAG_WITNESS_ONLY_TRIMMED in record.flags
    assert link(result, U_SAID).wit_ids == (Z_SAID,)


def test_reconcile_prefers_positive_kinds_over_belongs_elsewhere() -> None:
    elsewhere = Link(None, ("a", "b"), WitnessOnlyKind.BELONGS_ELSEWHERE)
    addition = Link(None, ("b",), WitnessOnlyKind.ADDITION)
    unit_link = Link("u", ("a",), Relation.EQUIVALENT)
    out = reconcile_witness_only([unit_link], [elsewhere, addition])
    assert out == [addition]


def test_witness_only_with_no_known_handle_is_dropped() -> None:
    w = window()
    answer = gold_answer(w)
    answer["witness_only"] = [{"wit": ["z9999"], "kind": "addition", "wit_quote": "x"}]
    result = run(answer, w)
    assert result.alignment.witness_only() == ()
    assert any(d.kind == "witness_only_dropped" for d in result.diagnostics)


# --------------------------------------------------------------------------- unresolved windows
def test_unresolved_window_makes_every_unit_unaligned() -> None:
    w = window()
    result = verify(Unresolved(w.key, "refused:bio"), w, lexicon(), SOURCE)
    assert result.alignment.links == ()
    assert dict(result.unresolved) == {u.id: "refused:bio" for u in w.units}
    assert result.diagnostics[0].kind == "window_unresolved"


def test_substituted_hint_is_verified_and_marked() -> None:
    w = window()
    result = verify(Unresolved(w.key, "substituted_model", raw(w, gold_answer(w))), w, lexicon(), SOURCE)
    assert result.alignment.links == ()
    assert result.hints and all(FLAG_HINT in h.flags and h.source == f"{SOURCE}:hint" for h in result.hints)


def test_answer_for_another_window_is_rejected() -> None:
    w = window()
    with pytest.raises(ValueError):
        verify(replace(raw(w, gold_answer(w)), window="I.2.w1"), w, lexicon(), SOURCE)


def test_verify_never_turns_a_failure_into_a_status() -> None:
    """Every unit ends up either linked or UNALIGNED with a reason, never silently dropped."""
    w = window()
    answer = gold_answer(w)
    answer["units"][1]["wit_quote"] = quote("not_verbatim")
    answer["units"][2]["wit"] = []
    result = run(answer, w)
    assert set(result.alignment.by_ref()) | set(result.unresolved) == {u.id for u in w.units}
    assert not set(result.alignment.by_ref()) & set(result.unresolved)
