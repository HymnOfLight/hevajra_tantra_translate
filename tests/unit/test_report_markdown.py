"""``report.markdown.render``: what each report level may print (synthesis 5.5, 6.1; A7)."""

from __future__ import annotations

import re
from types import MappingProxyType

import pytest

from hevajra_matrix.core.types import Alignment, Cell, Estimate, Grade, Link, Relation, Status, WitnessOnlyKind
from hevajra_matrix.evaluation.gate import GateReport
from hevajra_matrix.evaluation.sentinels import SentinelResult
from hevajra_matrix.report.markdown import (
    ReportInputs,
    WitnessOnlyRow,
    alignment_counts,
    blocking_reason,
    human_verified_counts,
    redact,
    render,
)

DECIMAL = re.compile(r"(?<![\w.:])-?\d+\.\d+(?![\w.])")
SCOPE = ("relative to Derge D417-418 (revised by gZhon nu dpal), not to the Sanskrit",
         "Chinese = Ming-based Taisho T0892; transmission changes are included",
         "units = provisional Derge pada/clause segments")
E3_REASON = "G4: no manuscript column; reference is the co-witness"
COUNTS = {"nondev": 90, "dev_present": 4, "partial": 3, "absent": 2, "unaligned": 1, "witness_only": 5}


def gate(level: int, **kw) -> GateReport:
    passed = {"G0": True, "G1": level >= 1, "G2": level >= 1, "G3": level >= 2}
    reasons = kw.pop("reasons", None)
    if reasons is None:
        reasons = (() if level >= 1 else ("G1: link F1 0.612 < 0.800",
                                          "G1: link_f1 does not beat dp:anchor: paired difference 0.010 "
                                          "[-0.020, 0.040]")) + \
            (() if level >= 2 else ("G3: stratum pos:absent:other verified 3 of 9",))
    return GateReport(level=level, confirmatory=kw.pop("confirmatory", False), passed=MappingProxyType(passed),
                      reasons=tuple(reasons), not_estimable=MappingProxyType(kw.pop("not_estimable", {"E3": E3_REASON})),
                      scope_note=kw.pop("scope_note", "exploratory; test set scored 2 times by 1 instrument versions"))


def inputs(**kw) -> ReportInputs:
    est = Estimate("E1_any", 0.123, 0.101, 0.150, 100, "scope", ("machine status error",))
    base = dict(
        run_id="20260930T000000Z-abc1234", reference="bo_derge_D417_418", witness="zh_T0892_song", scope=SCOPE,
        instrument_counts=COUNTS, control_counts={"dp:zero": {"nondev": 99, "absent": 1}, "dp:anchor": COUNTS},
        human_verified={"absent": 7, "witness_only": 2},
        sentinels=(SentinelResult("S2a", "relation_in", "proposal", "proposal", "verified", True, False,
                                  "D418:17b.6.2 is equivalent (expected reversal); score 0.5"),),
        gold_set="test",
        scores={"claude:consensus": {"link_f1": Estimate("link_f1", 0.85, 0.8, 0.9, 300, "gold:test")},
                "placebo:shuffled": {"link_f1": Estimate("link_f1", 0.1, 0.05, 0.15, 300, "gold:test")}},
        witness_only=(WitnessOnlyRow("+T0892:0601c01.1", "II.12", "addition", "A"),),
        estimates={"E1_any": est, "E1_any_blind": Estimate("E1_any_blind", 0.13, 0.11, 0.16, 100, "scope"),
                   "E3": Estimate.missing("E3", E3_REASON)},
        manski={"any": (0.12, 0.2)},
    )
    base.update(kw)
    return ReportInputs(**base)


def test_level_0_prints_only_counts_controls_sentinels_and_scope() -> None:
    text = render(inputs(), gate(0))
    assert "**Report level 0 (DESCRIPTIVE)**" in text
    assert "## Counts (unvalidated instrument output)" in text
    assert "| outcome | Claude consensus | P1 length-only DP | B0 anchor DP |" in text
    assert "| absent (no counterpart) | 2 | 1 | 2 |" in text
    assert all(f"- {s}" in text for s in SCOPE)
    assert "| S2a | proposal | verified (blocking) | FAIL:" in text
    assert "## Alignment scores" not in text and "## Witness-only material" not in text
    assert not DECIMAL.search(text), DECIMAL.search(text)
    assert not re.search(r"\[\s*-?\d", text) and "%" not in text and "Delta" not in text
    assert "0.612" not in text and "G1: link F1 <withheld> < <withheld>" in text


def test_every_estimand_is_printed_or_not_estimable_with_a_gate_reason() -> None:
    for level in (0, 1, 2):
        text = render(inputs(), gate(level))
        lines = [line for line in text.splitlines() if line.startswith("- E")]
        assert len(lines) == 8
        for line in lines:
            value = line.split(": ", 1)[1]
            assert re.match(r"NOT_ESTIMABLE: (G\d|stats): \S", value) or re.match(r"\d\.\d{3} \[", value), line
        assert f"- E3 decomposition of deviations (Vorlage, shared, residual): NOT_ESTIMABLE: {E3_REASON}" in text


def test_the_blocking_gate_depends_on_the_level() -> None:
    assert render(inputs(), gate(0)).count("E1 share of units deviating (D_any): NOT_ESTIMABLE: G1: link F1") == 1
    level1 = render(inputs(), gate(1))
    assert "E1 share of units deviating (D_any): NOT_ESTIMABLE: G3: stratum pos:absent:other verified 3 of 9" \
        in level1
    assert blocking_reason(gate(2), 2) is None
    assert blocking_reason(gate(1, reasons=()), 2) == "G3: not passed"


def test_level_1_adds_scores_with_intervals_and_the_witness_only_list() -> None:
    text = render(inputs(), gate(1))
    assert "## Counts (instrument output)" in text
    assert "| Claude consensus | 0.850 [0.800, 0.900] |" in text
    assert "| P2 shuffled Claude | 0.100 [0.050, 0.150] |" in text, "P2 is a control in score tables"
    assert "| +T0892:0601c01.1 | II.12 | addition | A |" in text
    assert "0.123" not in text, "estimates need level 2"


def test_level_1_prints_per_relation_recall_and_per_class_agreement_with_intervals() -> None:
    scores = {"claude:consensus": {"link_f1": Estimate("link_f1", 0.85, 0.8, 0.9, 300, "gold:test"),
                                   "relation_recall:abridged": Estimate("relation_recall:abridged", 0.4, 0.2, 0.6, 300,
                                                                        "gold:test"),
                                   "status_agreement:ABSENT": Estimate("status_agreement:ABSENT", 0.7, 0.6, 0.8, 300,
                                                                       "gold:test")},
              "dp:zero": {"link_f1": Estimate("link_f1", 0.5, 0.4, 0.6, 300, "gold:test")}}
    text = render(inputs(scores=scores), gate(1))
    assert "| metric | Claude consensus | P1 length-only DP |" in text
    assert "| recall of relation abridged | 0.400 [0.200, 0.600] | n/a |" in text
    assert "| status agreement ABSENT | 0.700 [0.600, 0.800] | n/a |" in text
    assert "recall of relation" not in render(inputs(scores=scores), gate(0))


def test_overall_refusal_rate_is_not_labelled_as_a_topic_group() -> None:
    # "refusal_rate" is both a metric and the prefix of its per-topic breakdown; the overall
    # value used to be printed as "refusal rate, topic " with an empty group.
    scores = {"claude:consensus": {"refusal_rate": Estimate("refusal_rate", 0.01, 0.0, 0.02, 300, "gold:test"),
                                   "refusal_rate:sensitive": Estimate("refusal_rate:sensitive", 0.02, 0.0, 0.05, 90,
                                                                      "gold:test")}}
    text = render(inputs(scores=scores), gate(1))
    assert "| refusal rate, all units | 0.010 [0.000, 0.020] |" in text
    assert "| refusal rate, topic sensitive | 0.020 [0.000, 0.050] |" in text
    assert "refusal rate, topic  |" not in text


def test_level_2_prints_estimates_blind_sensitivity_and_manski() -> None:
    text = render(inputs(), gate(2, confirmatory=True))
    assert "- E1 share of units deviating (D_any): 0.123 [0.101, 0.150] (n = 100; covers: machine status error)" in text
    assert "automation-bias sensitivity (blind verdicts): 0.130 [0.110, 0.160]" in text
    assert "prior sensitivity" not in text
    prior = Estimate("E1_any_prior", 0.11, 0.09, 0.14, 100, "scope")
    with_prior = render(inputs(estimates={**inputs().estimates, "E1_any_prior": prior}), gate(2))
    assert "  - prior sensitivity (Dirichlet symmetric in D, not Jeffreys): 0.110 [0.090, 0.140]" in with_prior
    assert "- any: [0.120, 0.200] (wider than the gate allows)" in text
    assert "**Confirmatory**" in text
    assert "E1 share of units absent: NOT_ESTIMABLE: stats: not computed in this run" in text


def test_g4_blocks_e4_even_at_level_2() -> None:
    e4 = Estimate("E4", 0.05, -0.01, 0.1, 80, "scope", ("chapter sampling",))
    reason = "G4: topic labels incomplete"
    text = render(inputs(estimates={"E4": e4}), gate(2, not_estimable={"E3": E3_REASON, "E4": reason}))
    assert f"E4 sensitive-vs-neutral contrast: NOT_ESTIMABLE: {reason}" in text
    text = render(inputs(estimates={"E4": e4}), gate(2, not_estimable={"E3": E3_REASON}))
    assert "E4 sensitive-vs-neutral contrast: 0.050 [-0.010, 0.100]" in text


def test_human_verified_counts_are_printed_at_every_level() -> None:
    for level in (0, 1, 2):
        text = render(inputs(), gate(level))
        section = text.split("## Human-verified counts (grade A)")[1].split("##")[0]
        assert "instrument-independent lower bounds" in section.lower()
        assert "| absent (no counterpart) | 7 |" in section and "| witness-only segments | 2 |" in section


def test_p2_counts_are_refused() -> None:
    with pytest.raises(ValueError, match="P2"):
        render(inputs(control_counts={"placebo:shuffled": COUNTS}), gate(0))


def test_no_instrument_output_and_no_gate() -> None:
    empty = GateReport(level=0, confirmatory=False, passed={}, reasons=("G0: evaluate has not run",))
    text = render(inputs(instrument_counts=None, control_counts={}, sentinels=()), empty)
    assert "No instrument output in this run" in text and "No alignment in this run." in text
    assert "the gates have not been evaluated" in text and "No sentinel was checked" in text


def test_redact_keeps_ids_and_withholds_values() -> None:
    reason = "G2: quote failure rate 0.031 in D418:17b.6.2 [0.01, 0.05]"
    assert redact(reason, 0) == "G2: quote failure rate <withheld> in D418:17b.6.2 <withheld>"
    assert redact(reason, 1) == reason


def test_counting_helpers() -> None:
    a = Alignment("x", "r", "w", (
        Link("u1", ("z1",), Relation.EQUIVALENT), Link("u2", (), Relation.NO_COUNTERPART),
        Link("u3", ("z2",), Relation.TRANSLITERATED), Link("u4", ("z3",), Relation.EQUIVALENT, polarity_flip=True),
        Link(None, ("z4", "z5"), WitnessOnlyKind.ADDITION)))
    counts = alignment_counts(a, {"u1": "prose", "u2": "prose", "u3": "mantra", "u4": "prose", "u5": "prose"})
    assert counts == {"nondev": 2, "dev_present": 1, "partial": 0, "absent": 1, "unaligned": 1, "witness_only": 2}
    cells = [Cell("u1", "w", Status.ABSENT, Grade.A, relation="no_counterpart"),
             Cell("u2", "w", Status.PRESENT, Grade.B, relation="equivalent"),
             Cell("u3", "w", Status.UNALIGNED, Grade.A, reason="unresolved_by_human"),
             Cell("+z9", "w", Status.NA, Grade.A, relation="addition")]
    assert human_verified_counts(cells, {}) == {"nondev": 0, "dev_present": 0, "partial": 0, "absent": 1,
                                                "unaligned": 1, "witness_only": 1}


def test_e5_is_a_census_count_without_interval() -> None:
    e5 = Estimate("E5", 3.0, 3.0, 3.0, 57, "scope", ("human-verified census of witness-only claims",))
    text = render(inputs(estimates={"E5": e5}), gate(2))
    assert "- E5 witness-only material (human-verified segments): 3 segments, 57 characters" in text


RELIABILITY = {"outcome": "status_correct", "n": 40, "accuracy": 0.8, "brier": 0.12, "brier_constant": 0.16,
               "beats_constant": True,
               "table": {"high": {"n": 30, "correct": 27, "accuracy": 0.9, "nominal": 0.9},
                         "medium": {"n": 8, "correct": 5, "accuracy": 0.625, "nominal": 0.7},
                         "low": {"n": 2, "correct": 0, "accuracy": 0.0, "nominal": 0.5}}}


def test_confidence_reliability_and_polarity_recall_from_level_1() -> None:
    kw = dict(reliability={"claude:consensus": RELIABILITY}, negation={"hits": 3, "n": 4})
    text = render(inputs(**kw), gate(1))
    assert "### Confidence reliability, Claude consensus" in text
    assert "Brier score 0.120 (nominal high/medium/low probabilities) vs 0.160 for a constant predictor: " \
           "confidence beats the constant, so it is informative." in text
    assert "| medium | 0.700 | 8 | 5 | 0.625 |" in text
    assert "P-negation (reported, never gated): polarity recall 3/4 perturbed units." in text
    worse = {**RELIABILITY, "brier": 0.2, "beats_constant": False}
    assert "so it is used only as a stratum" in render(inputs(reliability={"claude:consensus": worse}), gate(1))
    assert "Confidence reliability" not in render(inputs(**kw), gate(0))


E3_DETAILS = {"n_deviating": 5, "units": 10, "manuscripts": ["C", "K"], "cowitness": "bo_derge_D417_418", "m_min": 2,
              "counts": {"attested_vorlage": 0, "vorlage_ms": 1, "shared_revised": 1, "residual": 2,
                         "insufficient": 1},
              "verified_deviating": 0,
              "bound": {"unverified_shared": 1, "counts": {"attested_vorlage": 0, "vorlage_ms": 1, "shared_revised": 0,
                                                           "residual": 3, "insufficient": 1},
                        "shared": {"observed": 0, "expected": 0.0, "excess": 0.0, "phi": 0.0, "n": 5}},
              "vorlage": {"observed": 1, "expected": 1.25, "excess": -0.25, "phi": -0.125, "n": 7},
              "shared": {"observed": 1, "expected": 0.333, "excess": 0.667, "phi": 0.333, "n": 5}}


def test_e3_prints_counts_bound_and_excess_at_level_2_when_g4_passes() -> None:
    e3 = Estimate("E3", 5.0, None, None, 10, "scope", ("decomposition",))
    kw = dict(estimates={"E3": e3}, e3_details=E3_DETAILS)
    text = render(inputs(**kw), gate(2, not_estimable={}))
    assert "- E3 decomposition of deviations (Vorlage, shared, residual): 5 deviating units of 10; manuscripts C, K; " \
           "co-witness bo_derge_D417_418; m_min 2" in text
    assert "  - as measured: attested_vorlage 0, vorlage_ms 1, shared_revised 1, residual 2, insufficient 1" in text
    assert "  - A2 bound (0 deviating units human-verified; 1 unverified shared counted as not shared): " \
           "attested_vorlage 0, vorlage_ms 1, shared_revised 0, residual 3, insufficient 1" in text
    assert "  - shared (the co-witness deviates too): O 1, E 0.333, O - E 0.667, phi 0.333 (n = 5)" in text
    blocked = render(inputs(**kw), gate(1, not_estimable={}))
    assert "E3 decomposition of deviations (Vorlage, shared, residual): NOT_ESTIMABLE: G3:" in blocked
