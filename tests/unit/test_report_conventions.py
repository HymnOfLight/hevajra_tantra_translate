"""The report prints E1_any_conv as a sensitivity variant, with the conventions it counted."""

from __future__ import annotations

from hevajra_matrix.core.types import FLAG_CONVENTION_UNSUPPORTED, Cell, Estimate, Grade, Status
from hevajra_matrix.evaluation.gate import GateReport
from hevajra_matrix.report.markdown import (
    ESTIMANDS,
    ReportInputs,
    convention_counts,
    human_verified_counts,
    render,
)

TN, SR = "transcribed_name", "standing_rendering"
LABEL = dict(ESTIMANDS)["E1_any_conv"]


def gate(level: int) -> GateReport:
    passed = {"G0": True, "G1": level >= 1, "G2": level >= 1, "G3": level >= 2}
    reasons = () if level >= 2 else ("G3: stratum neg:B:other verified 3 of 40",)
    return GateReport(level=level, confirmatory=False, passed=passed, reasons=reasons)


def inputs(conventions=None, **est) -> ReportInputs:
    estimates = {"E1_any": Estimate("E1_any", 0.2, 0.1, 0.3, 50, "s", ("x",)),
                 "E1_any_conv": Estimate("E1_any_conv", 0.3, 0.2, 0.4, 50, "s", ("x",)), **est}
    return ReportInputs(run_id="r", reference="bo", witness="zh", scope=("s",), estimates=estimates,
                        manski={"any": (0.2, 0.21), "any_conv": (0.3, 0.31)}, conventions=conventions or {})


def test_e1_any_conv_follows_e1_any_and_is_labelled_a_sensitivity_variant():
    names = [name for name, _ in ESTIMANDS]
    assert names.index("E1_any_conv") == names.index("E1_any") + 1
    assert "sensitivity" in LABEL and "not primary" in LABEL and "conventions counted" in LABEL


def test_level_2_prints_the_caliber_and_the_conventions_counted():
    counts = {SR: 2, TN: 5, "unsupported": 1, "human": 3}
    text = render(inputs(counts), gate(2))
    assert f"- {LABEL}: 0.300 [0.200, 0.400] (n = 50; covers: x)" in text
    assert "  - caliber: a unit also counts when it carries a convention" in text
    assert ("  - conventions counted: 7 units (standing_rendering 2, transcribed_name 5); 1 without code support "
            "(convention_unsupported), 3 human-decided") in text
    assert "- any_conv (conventions counted, sensitivity): [0.300, 0.310]" in text
    assert "- any: [0.200, 0.210]" in text
    assert "E1_any_conv_blind" not in text and text.count("automation-bias") == 0


def test_a_v1_run_says_the_variant_equals_e1_any():
    text = render(inputs({SR: 0, TN: 0, "unsupported": 0, "human": 0}), gate(2))
    assert "no counted unit carries a convention (e.g. a collate.v1 run): equals E1_any" in text
    assert "conventions counted:" not in text.replace("conventions counted, sensitivity", "")


def test_below_level_2_the_variant_is_not_estimable_with_the_gate_reason():
    for level in (0, 1):
        text = render(inputs({TN: 5}), gate(level))
        assert f"- {LABEL}: NOT_ESTIMABLE: G" in text
        assert "caliber" not in text and "transcribed_name 5" not in text
    missing = render(inputs(E1_any_conv=Estimate.missing("E1_any_conv", "G3: no phase-2 verification or audit "
                                                                        "verdict yet")), gate(2))
    assert f"- {LABEL}: NOT_ESTIMABLE: G3: no phase-2 verification or audit verdict yet" in missing


def test_convention_counts():
    cells = [Cell("u1", "w", Status.PRESENT, Grade.B, relation="paraphrase", flags=frozenset({TN})),
             Cell("u2", "w", Status.PRESENT, Grade.C, relation="equivalent",
                  flags=frozenset({SR, FLAG_CONVENTION_UNSUPPORTED})),
             Cell("u3", "w", Status.PARTIAL, Grade.A, relation="abridged", flags=frozenset({TN})),
             Cell("u4", "w", Status.PRESENT, Grade.B, relation="equivalent", flags=frozenset({"crossing"})),
             Cell("u5", "w", Status.UNALIGNED, Grade.X, reason="no_majority"),
             Cell("u6", "w", Status.LACUNA, Grade.A),
             Cell("+z1", "w", Status.NA, Grade.A, relation="addition")]
    assert convention_counts(cells) == {SR: 1, TN: 2, "unsupported": 1, "human": 1}
    assert convention_counts([]) == {SR: 0, TN: 0, "unsupported": 0, "human": 0}
    # the partition of the counts tables is unchanged by conventions
    assert human_verified_counts(cells, {}) == {"nondev": 0, "dev_present": 0, "partial": 1, "absent": 0,
                                                "unaligned": 1, "witness_only": 1}


def test_the_convention_scores_are_listed_with_n_a_for_a_source_without_conventions():
    def est(name, point):
        return Estimate(name, point, point, point, 10, "gold:dev", ("x",))
    v2 = {"relation_kappa": est("relation_kappa", 0.8), "convention_precision": est("convention_precision", 0.9),
          "convention_precision:transcribed_name": est("convention_precision:transcribed_name", 1.0),
          "dany_conv_agreement:dev": est("dany_conv_agreement:dev", 0.7)}
    v1 = {"relation_kappa": est("relation_kappa", 0.6), "convention_precision": est("convention_precision", None)}
    text = render(ReportInputs(run_id="r", reference="bo", witness="zh", scope=("s",), estimates={},
                               gold_set="dev", scores={"claude:consensus": v2, "dp:zero": v1}), gate(2))
    assert "| relation kappa, all units | 0.800 [0.800, 0.800] | 0.600 [0.600, 0.600] |" in text
    assert "| convention precision, all units | 0.900 [0.900, 0.900] | n/a |" in text
    assert "| convention precision transcribed_name | 1.000 [1.000, 1.000] | n/a |" in text
    assert "| D_any_conv agreement dev | 0.700 [0.700, 0.700] | n/a |" in text
