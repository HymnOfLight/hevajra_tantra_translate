"""D_any_conv (collate.v2 conventions) in the statistics: E1 only, imputed, never primary.

E1_any_conv completes the final-column draws of E1_any with conventions imputed jointly from
the phase-2 verdicts (``stats.anyconv``): verified units keep their human convention, machine
conventions never enter. Every consumer that cannot know conventions (E3, E4, the MDE,
distances) refuses ``any_conv`` instead of computing D_any under that name.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hevajra_matrix.core.types import FLAG_CONVENTION_UNSUPPORTED, Cell, Grade, OutcomeClass, Status, Verdict
from hevajra_matrix.matrix.status import PRIMARY_OUTCOMES
from hevajra_matrix.pipeline import e3 as e3_stage
from hevajra_matrix.pipeline import e4 as e4_stage
from hevajra_matrix.stats import anyconv, contrast, distance
from hevajra_matrix.stats import twophase as tp

TN, SR = "transcribed_name", "standing_rendering"


def cell(unit: str, relation: str | None = "equivalent", grade: Grade = Grade.B, status: Status | None = None,
         flags: tuple[str, ...] = (), **kw) -> Cell:
    if status is None:
        status = {"no_counterpart": Status.ABSENT, "abridged": Status.PARTIAL}.get(relation or "", Status.PRESENT)
    return Cell(unit_id=unit, witness="T0892", status=status, grade=grade, relation=relation,
                flags=frozenset(flags), **kw)


def verdict(unit: str, final: str, task: str = "audit", convention: str = "", **kw) -> Verdict:
    return Verdict(batch_id="b1", item_id=unit, task=task, unit_id=unit, fingerprint="f", final_relation=final,
                   final_convention=convention, **kw)


# --------------------------------------------------------------------------- indicator and guards
def test_indicator_any_conv_needs_the_convention():
    assert tp.indicator(OutcomeClass.NONDEV, "any_conv", True) is True
    assert tp.indicator(OutcomeClass.NONDEV, "any_conv", False) is False
    assert tp.indicator(OutcomeClass.ABSENT, "any_conv", False) is True
    assert tp.indicator(OutcomeClass.NONDEV, "any", True) is False, "a convention never enters D_any"
    assert tp.indicator(OutcomeClass.NONDEV, "cov", True) is False
    with pytest.raises(ValueError, match="convention"):
        tp.indicator(OutcomeClass.NONDEV, "any_conv")
    with pytest.raises(ValueError):
        tp.indicator(OutcomeClass.NONDEV, "conv")


def test_require_primary_accepts_only_the_preregistrable_outcomes():
    assert PRIMARY_OUTCOMES == ("any", "cov")
    assert tp.require_primary("any", "E4") == "any" and tp.require_primary("cov", "E4") == "cov"
    for outcome in ("any_conv", "absent", ""):
        with pytest.raises(ValueError, match="E4 uses the primary outcome"):
            tp.require_primary(outcome, "E4")


def test_convention_flags_do_not_move_a_unit_between_strata():
    for flags, grade in (((), Grade.B), ((TN,), Grade.B), ((SR,), Grade.B), ((TN, FLAG_CONVENTION_UNSUPPORTED), Grade.C)):
        c = cell("u", "paraphrase", grade, flags=flags)
        assert tp.stratum_of(c, "other") == f"neg:{grade}:other"
    assert tp.stratum_of(cell("u", "no_counterpart", flags=(TN,)), "other") == "pos:absent:other"


# --------------------------------------------------------------------------- E1_any_conv
def _matrix():
    """Two verified units (one human convention, one human none over a machine convention),
    and four unverified negatives, two of them with a machine-majority convention."""
    built = [cell("v1", "paraphrase", Grade.A, flags=(TN,), source="verdict:b1"),
             cell("v2", "equivalent", Grade.A, source="verdict:b1"),
             cell("n0", "paraphrase", flags=(SR,)), cell("n1", "equivalent", Grade.C, flags=(TN, FLAG_CONVENTION_UNSUPPORTED)),
             cell("n2"), cell("n3")]
    strata = {c.unit_id: "neg:B:other" for c in built}
    verdicts = [verdict("v1", "paraphrase", convention=TN), verdict("v2", "equivalent", convention="none")]
    return built, strata, verdicts


def _shares(draws, conv):
    """Per draw, (D_any share, D_any_conv share)."""
    return [(sum(c is not OutcomeClass.NONDEV for c in d.values()) / len(d),
             sum(c is not OutcomeClass.NONDEV or k[u] for u, c in d.items()) / len(d)) for d, k in zip(draws, conv)]


def test_known_conventions_are_human_only():
    built, _, verdicts = _matrix()
    # the verdict decides v1 and v2; the machine conventions of n0 and n1 never enter
    assert anyconv.known_conventions(built, verdicts) == {"v1": True, "v2": False}
    # without its verdict, a grade A cell's (human) flag decides; "" is read as none
    assert anyconv.known_conventions(built, []) == {"v1": True, "v2": False}
    blank = [verdict("v1", "paraphrase", convention="")]
    assert anyconv.known_conventions(built, blank) == {"v1": False, "v2": False}
    gold = Verdict(batch_id="g", item_id="n2", task="gold", unit_id="n2", fingerprint="f",
                   blind_relation="equivalent", blind_convention=TN)
    assert anyconv.known_conventions(built, [gold])["n2"] is True, "gold: the blind decision is its final one"
    with pytest.raises(ValueError):
        anyconv.known_conventions(built, [verdict("v1", "paraphrase", convention="bogus")])


def test_e1_any_conv_imputes_the_convention_from_the_sample_not_the_machine():
    built, strata, verdicts = _matrix()
    draws = tp.draws(built, verdicts, strata, 400, seed=7)
    conv = anyconv.convention_draws(built, verdicts, strata, draws, seed=7)
    any_ = tp.prevalence(draws, "any")
    any_conv = tp.prevalence(draws, "any_conv", name="E1_any_conv", conventions=conv)
    shares = _shares(draws, conv)
    assert any_conv.point == pytest.approx(sum(b for _, b in shares) / len(shares))
    assert all(k["v1"] and not k["v2"] for k in conv), "verified units keep the human convention"
    # the gap per draw is the share of NONDEV units with a convention: v1 always, imputed ones sometimes
    gaps = [b - a for a, b in shares]
    assert min(gaps) >= 1 / 6 - 1e-12 and max(gaps) > 1 / 6
    imputed = [k[u] for d, k in zip(draws, conv) for u in ("n0", "n1", "n2", "n3") if d[u] is OutcomeClass.NONDEV]
    # q ~ Beta(c = 1, x_NONDEV without convention + 1/2 = 1.5): mean 0.4
    assert sum(imputed) / len(imputed) == pytest.approx(0.4, abs=0.05)
    assert any_conv.point > any_.point and any_conv.name == "E1_any_conv" and any_conv.n == any_.n == 6
    assert conv == anyconv.convention_draws(built, verdicts, strata, draws, seed=7), "deterministic"


def test_joint_imputation_is_the_five_category_posterior_predictive():
    # sample: NONDEV without (1), NONDEV with a convention (2), DEV_PRESENT (1). The joint
    # Dirichlet(x + prior) with the unchanged Jeffreys prior (1/2 per class, the NONDEV one on
    # NONDEV without a convention) predicts (1.5, 2, 1.5, 0.5, 0.5) / 6 for one imputed unit.
    built = [cell("a"), cell("b", "paraphrase"), cell("c", "paraphrase"), cell("d", "substitution"), cell("u")]
    strata = {c.unit_id: "neg:B:other" for c in built}
    verdicts = [verdict("a", "equivalent", convention="none"), verdict("b", "paraphrase", convention=TN),
                verdict("c", "paraphrase", convention=SR), verdict("d", "substitution")]
    draws = tp.draws(built, verdicts, strata, 20000, seed=11)
    conv = anyconv.convention_draws(built, verdicts, strata, draws, seed=11)
    n = len(draws)
    nondev = [k["u"] for d, k in zip(draws, conv) if d["u"] is OutcomeClass.NONDEV]
    assert (len(nondev) - sum(nondev)) / n == pytest.approx(1.5 / 6, abs=0.012)
    assert sum(nondev) / n == pytest.approx(2 / 6, abs=0.012)
    assert sum(d["u"] is OutcomeClass.DEV_PRESENT for d in draws) / n == pytest.approx(1.5 / 6, abs=0.012)


def test_c12_v1_run_imputes_conventions_onto_unreviewed_transcribed_names():
    # collate.v1: 10 machine 'transliterated' prose units, 2 verified as equivalent + transcribed_name
    stratum = "pos:transliterated_prose:other"
    built = [cell(f"t{i}", "equivalent", Grade.A, flags=(TN,), source="verdict:b1") for i in range(2)]
    built += [cell(f"t{i}", "transliterated") for i in range(2, 10)]
    strata = {c.unit_id: stratum for c in built}
    verdicts = [verdict(f"t{i}", "equivalent", task="verify", convention=TN, stratum=stratum) for i in range(2)]
    draws = tp.draws(built, verdicts, strata, 2000, seed=1)
    conv = anyconv.convention_draws(built, verdicts, strata, draws, seed=1)
    any_ = tp.prevalence(draws, "any")
    any_conv = tp.prevalence(draws, "any_conv", conventions=conv)
    # D_any: imputed units deviate with probability 1.5/4; D_any_conv: all but NONDEV without a
    # convention, (2.5/4) * E[1 - q] with q ~ Beta(2, 1/2): 0.625 * 0.2 = 0.125
    assert any_.point == pytest.approx(8 * 0.375 / 10, abs=0.02)
    assert any_conv.point == pytest.approx((2 + 8 * 0.875) / 10, abs=0.02)
    # plugging in the (absent) v1 machine conventions gave E1_any + 0.2, about 0.5
    assert any_conv.point > 0.85


def test_v2_machine_conventions_the_human_rejects_are_calibrated_away():
    # every machine cell claims a transcribed name; the three verified units are none
    built = [cell(f"m{i}", "equivalent", flags=(TN,)) for i in range(7)]
    built += [cell(f"h{i}", "equivalent", Grade.A, source="verdict:b1") for i in range(3)]
    strata = {c.unit_id: "neg:B:other" for c in built}
    verdicts = [verdict(f"h{i}", "equivalent", convention="none") for i in range(3)]
    draws = tp.draws(built, verdicts, strata, 300, seed=5)
    conv = anyconv.convention_draws(built, verdicts, strata, draws, seed=5)
    assert not any(any(k.values()) for k in conv)
    a = tp.prevalence(draws, "any")
    b = tp.prevalence(draws, "any_conv", conventions=conv)
    assert (a.point, a.lo, a.hi, a.n) == (b.point, b.lo, b.hi, b.n)
    assert tp.manski(built, "any_conv") == (pytest.approx(0.7), pytest.approx(0.7)), "Manski reads the (v2) cells"


def test_e1_any_conv_equals_e1_any_without_conventions():
    # no verified unit and no sample verdict records a convention: equal draw by draw,
    # whatever the machine cells say
    built, strata, verdicts = _matrix()
    v1 = [Cell(c.unit_id, c.witness, c.status, c.grade, relation=c.relation, source=c.source) for c in built]
    plain = [verdict("v1", "paraphrase"), verdict("v2", "equivalent")]
    machine_conventions = [v1[0], v1[1], *built[2:]]          # n0, n1 carry machine conventions
    for cells in (v1, machine_conventions):
        draws = tp.draws(cells, plain, strata, 100, seed=3)
        conv = anyconv.convention_draws(cells, plain, strata, draws, seed=3)
        a = tp.prevalence(draws, "any", name="E1")
        b = tp.prevalence(draws, "any_conv", name="E1", conventions=conv)
        assert (a.point, a.lo, a.hi, a.n, a.sources) == (b.point, b.lo, b.hi, b.n, b.sources)
        assert anyconv.summary(cells, plain, strata) == {"verdicts": 2, "unannotated": 2, "with_convention": 0,
                                                         "nondev_with_convention": 0, "verified_with_convention": 0,
                                                         "verified_unannotated": 0, "verified_nondev_unannotated": 0}


def test_summary_counts_the_verdicts_without_an_annotated_convention():
    built, strata, verdicts = _matrix()
    extra = [verdict("n2", "equivalent"), verdict("n3", "no_counterpart", convention=""),
             verdict("n0", "paraphrase", task="gold", convention=TN),          # gold: not a sample verdict
             verdict("n1", "abridged", convention=SR)]
    assert anyconv.summary(built, verdicts + extra, strata) == {
        "verdicts": 5, "unannotated": 2, "with_convention": 2, "nondev_with_convention": 1,
        "verified_with_convention": 3,    # v1, the gold n0 and the audited n1
        "verified_unannotated": 0, "verified_nondev_unannotated": 0}    # sample verdicts are counted above
    assert anyconv.convention_counts(verdicts + extra, strata) == {"neg:B:other": (2, 1)}


def test_summary_counts_the_units_gold_or_resolve_decided_without_a_convention():
    built, strata, verdicts = _matrix()
    built[2] = cell("n0", "paraphrase", Grade.A, source="gold:dev_r1")
    gold = [Verdict(batch_id="g", item_id="gold:n0", task="gold", unit_id="n0", fingerprint="f",
                    blind_relation="paraphrase")]                                  # a '?' left on the gold sheet
    resolved = [verdict("n2", "equivalent", task="resolve"), verdict("n3", "no_counterpart", task="resolve"),
                verdict("n1", "abridged", task="resolve", convention="none")]
    out = anyconv.summary(built, verdicts + resolved, strata, gold=gold)
    assert (out["verified_unannotated"], out["verified_nondev_unannotated"]) == (3, 2)   # n0, n2; n3 deviates
    assert out["unannotated"] == 0 and out["verified_with_convention"] == 1, "they count as none"
    assert anyconv.summary(built, verdicts + resolved, strata)["verified_unannotated"] == 2, "no gold verdicts given"


def test_summary_takes_the_deciding_verdict_known_conventions_reads():
    # a unit with gold and a later review verdict: the review verdict decides, as in E1_any_conv
    built, strata, verdicts = _matrix()
    built[2] = cell("n0", "paraphrase", Grade.A, source="gold:dev_r1")
    unannotated = [Verdict(batch_id="g", item_id="gold:n0", task="gold", unit_id="n0", fingerprint="f",
                           blind_relation="paraphrase")]
    audited = verdicts + [verdict("n0", "paraphrase", convention=TN)]
    out = anyconv.summary(built, audited, strata, gold=unannotated)
    assert anyconv.known_conventions(built, audited)["n0"] is True
    assert (out["verified_with_convention"], out["verified_unannotated"]) == (2, 0)     # v1, n0
    annotated = [replace(unannotated[0], blind_convention=TN)]
    resolved = verdicts + [verdict("n0", "paraphrase", task="resolve")]
    out = anyconv.summary(built, resolved, strata, gold=annotated)
    assert anyconv.known_conventions(built, resolved)["n0"] is False
    assert (out["verified_with_convention"], out["verified_unannotated"]) == (1, 1)     # v1; n0


def test_e1_any_conv_refuses_missing_conventions():
    draws = [{"a": OutcomeClass.NONDEV}]
    with pytest.raises(ValueError, match="convention"):
        tp.prevalence(draws, "any_conv")
    with pytest.raises(ValueError, match="one map per draw"):
        tp.prevalence(draws, "any_conv", conventions={"a": True})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one map per draw"):
        tp.prevalence(draws, "any_conv", conventions=[])
    with pytest.raises(KeyError):
        tp.prevalence(draws, "any_conv", conventions=[{}])


def test_manski_any_conv_reads_the_cell_flags():
    cells = [cell("a", "no_counterpart"), cell("b", flags=(TN,)), cell("c"), cell("d", "paraphrase", flags=(SR,)),
             cell("u", None, Grade.X, Status.UNALIGNED), cell("l", None, Grade.B, Status.LACUNA, flags=(TN,))]
    assert tp.manski(cells, "any") == (pytest.approx(1 / 5), pytest.approx(2 / 5))
    assert tp.manski(cells, "any_conv") == (pytest.approx(3 / 5), pytest.approx(4 / 5))
    assert tp.manski(cells, "cov") == tp.manski([cell(c.unit_id, c.relation, c.grade, c.status) for c in cells], "cov")


# --------------------------------------------------------------------------- E2-E4, distances: primary only
def test_e4_functions_refuse_any_conv():
    draws = [{"a": OutcomeClass.NONDEV, "b": OutcomeClass.ABSENT}]
    with pytest.raises(ValueError, match="primary outcome"):
        tp.unit_means(draws, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):
        contrast.delta(draws, {"a": True, "b": False}, {"a": "s", "b": "s"}, {"a": "c", "b": "c"}, 0, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):     # refused even with nothing to compute
        contrast.delta([], {}, {}, {}, 0, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):
        contrast.misclassification_table([], {}, {}, outcome="any_conv")
    assert tp.unit_means(draws, "cov") == {"a": 0.0, "b": 1.0}


def test_pipeline_e3_e4_refuse_any_conv():
    with pytest.raises(ValueError, match="E3 uses the primary outcome"):
        e3_stage.unit_evidence(None, [], None, "any_conv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="primary outcome"):
        e4_stage.machine_indicator([cell("a", flags=(TN,))], {}, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):
        e4_stage.mde_on_labels(SimpleNamespace(), [], {}, "any_conv", seed=0)  # type: ignore[arg-type]
    assert e4_stage.machine_indicator([cell("a", flags=(TN,)), cell("b", "abridged")], {}, "any") == {"a": 0.0, "b": 1.0}


def test_distance_ignores_conventions_and_refuses_unknown_weights():
    cells = [Cell("u1", w, Status.PRESENT, Grade.B, relation="equivalent", flags=frozenset(f))
             for w, f in (("w1", ()), ("w2", (TN,)))]
    assert distance.witness_distance(cells, "w1", "w2") == 0.0, "conventions are not a distance dimension"
    with pytest.raises(ValueError, match="unknown distance weight"):
        distance.witness_distance(cells, "w1", "w2", weights={"convention": 1.0})
    assert distance.witness_distance(cells, "w1", "w2", weights={"outcome": 2.0, "d_len": 0.5}) == 0.0
