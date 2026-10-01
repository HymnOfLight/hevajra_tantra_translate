"""Two-phase estimator: strata, draws, prevalence, Manski bounds, revision rates."""

from __future__ import annotations

import random

import pytest

from hevajra_matrix.core.types import Cell, Grade, OutcomeClass, Status, Verdict
from hevajra_matrix.stats import twophase as tp


def cell(unit: str, relation: str | None = "equivalent", grade: Grade = Grade.B, status: Status | None = None,
         **kw) -> Cell:
    if status is None:
        status = {"no_counterpart": Status.ABSENT, "abridged": Status.PARTIAL}.get(relation or "", Status.PRESENT)
    return Cell(unit_id=unit, witness="T0892", status=status, grade=grade, relation=relation, **kw)


def verdict(unit: str, final: str = "", blind: str = "", task: str = "verify", stratum: str = "", **kw) -> Verdict:
    return Verdict(batch_id="b1", item_id=unit, task=task, unit_id=unit, fingerprint="f",
                   stratum=stratum, blind_relation=blind, final_relation=final, **kw)


# --------------------------------------------------------------------------- strata
@pytest.mark.parametrize("relation, flip, flags, kind, expected", [
    ("no_counterpart", False, (), "", "pos:absent:sensitive"),
    ("reversal", False, (), "", "pos:reversal:sensitive"),
    ("equivalent", True, (), "", "pos:reversal:sensitive"),
    ("substitution", False, (), "", "pos:substitution:sensitive"),
    ("category_name_omitted", False, (), "", "pos:category_name_omitted:sensitive"),
    ("generalised", False, ("relocation",), "", "pos:relocated:sensitive"),
    ("abridged", False, (), "", "pos:abridged:sensitive"),
    ("generalised", False, (), "", "pos:generalised:sensitive"),
    ("transliterated", False, (), "verse", "pos:transliterated_prose:sensitive"),
    ("transliterated", False, (), "mantra", "neg:B:sensitive"),
    ("paraphrase", False, (), "", "neg:B:sensitive"),
])
def test_stratum_of_machine_classes(relation, flip, flags, kind, expected):
    c = cell("u", relation, polarity_flip=flip, flags=frozenset(flags))
    assert tp.stratum_of(c, "sensitive", kind) == expected


def test_stratum_of_topic_grade_unresolved_excluded():
    assert tp.stratum_of(cell("u", grade=Grade.C), "neutral") == "neg:C:other"
    assert tp.stratum_of(cell("u", grade=Grade.B), "frame") == "neg:B:other"
    assert tp.stratum_of(cell("u", None, Grade.X, Status.UNALIGNED, reason="no_majority"), "sensitive") == "unresolved"
    assert tp.stratum_of(cell("u", None, Grade.B, Status.LACUNA), "neutral") == "excluded"
    with pytest.raises(ValueError):
        tp.stratum_of(cell("u", grade=Grade.A), "neutral")   # strata are computed on machine cells
    with pytest.raises(ValueError):
        tp.stratum_of(cell("u", "no_counterpart", grade=Grade.A), "neutral")


def test_census_classes_match_preregistration(repo_root):
    from hevajra_matrix.core.io import read_yaml
    spec = read_yaml(repo_root / "config" / "preregistration.yaml")["verification"]
    produced = {"absent", "reversal", "substitution", "category_name_omitted", "relocated",
                "abridged", "generalised", "transliterated_prose"}
    # witness_only and grade_x never reach stratum_of as positives (orphans; UNALIGNED)
    assert produced <= set(spec["census_classes"]) | set(spec["sampled_classes"])


def test_relocation_flag_matches_collate():
    from hevajra_matrix.collate.verify import FLAG_RELOCATION
    assert tp.FLAG_RELOCATION == FLAG_RELOCATION


# --------------------------------------------------------------------------- draws
def _population(n_pos=20, n_neg=40):
    cells = [cell(f"p{i}", "no_counterpart") for i in range(n_pos)] + [cell(f"n{i}") for i in range(n_neg)]
    strata = {c.unit_id: ("pos:absent:other" if c.unit_id.startswith("p") else "neg:B:other") for c in cells}
    return cells, strata


def test_census_stratum_adds_no_variance():
    cells, strata = _population()
    verdicts = [verdict(f"p{i}", "no_counterpart" if i % 4 else "equivalent") for i in range(20)]
    verdicts += [verdict(f"n{i}", "equivalent") for i in range(40)]
    ds = tp.draws(cells, verdicts, strata, n_draws=50, seed=1)
    est = tp.prevalence(ds, "any")
    assert est.lo == est.hi == est.point == pytest.approx(15 / 60)
    assert est.n == 60


def test_sampled_stratum_adds_variance_and_fixes_verified_units():
    cells, strata = _population()
    verdicts = [verdict(f"n{i}", "equivalent") for i in range(10)] + [verdict("p0", "no_counterpart")]
    ds = tp.draws(cells, verdicts, strata, n_draws=100, seed=2)
    assert all(d[f"n{i}"] is OutcomeClass.NONDEV for d in ds for i in range(10))
    est = tp.prevalence(ds, "any")
    assert est.hi > est.lo


def test_draws_are_deterministic_given_seed():
    cells, strata = _population()
    verdicts = [verdict("n1", "no_counterpart"), verdict("p1", "no_counterpart")]
    assert tp.draws(cells, verdicts, strata, 5, seed=3) == tp.draws(cells, verdicts, strata, 5, seed=3)


def test_gold_and_resolve_fix_units_but_do_not_calibrate():
    cells, strata = _population(n_pos=0, n_neg=5)
    verdicts = [verdict("n0", "no_counterpart", task="gold"), verdict("n1", "equivalent", task="audit")]
    x = tp.sample_counts(verdicts, strata)
    assert x == {"neg:B:other": {OutcomeClass.NONDEV: 1, OutcomeClass.DEV_PRESENT: 0,
                                 OutcomeClass.PARTIAL: 0, OutcomeClass.ABSENT: 0}}
    ds = tp.draws(cells, verdicts, strata, 20, seed=4)
    assert all(d["n0"] is OutcomeClass.ABSENT for d in ds)


def test_verdict_stratum_recorded_at_sampling_wins():
    x = tp.sample_counts([verdict("n0", "equivalent", stratum="neg:C:other")], {"n0": "neg:B:other"})
    assert set(x) == {"neg:C:other"}


def test_unaligned_and_lacuna_units_leave_the_draws_unless_resolved():
    cells = [cell("a"), cell("u", None, Grade.X, Status.UNALIGNED), cell("r", None, Grade.X, Status.UNALIGNED),
             cell("l", None, Grade.B, Status.LACUNA)]
    strata = {"a": "neg:B:other", "u": "unresolved", "r": "unresolved", "l": "excluded"}
    ds = tp.draws(cells, [verdict("r", "no_counterpart", task="resolve"), verdict("a", "equivalent", task="audit")],
                  strata, 3, seed=5)
    assert all(set(d) == {"a", "r"} for d in ds)


def test_unverified_unit_without_stratum_is_an_error():
    with pytest.raises(ValueError, match="without a sampling stratum"):
        tp.draws([cell("a")], [], {}, 1, seed=0)
    unaligned = cell("u", None, Grade.X, Status.UNALIGNED)
    assert tp.draws([unaligned], [], {}, 1, seed=0) == [{}]


def test_blind_and_final_columns_switch():
    cells = [cell("a"), cell("b")]
    strata = {"a": "neg:B:other", "b": "neg:B:other"}
    verdicts = [verdict("a", final="no_counterpart", blind="equivalent", blind_date="2026-01-01",
                        final_date="2026-01-02"),
                verdict("b", final="equivalent", blind="equivalent")]
    final = tp.prevalence(tp.draws(cells, verdicts, strata, 3, 1, column="final"))
    blind = tp.prevalence(tp.draws(cells, verdicts, strata, 3, 1, column="blind"))
    assert final.point == pytest.approx(0.5)
    assert blind.point == pytest.approx(0.0)
    with pytest.raises(ValueError):
        tp.verdict_relation(verdicts[0], "assisted")  # type: ignore[arg-type]


def test_gold_counts_for_blind_column_but_final_only_grade_a_does_not():
    cells = [cell("g", "no_counterpart", Grade.A, source="gold:test"),
             cell("v", "no_counterpart", Grade.A, source="verdict:b1")]
    assert set(tp.known_outcomes(cells, [], "blind")) == {"g"}
    assert set(tp.known_outcomes(cells, [], "final")) == {"g", "v"}


def test_latest_verdict_decides_and_gold_wins():
    vs = [verdict("a", "equivalent", final_date="2026-02-01"), verdict("a", "no_counterpart", final_date="2026-01-01")]
    assert tp.known_outcomes([cell("a")], vs)["a"] is OutcomeClass.NONDEV
    vs.append(verdict("a", "abridged", task="gold", final_date="2025-01-01"))
    assert tp.known_outcomes([cell("a")], vs)["a"] is OutcomeClass.PARTIAL


def test_verdict_transliteration_of_mantra_is_nondev():
    v = verdict("m", "transliterated")
    assert tp.verdict_outcome(v, "final", "mantra") is OutcomeClass.NONDEV
    assert tp.verdict_outcome(v, "final", "verse") is OutcomeClass.DEV_PRESENT
    assert tp.verdict_outcome(verdict("m", "lacuna"), "final") is None


# --------------------------------------------------------------------------- prevalence
def test_prevalence_outcomes_and_weights():
    d = {"a": OutcomeClass.ABSENT, "b": OutcomeClass.PARTIAL, "c": OutcomeClass.DEV_PRESENT,
         "d": OutcomeClass.NONDEV}
    assert tp.prevalence([d], "any").point == pytest.approx(0.75)
    assert tp.prevalence([d], "cov").point == pytest.approx(0.5)
    assert tp.prevalence([d], "absent").point == pytest.approx(0.25)
    w = tp.prevalence([d], "any", weights={"a": 1, "b": 1, "c": 1, "d": 5}, name="E2")
    assert w.point == pytest.approx(3 / 8) and w.name == "E2"
    assert any("segmentation" in s for s in w.sources)
    assert tp.prevalence([{}], "any").not_estimable


def test_percentile_interpolates():
    assert tp.percentile([0, 10], 50) == 5
    assert tp.percentile([3], 97.5) == 3
    with pytest.raises(ValueError):
        tp.percentile([], 50)


def test_unit_means():
    ds = [{"a": OutcomeClass.ABSENT}, {"a": OutcomeClass.NONDEV}]
    assert tp.unit_means(ds) == {"a": 0.5}


def _synthetic(rng: random.Random, n_pos=60, n_neg=120, n_sample=25):
    """Machine positives with PPV 0.7, negatives with FNR 0.1; a uniform sample verified per stratum."""
    cells, strata, truth = [], {}, {}
    for i in range(n_pos + n_neg):
        pos = i < n_pos
        uid = f"u{i}"
        cells.append(cell(uid, "no_counterpart" if pos else "equivalent"))
        strata[uid] = "pos:absent:other" if pos else "neg:B:other"
        truth[uid] = rng.random() < (0.7 if pos else 0.1)
    verdicts = []
    for group in (range(n_pos), range(n_pos, n_pos + n_neg)):
        for i in rng.sample(list(group), n_sample):
            verdicts.append(verdict(f"u{i}", "no_counterpart" if truth[f"u{i}"] else "equivalent"))
    return cells, strata, verdicts, sum(truth.values()) / len(truth)


def test_interval_coverage_of_a_planted_prevalence():
    rng = random.Random(20260930)
    covered = 0
    n_sets = 100
    for k in range(n_sets):
        cells, strata, verdicts, theta = _synthetic(rng)
        est = tp.prevalence(tp.draws(cells, verdicts, strata, n_draws=100, seed=k))
        covered += est.lo <= theta <= est.hi
    assert covered / n_sets >= 0.9


# --------------------------------------------------------------------------- Manski, revisions
def test_manski_bounds():
    cells = [cell("a", "no_counterpart"), cell("b"), cell("c"),
             cell("u", None, Grade.X, Status.UNALIGNED), cell("l", None, Grade.B, Status.LACUNA)]
    assert tp.manski(cells, "any") == (pytest.approx(1 / 4), pytest.approx(2 / 4))
    assert tp.manski([cell("a")]) == (0.0, 0.0)
    with pytest.raises(ValueError):
        tp.manski([cell("l", None, Grade.B, Status.NA)])


def test_revision_rate_per_stratum():
    vs = [verdict("a", "no_counterpart", "equivalent", stratum="neg:B:other"),
          verdict("b", "paraphrase", "equivalent", stratum="neg:B:other"),
          verdict("c", "equivalent", "equivalent", stratum="neg:B:other"),
          verdict("d", "equivalent", "", stratum="neg:C:other"),
          verdict("e", "abridged", "abridged")]
    rates = tp.revision_rate(vs)
    b = rates["neg:B:other"]
    assert (b.n, b.relation_changed, b.outcome_changed) == (3, 2, 1)
    assert b.rate == pytest.approx(1 / 3)
    assert "neg:C:other" not in rates                  # no blind decision: not a revision pair
    assert rates["unstratified"].rate == 0.0


# --------------------------------------------------------------------------- v0.3 review fixes
def test_unrevealed_sample_verdict_has_no_final_decision():
    """A verify/audit verdict imported blind but not revealed is not a final decision (A1)."""
    from hevajra_matrix.review.verdicts import decision
    v = verdict("n0", final="", blind="no_counterpart", stratum="neg:B:other")
    assert decision(v) is None
    assert tp.verdict_relation(v, "final") == "" and tp.verdict_outcome(v, "final") is None
    assert tp.verdict_relation(v, "blind") == "no_counterpart"
    assert tp.sample_counts([v], {"n0": "neg:B:other"}, "final") == {}
    assert tp.known_outcomes([cell("n0")], [v], "final") == {}
    assert tp.known_outcomes([cell("n0")], [v], "blind") == {"n0": OutcomeClass.ABSENT}
    gold = verdict("g", final="", blind="no_counterpart", task="gold")
    assert tp.verdict_relation(gold, "final") == "no_counterpart"     # gold has no reveal stage


def test_blind_column_uses_the_blind_polarity_flip():
    """The reveal import overwrites polarity_flip with the final one; the blind flip is the blind relation's."""
    v = verdict("a", final="reversal", blind="equivalent", polarity_flip=True, stratum="pos:reversal:other")
    assert tp.verdict_outcome(v, "final") is OutcomeClass.DEV_PRESENT
    assert tp.verdict_outcome(v, "blind") is OutcomeClass.NONDEV
    rev = tp.revision_rate([v])["pos:reversal:other"]
    assert (rev.relation_changed, rev.outcome_changed) == (1, 1)
    back = verdict("b", final="equivalent", blind="reversal", polarity_flip=False)
    assert tp.verdict_outcome(back, "blind") is OutcomeClass.DEV_PRESENT
    assert tp.verdict_outcome(back, "final") is OutcomeClass.NONDEV
    blind_only = verdict("c", blind="reversal", polarity_flip=True)
    assert tp.verdict_outcome(blind_only, "blind") is OutcomeClass.DEV_PRESENT
    gold = verdict("g", blind="paraphrase", task="gold", polarity_flip=True)      # gold flip on a non-reversal
    assert tp.verdict_outcome(gold, "blind") is OutcomeClass.DEV_PRESENT


def test_stratum_without_sample_verdict_is_refused_not_imputed_from_the_prior():
    cells = [cell(f"n{i}") for i in range(40)]
    # the plan stamped every audited unit 'neg:B:other'; topics imported later moved 20 units
    strata = {f"n{i}": "neg:B:sensitive" if i >= 20 else "neg:B:other" for i in range(40)}
    verdicts = [verdict(f"n{i}", "equivalent", task="audit", stratum="neg:B:other") for i in range(10)]
    assert tp.uncalibrated_strata(cells, verdicts, strata) == {"neg:B:sensitive": 20}
    with pytest.raises(tp.UncalibratedStrata, match="neg:B:sensitive") as err:
        tp.draws(cells, verdicts, strata, 10, seed=0)
    assert err.value.strata == {"neg:B:sensitive": 20}
    verdicts.append(verdict("n25", "equivalent", task="audit", stratum="neg:B:sensitive"))
    assert tp.uncalibrated_strata(cells, verdicts, strata) == {}
    assert len(tp.draws(cells, verdicts, strata, 10, seed=0)) == 10


def test_prior_symmetric_in_d_imputes_fewer_phantom_deviations():
    cells = [cell(f"n{i}") for i in range(2000)]
    strata = {c.unit_id: "neg:B:other" for c in cells}
    verdicts = [verdict(f"n{i}", "equivalent", task="audit") for i in range(40)]
    jeffreys = tp.prevalence(tp.draws(cells, verdicts, strata, 400, seed=1)).point
    symmetric = tp.prevalence(tp.draws(cells, verdicts, strata, 400, seed=1, prior="symmetric_d")).point
    assert jeffreys == pytest.approx(1.5 / 42 * 1960 / 2000, abs=0.006)
    assert symmetric == pytest.approx(0.5 / 41 * 1960 / 2000, abs=0.004)
    assert sum(tp.PRIOR_SYMMETRIC_D[1:]) == pytest.approx(tp.PRIOR_SYMMETRIC_D[0])
    with pytest.raises(ValueError, match="prior"):
        tp.draws(cells, verdicts, strata, 1, seed=1, prior="flat")
