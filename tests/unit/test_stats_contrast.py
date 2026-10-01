"""Sensitive-vs-neutral contrast: estimator, interval, tests, robustness and diagnostics."""

from __future__ import annotations

import random

import pytest

from hevajra_matrix.core.types import Cell, Estimate, Grade, OutcomeClass, Status, Verdict
from hevajra_matrix.stats import contrast as ct
from hevajra_matrix.stats import twophase as tp

A, N = OutcomeClass.ABSENT, OutcomeClass.NONDEV


def test_stratified_rd_by_hand():
    # stratum x: exposed 1/2, unexposed 0/2 -> rd .5, w 1; stratum y: exposed 1/1, unexposed 1/3 -> rd 2/3, w .75
    y = {"a": 1, "b": 0, "c": 0, "d": 0, "e": 1, "f": 1, "g": 0, "h": 0, "solo": 1}
    exposure = {"a": True, "b": True, "c": False, "d": False, "e": True, "f": False, "g": False, "h": False,
                "solo": True}
    stratum = {u: "x" for u in "abcd"} | {u: "y" for u in "efgh"} | {"solo": "z"}
    expected = (1 * 0.5 + 0.75 * (2 / 3)) / 1.75
    assert ct.stratified_rd(y, exposure, stratum) == pytest.approx(expected)
    assert ct.stratified_rd(y, {"a": True, "b": True}, stratum) is None


def test_tertile_strata():
    lengths = {f"u{i}": float(i) for i in range(9)}
    chapters = {u: ("I.1" if i < 5 else "I.2") for i, u in enumerate(lengths)}
    s = ct.tertile_strata(lengths, chapters)
    assert s["u0"] == "I.1|t1" and s["u4"] == "I.1|t2" and s["u8"] == "I.2|t3"
    assert ct.tertile_strata({}, {}) == {}


def test_tertiles_stay_meaningful_when_most_units_share_one_length():
    # Real text: 2503 of 3042 units have 7 syllables; value cut points put both cuts on 7,
    # left t2 empty and merged the 7s with the long units in t3.
    lengths = {f"s{i}": 5.0 for i in range(300)} | {f"m{i}": 7.0 for i in range(2503)} \
        | {f"l{i}": 11.0 for i in range(239)}
    s = ct.tertile_strata(lengths, {u: "I.1" for u in lengths})
    assert {s[u] for u in lengths if u[0] == "s"} == {"I.1|t1"}
    assert {s[u] for u in lengths if u[0] == "m"} == {"I.1|t2"}
    assert {s[u] for u in lengths if u[0] == "l"} == {"I.1|t3"}


def _world(rng: random.Random, n_chapters=20, per_chapter=150, p0=0.10, rd=0.11, share=0.3):
    truth, exposure, chapter = {}, {}, {}
    for c in range(n_chapters):
        base = p0 + rng.uniform(-0.04, 0.04)
        for i in range(per_chapter):
            u = f"c{c}.{i}"
            exposed = rng.random() < share
            truth[u] = rng.random() < base + (rd if exposed else 0.0)
            exposure[u], chapter[u] = exposed, f"ch{c}"
    return truth, exposure, chapter


def _true_rd(truth, exposure, chapter):
    return ct.stratified_rd({u: float(v) for u, v in truth.items()}, exposure, chapter)


def test_delta_recovers_planted_effect_with_known_outcomes():
    truth, exposure, chapter = _world(random.Random(1))
    draw = {u: (A if d else N) for u, d in truth.items()}
    est = ct.delta([draw] * 400, exposure, chapter, chapter, seed=2)
    realised = _true_rd(truth, exposure, chapter)
    assert est.point == pytest.approx(realised)
    assert est.lo < realised < est.hi
    assert est.point == pytest.approx(0.11, abs=0.05)    # planted effect, within ~3 SE of one world
    assert 0.02 < est.hi - est.lo < 0.12
    assert any("chapter" in s for s in est.sources)


def test_delta_not_estimable_without_overlap():
    est = ct.delta([{"a": A, "b": N}], {"a": True, "b": True}, {"a": "s", "b": "s"}, {"a": "c", "b": "c"}, 0)
    assert est.not_estimable and est.point is None


def test_attenuation_is_corrected_by_stratified_imputation():
    """Se .70 / Sp .90 shrink a true 0.110 to about 0.066; two-phase imputation undoes it."""
    rng = random.Random(7)
    truth, exposure, chapter = _world(rng, n_chapters=20, per_chapter=600)
    machine = {u: (rng.random() < 0.70) if d else (rng.random() >= 0.90) for u, d in truth.items()}
    true_rd = _true_rd(truth, exposure, chapter)
    naive = ct.stratified_rd({u: float(v) for u, v in machine.items()}, exposure, chapter)
    assert naive == pytest.approx(true_rd * (0.70 + 0.90 - 1), abs=0.012)
    assert naive == pytest.approx(0.066, abs=0.015)

    cells, strata = [], {}
    for u, pos in machine.items():
        cells.append(Cell(unit_id=u, witness="T0892", status=Status.ABSENT if pos else Status.PRESENT,
                          grade=Grade.B, relation="no_counterpart" if pos else "equivalent"))
        topic = "sensitive" if exposure[u] else "neutral"
        strata[u] = tp.stratum_of(cells[-1], topic)
    verdicts = []
    by_stratum: dict[str, list[str]] = {}
    for u in sorted(strata):
        by_stratum.setdefault(strata[u], []).append(u)
    for units in by_stratum.values():
        for u in rng.sample(units, 400):
            verdicts.append(Verdict(batch_id="b", item_id=u, task="audit", unit_id=u, fingerprint="f",
                                    final_relation="no_counterpart" if truth[u] else "equivalent"))
    draws = tp.draws(cells, verdicts, strata, n_draws=30, seed=3)
    est = ct.delta(draws, exposure, chapter, chapter, seed=4)
    assert est.point == pytest.approx(true_rd, abs=0.03)
    assert abs(est.point - true_rd) < abs(naive - true_rd)
    assert est.lo < true_rd < est.hi


def test_permutation_p():
    truth, exposure, chapter = _world(random.Random(11), n_chapters=10, per_chapter=100)
    y = {u: float(v) for u, v in truth.items()}
    assert ct.permutation_p(y, exposure, chapter, n_perm=200, seed=1) < 0.02
    rng = random.Random(12)
    null = {u: float(rng.random() < 0.1) for u in sorted(y)}
    assert ct.permutation_p(null, exposure, chapter, n_perm=200, seed=1) > 0.05
    assert ct.permutation_p({"a": 1.0}, {"a": True}, {"a": "s"}, 10, 0) is None


def _est(lo, hi) -> Estimate:
    return Estimate(name="E4", point=(lo + hi) / 2, lo=lo, hi=hi, n=10, scope="")


def test_tost():
    assert ct.tost(_est(-0.04, 0.05), 0.10)
    assert not ct.tost(_est(-0.04, 0.12), 0.10)
    assert not ct.tost(_est(-0.10, 0.02), 0.10)
    assert not ct.tost(Estimate.missing("E4", "G4"), 0.10)


def test_matched_rd():
    y = {"s1": 1.0, "s2": 1.0, "n1": 0.0, "n2": 1.0, "n3": 0.0, "far": 0.0}
    exposure = {"s1": True, "s2": True, "n1": False, "n2": False, "n3": False, "far": False}
    group = {"s1": "I.1|verse", "s2": "I.2|verse", "n1": "I.1|verse", "n2": "I.2|verse",
             "n3": "I.2|verse", "far": "I.1|verse"}
    length = {"s1": 10, "s2": 10, "n1": 11, "n2": 10, "n3": 100, "far": 100}
    rd, pairs = ct.matched_rd(y, exposure, group, length)
    assert pairs == 2
    assert rd == pytest.approx(((1 - 0) + (1 - 1)) / 2)
    assert ct.matched_rd({}, {}, {}, {}) == (None, 0)


def test_overlap_diagnostics():
    exposure = {"a": True, "b": False, "c": True, "d": True, "x": False}
    stratum = {"a": "s1", "b": "s1", "c": "s2", "d": "s2"}
    o = ct.overlap_diagnostics(exposure, stratum)
    assert (o.n_strata, o.n_overlap_strata, o.n_units, o.n_used) == (2, 1, 4, 2)
    assert o.dropped == ("c", "d")


def test_misclassification_table():
    def v(unit, machine, final, task="audit", stratum="neg:B:other"):
        return Verdict(batch_id="b", item_id=unit, task=task, unit_id=unit, fingerprint="f", stratum=stratum,
                       machine_relation=machine, final_relation=final)
    verdicts = [v("a", "equivalent", "no_counterpart"), v("b", "equivalent", "equivalent"),
                v("c", "equivalent", "paraphrase"), v("g", "equivalent", "no_counterpart", task="gold")]
    chapters = {"a": "I.1", "b": "I.1", "c": "I.2", "g": "I.2"}
    tertiles = {"a": "t1", "b": "t2", "c": "t2", "g": "t1"}
    rows = {(r.factor, r.level): r for r in ct.misclassification_table(verdicts, chapters, tertiles)}
    assert (rows["chapter", "I.1"].n, rows["chapter", "I.1"].errors) == (2, 1)
    assert rows["chapter", "I.2"].rate == 0.0
    assert (rows["tertile", "t2"].n, rows["tertile", "t2"].errors) == (2, 0)
    assert rows["tertile", "t1"].rate == 1.0            # the gold verdict is not in the sample



def test_misclassification_table_keeps_the_machine_polarity_flip():
    """A machine paraphrase + flip is a machine positive (pos:reversal:*), not a machine negative."""
    def v(unit, final, flip, stratum="pos:reversal:other", machine_flip=False):
        return Verdict(batch_id="b", item_id=unit, task="verify", unit_id=unit, fingerprint="f", stratum=stratum,
                       machine_relation="paraphrase", machine_polarity_flip=machine_flip,
                       blind_relation=final, final_relation=final, polarity_flip=flip)
    agree = ct.misclassification_table([v("a", "reversal", True)], {"a": "I.1"}, {"a": "t1"})
    assert all(r.errors == 0 for r in agree)                  # human confirms the machine deviation
    miss = ct.misclassification_table([v("b", "equivalent", False)], {"b": "I.1"}, {"b": "t1"})
    assert all(r.errors == 1 for r in miss)                   # human says no deviation: a machine error
    recorded = ct.misclassification_table([v("c", "reversal", True, stratum="", machine_flip=True)],
                                          {"c": "I.1"}, {"c": "t1"})
    assert all(r.errors == 0 for r in recorded)
