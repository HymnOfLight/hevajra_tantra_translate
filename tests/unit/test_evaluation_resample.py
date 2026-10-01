"""Agreement statistics, window-cluster bootstrap and McNemar (evaluation.resample)."""

from __future__ import annotations

import pytest

from hevajra_matrix.evaluation.resample import (cohen_kappa, f1, mcnemar, paired_difference, positive_agreement,
                                                resample_windows, window_bootstrap)

# 10 units: 5 (P,P), 1 (P,A), 2 (A,A), 1 (A,P), 1 (PA,PA)
PAIRS = [("P", "P")] * 5 + [("P", "A"), ("A", "A"), ("A", "A"), ("A", "P"), ("PA", "PA")]


def test_cohen_kappa_by_hand():
    # observed 0.8; chance (6*6 + 3*3 + 1*1) / 100 = 0.46
    assert cohen_kappa(PAIRS) == pytest.approx((0.8 - 0.46) / (1 - 0.46))


def test_kappa_undefined_and_unaligned_prediction():
    assert cohen_kappa([]) is None
    assert cohen_kappa([("P", "P"), ("P", "P")]) is None           # chance agreement 1
    # an unresolved prediction is a disagreement that adds nothing to chance agreement
    k = cohen_kappa([("P", "P"), ("A", "A"), ("A", "UNALIGNED")])
    assert k == pytest.approx((2 / 3 - (1 * 1 + 2 * 1) / 9) / (1 - 3 / 9))


def test_positive_agreement_by_hand():
    assert positive_agreement(PAIRS, "P") == pytest.approx(10 / 12)
    assert positive_agreement(PAIRS, "A") == pytest.approx(4 / 6)
    assert positive_agreement(PAIRS, "PA") == 1.0
    assert positive_agreement(PAIRS, "X") is None


def test_f1_edges():
    assert f1(None, 0.5) is None
    assert f1(0.0, 0.0) == 0.0
    assert f1(2 / 3, 0.5) == pytest.approx(4 / 7)


def test_resample_is_reproducible_and_with_replacement():
    a = resample_windows(["w1", "w2", "w3"], 50, seed=7)
    assert a == resample_windows(["w1", "w2", "w3"], 50, seed=7)
    assert all(len(s) == 3 for s in a) and any(len(set(s)) < 3 for s in a)
    assert resample_windows([], 5, 1) == []
    with pytest.raises(ValueError):
        resample_windows(["w1"], 0, 1)


def test_window_bootstrap_interval_contains_point():
    values = {"w1": 0.2, "w2": 0.4, "w3": 0.9, "w4": 0.5}
    point, lo, hi = window_bootstrap(lambda ws: sum(values[w] for w in ws) / len(ws), list(values), 500, seed=1)
    assert point == pytest.approx(0.5)
    assert 0.2 <= lo < point < hi <= 0.9


def test_undefined_replicates_are_dropped():
    point, lo, hi = window_bootstrap(lambda ws: None, ["w1", "w2"], 20, seed=1)
    assert (point, lo, hi) == (None, None, None)


def test_paired_difference_sign():
    better = {"w1": 0.9, "w2": 0.8, "w3": 0.85, "w4": 0.95}
    worse = {"w1": 0.5, "w2": 0.6, "w3": 0.4, "w4": 0.55}

    def metric(scores, ws):
        return sum(scores[w] for w in ws) / len(ws)

    point, lo, hi = paired_difference(metric, better, worse, list(better), 500, seed=3)
    assert point > 0 and lo > 0
    point, lo, hi = paired_difference(metric, worse, better, list(better), 500, seed=3)
    assert point < 0 and hi < 0


def test_paired_bootstrap_uses_the_same_windows_for_both():
    # identical inputs give a difference of exactly 0 in every replicate
    same = {"w1": 0.1, "w2": 0.7, "w3": 0.3}
    assert paired_difference(lambda s, ws: sum(s[w] for w in ws), same, dict(same), list(same), 100, 2) == (0, 0, 0)


def test_mcnemar_exact():
    a = {f"u{i}": True for i in range(10)}
    b = {f"u{i}": i < 2 for i in range(10)}              # 8 discordant, all favouring a
    result = mcnemar(a, b)
    assert (result.only_a, result.only_b, result.n) == (8, 0, 10)
    assert result.p_value == pytest.approx(2 / 2 ** 8)
    assert mcnemar(a, dict(a)).p_value == 1.0
    with pytest.raises(ValueError):
        mcnemar(a, {"u0": True})
