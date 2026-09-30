"""Planning power: reproduces the synthesis 6.5 and 0.2 tables at small n_sim (loose tolerance)."""

from __future__ import annotations

import pytest

from hevajra_matrix.stats import power

# Planning inputs of synthesis 6.5: provisional chapter sizes and keyword-proxy sensitive shares.
SIZES = [163, 124, 91, 13, 93, 109, 121, 226, 88, 173, 58, 56, 255, 295, 437, 295, 50, 58, 49, 181, 23, 60, 27]
SHARES = [.19, .23, .12, .29, .09, .05, .09, .09, .10, .03, .02, .09, .05, .11, .09, .09, .10, .16, .09, .08,
          .22, .08, .07]
RATES = {"S_E0": 0.6, "S_EW": 0.3, "N_E0": 0.4, "N_EW": 0.25}


@pytest.mark.parametrize("df, expected", [(5, 2.5706), (10, 2.2281), (22, 2.0739), (60, 2.0003)])
def test_t_quantile(df, expected):
    assert power.t_quantile_975(df) == pytest.approx(expected, abs=0.002)


@pytest.mark.parametrize("p0, delta, se, sp, table", [
    (0.05, 0.05, 1.0, 1.0, 0.76),
    (0.05, 0.10, 0.85, 0.97, 0.95),
    (0.10, 0.10, 0.70, 0.90, 0.60),
])
def test_simulate_delta_reproduces_planning_table(p0, delta, se, sp, table):
    assert power.simulate_delta(SIZES, SHARES, p0, delta, se, sp, n_sim=300, seed=1) == pytest.approx(table, abs=0.08)


def test_simulate_delta_sanity():
    assert power.simulate_delta(SIZES, SHARES, 0.10, 0.0, n_sim=150, seed=1) < 0.12   # size, not power
    with pytest.raises(ValueError):
        power.simulate_delta([1, 2], [0.1], 0.1, 0.1)


@pytest.mark.parametrize("n_items, h1, h2, h3", [(40, 0.99, 0.73, 0.46), (100, 1.00, 0.98, 0.79)])
def test_simulate_experiment_reproduces_planning_table(n_items, h1, h2, h3):
    got = power.simulate_experiment(n_items, 3, RATES, n_sim=250, seed=7)
    assert got["H1"] == pytest.approx(h1, abs=0.06)
    assert got["H2"] == pytest.approx(h2, abs=0.09)
    assert got["H3"] == pytest.approx(h3, abs=0.09)


def test_simulate_experiment_null_and_holm():
    null = {"S_E0": 0.4, "S_EW": 0.4, "N_E0": 0.4, "N_EW": 0.4}
    got = power.simulate_experiment(30, 3, null, n_sim=200, seed=2)
    assert max(got.values()) < 0.12
    assert power._holm_one_sided([2.5, 1.7]) == [True, True]
    assert power._holm_one_sided([1.9, 1.7]) == [False, False]      # first step needs z > 1.96
    assert power._holm_one_sided([1.0, 2.0]) == [False, True]
