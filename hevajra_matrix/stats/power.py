"""Planning power by simulation (synthesis 6.5 and 0.2), pure Python.

``simulate_delta`` is re-run on the human topic labels and the dev-gold error rates before
``prereg freeze``; the MDE it gives is what the preregistration stores. The planning tables in
the synthesis came from keyword-proxy topic shares and are illustrative only.
"""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

Z975 = 1.959963984540054
Z95 = 1.6448536269514722


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def t_quantile_975(df: int) -> float:
    """0.975 quantile of Student's t (Cornish-Fisher expansion; error < 0.002 for df >= 5)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    z = Z975
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def _cmh(rows: Sequence[tuple[int, int, int, int]]) -> float:
    """CMH risk difference over (n_s, y_s, n_n, y_n) rows."""
    weights = [s * n / (s + n) for s, _, n, _ in rows]
    return sum(w * (ys / s - yn / n) for w, (s, ys, n, yn) in zip(weights, rows)) / sum(weights)


def _observed(rng: random.Random, k: int, p_true: float, se: float, sp: float) -> int:
    """Positives reported by an instrument with sensitivity ``se`` and specificity ``sp``."""
    y = 0
    for _ in range(k):
        truth = rng.random() < p_true
        y += (rng.random() < se) if truth else (rng.random() >= sp)
    return y


def simulate_delta(
    chapter_sizes: Sequence[int], sens_share: Sequence[float], p0: float, delta: float,
    se: float = 1.0, sp: float = 1.0, n_sim: int = 600, seed: int = 1,
    sd_chapter: float = 0.5, sd_chapter_topic: float = 0.3,
) -> float:
    """Power of the naive (machine-label) stratified Delta test at two-sided alpha 0.05.

    Per chapter: a logit random intercept (``sd_chapter``) and a random topic effect around
    logit(p0 + delta) - logit(p0) (``sd_chapter_topic``); outcomes are then observed through an
    instrument with ``se``/``sp``. The estimator is the chapter-stratified CMH risk difference,
    tested with a jackknife-over-chapters t statistic. Imperfect ``se``/``sp`` shows the power
    lost to non-differential error when it is not corrected.
    """
    if len(chapter_sizes) != len(sens_share):
        raise ValueError("chapter_sizes and sens_share differ in length")
    rng = random.Random(seed)
    a = logit(p0)
    b = logit(min(0.95, p0 + delta)) - a
    hits = 0
    for _ in range(n_sim):
        rows = []
        for size, share in zip(chapter_sizes, sens_share):
            n_s = max(0, round(size * min(0.9, share)))
            n_n = size - n_s
            base = rng.gauss(0.0, sd_chapter)
            effect = b + rng.gauss(0.0, sd_chapter_topic)
            y_s = _observed(rng, n_s, expit(a + base + effect), se, sp)
            y_n = _observed(rng, n_n, expit(a + base), se, sp)
            if n_s > 0 and n_n > 0:
                rows.append((n_s, y_s, n_n, y_n))
        k = len(rows)
        if k < 3:
            continue
        estimate = _cmh(rows)
        leave_one_out = [_cmh(rows[:i] + rows[i + 1:]) for i in range(k)]
        mean = sum(leave_one_out) / k
        se_jk = math.sqrt((k - 1) / k * sum((x - mean) ** 2 for x in leave_one_out))
        if se_jk > 0 and abs(estimate / se_jk) > t_quantile_975(k - 1):
            hits += 1
    return hits / n_sim


def _mean_var(xs: Sequence[float]) -> tuple[float, float]:
    """Mean and the variance of that mean."""
    m = sum(xs) / len(xs)
    return m, sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)


def _z(num: float, var: float) -> float:
    return num / math.sqrt(var) if var > 0 else 0.0


def simulate_experiment(
    n_items: int, reps: int, rates: Mapping[str, float], sd_item: float = 1.0,
    n_sim: int = 1500, seed: int = 7,
) -> dict[str, float]:
    """Power of H1-H3 of the over-attribution experiment (synthesis 8).

    ``rates``: P(Y_over) for keys "S_E0", "S_EW", "N_E0", "N_EW" (arm S = sensitive,
    N = neutral; E0 = no evidence, EW = witness evidence). Items carry a logit random effect
    (``sd_item``) shared by their conditions; replicates are averaged within item x condition.

    H1 sensitive arm, E0 > EW (paired);  H2 under E0, S > N;  Holm over H1-H2, one-sided 0.05.
    H3 difference-in-differences, one-sided 0.05, exploratory (not in the Holm family).
    """
    rng = random.Random(seed)
    h1 = h2 = h3 = 0
    for _ in range(n_sim):
        e0: dict[str, list[float]] = {}
        diff: dict[str, list[float]] = {}
        for arm in ("S", "N"):
            la, lb = logit(rates[f"{arm}_E0"]), logit(rates[f"{arm}_EW"])
            e0[arm], diff[arm] = [], []
            for _item in range(n_items):
                u = rng.gauss(0.0, sd_item)
                ya = sum(rng.random() < expit(la + u) for _ in range(reps)) / reps
                yb = sum(rng.random() < expit(lb + u) for _ in range(reps)) / reps
                e0[arm].append(ya)
                diff[arm].append(ya - yb)
        m_s, v_s = _mean_var(diff["S"])
        m_n, v_n = _mean_var(diff["N"])
        a_s, av_s = _mean_var(e0["S"])
        a_n, av_n = _mean_var(e0["N"])
        z1, z2 = _z(m_s, v_s), _z(a_s - a_n, av_s + av_n)
        reject = _holm_one_sided((z1, z2))
        h1 += reject[0]
        h2 += reject[1]
        h3 += _z(m_s - m_n, v_s + v_n) > Z95
    return {"H1": h1 / n_sim, "H2": h2 / n_sim, "H3": h3 / n_sim}


def _holm_one_sided(zs: Sequence[float]) -> list[bool]:
    """Holm at one-sided alpha 0.05 for two z statistics (thresholds z_.975 then z_.95)."""
    order = sorted(range(len(zs)), key=lambda i: -zs[i])
    thresholds = (Z975, Z95)
    reject = [False] * len(zs)
    for rank, i in enumerate(order):
        if zs[i] <= thresholds[rank]:
            break
        reject[i] = True
    return reject
