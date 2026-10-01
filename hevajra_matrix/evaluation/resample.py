"""Agreement statistics and window-cluster resampling for gold scoring (synthesis 5.2).

Gold is annotated in windows of contiguous units, and errors within a window are not
independent (one misplaced link shifts its neighbours). Every interval is therefore a
cluster bootstrap over windows: a replicate draws as many windows as there are, with
replacement, and recomputes the metric on the pooled units of the drawn windows (a window
drawn twice counts twice). Comparisons between two aligners are paired: both are scored on
the same resampled windows, so the interval is for the difference itself (critique A7;
two separate intervals that merely fail to overlap would be needlessly conservative).

Everything here is pure and deterministic given a seed; stdlib only. None stands for an
undefined value (an empty denominator, a kappa with chance agreement 1). Replicates in which
the metric is undefined are dropped; if none is left the interval is (None, None).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping, Sequence, TypeVar

from ..stats.twophase import percentile

T = TypeVar("T")
Interval = tuple[float | None, float | None, float | None]     # (point, lo, hi)
CONFIDENCE = 95.0


# --------------------------------------------------------------------------- agreement
def cohen_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> float | None:
    """Cohen's kappa of (gold label, predicted label) pairs.

    A predicted label that gold never uses (e.g. "UNALIGNED") is allowed: it adds nothing to
    chance agreement and counts as a disagreement, so an aligner cannot raise its kappa by
    leaving hard units unresolved. None when there is no pair or chance agreement is 1.
    """
    n = len(pairs)
    if n == 0:
        return None
    gold = Counter(g for g, _ in pairs)
    pred = Counter(p for _, p in pairs)
    observed = sum(1 for g, p in pairs if g == p) / n
    chance = sum(gold[k] * pred.get(k, 0) for k in gold) / (n * n)
    return None if chance >= 1.0 else (observed - chance) / (1.0 - chance)


def positive_agreement(pairs: Sequence[tuple[Hashable, Hashable]], label: Hashable) -> float | None:
    """Specific (positive) agreement for one class: 2 n_kk / (n_gold_k + n_pred_k).

    Reported beside kappa (critique A7) because kappa is depressed when one class dominates
    (the prevalence paradox); per-class agreement shows where the aligner actually fails.
    """
    both = sum(1 for g, p in pairs if g == label and p == label)
    total = sum(1 for g, _ in pairs if g == label) + sum(1 for _, p in pairs if p == label)
    return 2 * both / total if total else None


def ratio(hits: int, n: int) -> float | None:
    return hits / n if n else None


def f1(precision: float | None, recall: float | None) -> float | None:
    """Harmonic mean; None when either part is undefined, 0 when both are 0."""
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- bootstrap
def resample_windows(windows: Sequence[T], n_boot: int, seed: int) -> list[list[T]]:
    """``n_boot`` bootstrap samples of the windows (with replacement, same size)."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    ordered = list(windows)
    if not ordered:
        return []
    rng = random.Random(seed)
    return [rng.choices(ordered, k=len(ordered)) for _ in range(n_boot)]


def percentile_interval(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    """The 95% percentile interval of replicate values, undefined (None) replicates dropped."""
    kept = [v for v in values if v is not None]
    if not kept:
        return None, None
    tail = (100.0 - CONFIDENCE) / 2
    return percentile(kept, tail), percentile(kept, 100.0 - tail)


def window_bootstrap(metric_fn: Callable[[Sequence[T]], float | None], windows: Sequence[T],
                     n_boot: int, seed: int) -> Interval:
    """(point, lo, hi): the metric on all windows and its 95% window-cluster percentile interval.

    ``metric_fn`` receives a sequence of window ids (with repeats in a replicate) and returns
    the metric computed on the pooled units of those windows.
    """
    point = metric_fn(list(windows))
    lo, hi = percentile_interval(metric_fn(sample) for sample in resample_windows(windows, n_boot, seed))
    return point, lo, hi


def paired_difference(metric_fn: Callable[[T, Sequence[Hashable]], float | None], a: T, b: T,
                      windows: Sequence[Hashable], n_boot: int, seed: int) -> Interval:
    """(point, lo, hi) of ``metric(a) - metric(b)`` under a paired window-cluster bootstrap.

    ``metric_fn(x, windows)`` scores ``x`` (e.g. an ``AlignmentScores``) on the given
    windows. Both aligners are scored on the SAME resampled windows in every replicate.
    """
    def diff(sample: Sequence[Hashable]) -> float | None:
        x, y = metric_fn(a, sample), metric_fn(b, sample)
        return None if x is None or y is None else x - y

    return window_bootstrap(diff, windows, n_boot, seed)


# --------------------------------------------------------------------------- McNemar
@dataclass(frozen=True)
class McNemar:
    """Exact McNemar test on paired correctness.

    ``only_a``  units a got right and b got wrong; ``only_b`` the reverse. ``p_value`` is the
    two-sided exact binomial p-value of the discordant split (1.0 without discordant pairs).
    """

    n: int
    only_a: int
    only_b: int
    p_value: float


def mcnemar(correct_a: Mapping[str, bool], correct_b: Mapping[str, bool]) -> McNemar:
    """Compare two aligners unit by unit; both mappings must cover the same units."""
    if set(correct_a) != set(correct_b):
        raise ValueError("McNemar needs both aligners scored on the same units")
    only_a = sum(1 for u, ok in correct_a.items() if ok and not correct_b[u])
    only_b = sum(1 for u, ok in correct_a.items() if not ok and correct_b[u])
    return McNemar(n=len(correct_a), only_a=only_a, only_b=only_b, p_value=_binomial_two_sided(only_a, only_b))


def _binomial_two_sided(x: int, y: int) -> float:
    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
