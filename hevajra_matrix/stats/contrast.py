"""Sensitive-vs-neutral contrast Delta (E4, synthesis 6.3) and its diagnostics.

    Delta = sum_c w_c (mean_{c,s} - mean_{c,n}) / sum_c w_c,   w_c = n_s n_n / (n_s + n_n)

over strata c (chapter x tertile of reference length) that contain both exposures. The
interval joins the two uncertainties that matter for a superpopulation claim about the
translation process: each two-phase draw (measurement) is combined with a resampling of
chapters with replacement (chapter clustering).

Inputs are plain mappings keyed by reference unit id, so the functions are indifferent to
where outcomes come from: two-phase draws, human-verified units only (``twophase.known_outcomes``,
the "Delta from human-verified status only" option), or naive machine labels (the attenuation
check printed beside the corrected Delta). The caller removes excluded units (frame,
mantra_control, uniform_across_list, LACUNA) from ``exposure`` before calling.

Assumption (critique B13): imputation strata are machine class x topic group, so machine
error is assumed independent of chapter and length within a stratum. ``misclassification_table``
tabulates verified error rates by chapter and length tertile so that the assumption is checked,
and ``overlap_diagnostics`` reports how many strata and units the estimator actually uses.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..core.types import Estimate, OutcomeClass, Relation, Verdict
from ..matrix.status import outcome_class
from .twophase import SAMPLE_TASKS, Column, indicator, summarise, verdict_outcome


def stratified_rd(y: Mapping[str, float], exposure: Mapping[str, bool], stratum: Mapping[str, str]) -> float | None:
    """CMH-weighted risk difference (exposed minus unexposed); None without any overlap stratum.

    Units must be in all three mappings to count.
    """
    cells: dict[str, list[list[float]]] = {}
    for unit, exposed in exposure.items():
        if unit not in y or unit not in stratum:
            continue
        row = cells.setdefault(stratum[unit], [[0.0, 0.0], [0.0, 0.0]])  # [unexposed, exposed] x [n, sum]
        side = row[1 if exposed else 0]
        side[0] += 1
        side[1] += float(y[unit])
    return _cmh(cells.values())


def _cmh(rows: Iterable[list[list[float]]]) -> float | None:
    num = den = 0.0
    for (n_n, s_n), (n_s, s_s) in rows:
        if n_n == 0 or n_s == 0:
            continue
        w = n_s * n_n / (n_s + n_n)
        num += w * (s_s / n_s - s_n / n_n)
        den += w
    return num / den if den else None


def tertile_strata(lengths: Mapping[str, float], chapter_of: Mapping[str, str]) -> dict[str, str]:
    """Stratum "chapter|t1..t3": the chapter and the tertile of reference length over the text.

    Tertile cut points are taken over all units (so "t1" means short everywhere); log length
    has the same tertiles as length.
    """
    values = sorted(lengths.values())
    if not values:
        return {}
    cut1, cut2 = values[len(values) // 3], values[(2 * len(values)) // 3]
    out = {}
    for unit, length in lengths.items():
        t = 1 if length < cut1 else 2 if length < cut2 else 3
        out[unit] = f"{chapter_of[unit]}|t{t}"
    return out


def delta(
    draw_list: Sequence[Mapping[str, OutcomeClass]], exposure: Mapping[str, bool], strata: Mapping[str, str],
    chapter_of: Mapping[str, str], seed: int, outcome: str = "any", name: str = "E4", scope: str = "",
) -> Estimate:
    """Delta with a joint imputation x chapter-bootstrap interval (2.5-97.5 percentiles).

    The point is the mean over draws of Delta on all chapters; each draw also yields one Delta
    on a with-replacement resample of chapters. A chapter drawn twice enters as two copies of
    its strata.
    """
    rng = random.Random(seed)
    units = [u for u in exposure if u in strata and u in chapter_of]
    chapters = sorted({chapter_of[u] for u in units})
    by_chapter: dict[str, list[str]] = {c: [] for c in chapters}
    for u in units:
        by_chapter[chapter_of[u]].append(u)
    points: list[float] = []
    boots: list[float] = []
    for vector in draw_list:
        y = {u: float(indicator(vector[u], outcome)) for u in units if u in vector}
        full = stratified_rd(y, exposure, strata)
        if full is None:
            continue
        points.append(full)
        rows: dict[tuple[int, str], list[list[float]]] = {}
        for copy, chapter in enumerate(rng.choice(chapters) for _ in chapters):
            for u in by_chapter[chapter]:
                if u not in y:
                    continue
                row = rows.setdefault((copy, strata[u]), [[0.0, 0.0], [0.0, 0.0]])
                side = row[1 if exposure[u] else 0]
                side[0] += 1
                side[1] += y[u]
        boot = _cmh(rows.values())
        if boot is not None:
            boots.append(boot)
    if not points or not boots:
        return Estimate.missing(name, "no stratum contains both exposures", scope)
    n_units = sum(u in draw_list[0] for u in units)
    sources = (
        "machine status error: two-phase draws",
        "chapter sampling: chapter-cluster bootstrap",
        "confounding by chapter and length: stratification",
    )
    est = summarise(boots, name, n_units, scope, sources)
    return Estimate(name=name, point=sum(points) / len(points), lo=est.lo, hi=est.hi, n=est.n,
                    scope=scope, sources=sources)


def permutation_p(
    y: Mapping[str, float], exposure: Mapping[str, bool], stratum: Mapping[str, str], n_perm: int, seed: int,
) -> float | None:
    """Two-sided p of Delta when topic labels are permuted within strata (y: posterior means)."""
    observed = stratified_rd(y, exposure, stratum)
    if observed is None:
        return None
    groups: dict[str, list[str]] = {}
    for unit in exposure:
        if unit in y and unit in stratum:
            groups.setdefault(stratum[unit], []).append(unit)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        permuted: dict[str, bool] = {}
        for members in groups.values():
            labels = [exposure[u] for u in members]
            rng.shuffle(labels)
            permuted.update(zip(members, labels))
        value = stratified_rd(y, permuted, stratum)
        hits += value is not None and abs(value) >= abs(observed) - 1e-12
    return (hits + 1) / (n_perm + 1)


def tost(estimate: Estimate, margin: float) -> bool:
    """Equivalence within +/- margin: the whole interval lies inside (-margin, margin).

    With the 2.5-97.5 interval this is TOST at alpha = 0.025 per side (conservative). A True
    result is reported as "equivalent within +/-margin", never as "no effect".
    """
    if estimate.not_estimable or estimate.lo is None or estimate.hi is None:
        return False
    return -margin < estimate.lo and estimate.hi < margin


def matched_rd(
    y: Mapping[str, float], exposure: Mapping[str, bool], group: Mapping[str, str],
    length: Mapping[str, float], tolerance: float = 0.35, seed: int = 20260930,
) -> tuple[float | None, int]:
    """Robustness check: greedy 1:1 caliper matching, returns (mean difference, pairs).

    Each exposed unit takes the first unused unexposed unit of the same ``group`` (e.g.
    chapter x reference kind) with |log length ratio| <= ``tolerance``; candidate order is
    shuffled with ``seed``.
    """
    rng = random.Random(seed)
    pool: dict[str, list[str]] = {}
    for unit in sorted(exposure):
        if not exposure[unit] and unit in y and unit in group:
            pool.setdefault(group[unit], []).append(unit)
    for members in pool.values():
        rng.shuffle(members)
    diffs: list[float] = []
    for unit in sorted(exposure):
        if not exposure[unit] or unit not in y or unit not in group:
            continue
        candidates = pool.get(group[unit], [])
        own = max(1.0, length[unit])
        for i, other in enumerate(candidates):
            if abs(math.log(max(1.0, length[other]) / own)) <= tolerance:
                diffs.append(y[unit] - y[other])
                del candidates[i]
                break
    return (sum(diffs) / len(diffs) if diffs else None), len(diffs)


@dataclass(frozen=True)
class Overlap:
    """How much of the data the stratified estimator can use (critique B13)."""

    n_strata: int
    n_overlap_strata: int      # strata that contain both exposures
    n_units: int
    n_used: int                # units in overlap strata
    dropped: tuple[str, ...]   # units in strata with only one exposure


def overlap_diagnostics(exposure: Mapping[str, bool], stratum: Mapping[str, str]) -> Overlap:
    groups: dict[str, list[str]] = {}
    for unit in exposure:
        if unit in stratum:
            groups.setdefault(stratum[unit], []).append(unit)
    dropped: list[str] = []
    overlap = 0
    for members in groups.values():
        if len({exposure[u] for u in members}) == 2:
            overlap += 1
        else:
            dropped.extend(members)
    n_units = sum(len(m) for m in groups.values())
    return Overlap(len(groups), overlap, n_units, n_units - len(dropped), tuple(sorted(dropped)))


@dataclass(frozen=True)
class ErrorRow:
    """Verified machine error rate in one imputation stratum, split by one factor."""

    stratum: str
    factor: str                # "chapter" or "tertile"
    level: str
    n: int
    errors: int

    @property
    def rate(self) -> float:
        return self.errors / self.n


def misclassification_table(
    verdicts: Iterable[Verdict], chapter_of: Mapping[str, str], tertile_of: Mapping[str, str],
    column: Column = "final", outcome: str = "any", ref_kinds: Mapping[str, str] | None = None,
) -> list[ErrorRow]:
    """Error of the machine D against verified D, by chapter and by length tertile within stratum.

    Only phase-2 sample verdicts (verify, audit) with a machine relation enter: they are the
    probability sample the imputation relies on. Roughly equal rates across levels support the
    within-stratum independence assumption; clear differences call for finer strata.
    """
    kinds = ref_kinds or {}
    acc: dict[tuple[str, str, str], list[int]] = {}
    for v in verdicts:
        if v.task not in SAMPLE_TASKS or not v.machine_relation:
            continue
        human = verdict_outcome(v, column, kinds.get(v.unit_id, ""))
        machine = _machine_outcome(v, kinds.get(v.unit_id, ""))
        if human is None or machine is None:
            continue
        error = indicator(human, outcome) != indicator(machine, outcome)
        for factor, mapping in (("chapter", chapter_of), ("tertile", tertile_of)):
            key = (v.stratum or "unstratified", factor, mapping.get(v.unit_id, "?"))
            row = acc.setdefault(key, [0, 0])
            row[0] += 1
            row[1] += error
    return [ErrorRow(s, f, lvl, n, e) for (s, f, lvl), (n, e) in sorted(acc.items())]


def _machine_outcome(v: Verdict, ref_kind: str) -> OutcomeClass | None:
    """Outcome of the machine cell the verdict was sampled from.

    The sheet's ``machine_relation`` alone loses a machine polarity flip on a non-reversal
    relation (e.g. paraphrase + flip, a machine positive in ``pos:reversal:*``). The flip is
    taken from ``machine_polarity_flip`` when recorded, else from the stratum fixed at
    sampling time, which ``stratum_of`` derived from the full machine cell.
    """
    flip = v.machine_polarity_flip or v.stratum.startswith("pos:reversal:")
    try:
        return outcome_class(Relation(v.machine_relation), flip, ref_kind)
    except ValueError:
        return None


__all__ = [
    "stratified_rd", "tertile_strata", "delta", "permutation_p", "tost", "matched_rd",
    "Overlap", "overlap_diagnostics", "ErrorRow", "misclassification_table",
]
