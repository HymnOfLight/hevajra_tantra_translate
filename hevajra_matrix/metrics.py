"""Deviation statistics: three-way decomposition, block bootstrap, witness distances.

Implements docs/02 §5, §7, §8. Pure Python, deterministic given a seed.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .matrix import COUNTED, DIMS, Cell, WitnessMatrix

# --------------------------------------------------------------------------- decomposition
DECOMP_LABELS = ("vorlage_explained", "shared_with_cowitness", "residual")


@dataclass
class Decomposition:
    witness: str
    cowitness: str | None
    sanskrit_witnesses: list[str]
    n_deviating: int = 0
    n_counted: int = 0
    counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in DECOMP_LABELS})
    by_unit: dict[str, str] = field(default_factory=dict)

    def rates(self) -> dict[str, float]:
        d = self.n_counted or 1
        out = {k: v / d for k, v in self.counts.items()}
        out["total_deviation"] = self.n_deviating / d
        return out


def _deviates(c: Cell | None) -> bool | None:
    """True if the witness does not fully retain the unit; None if not countable."""
    if c is None or c.status not in COUNTED:
        return None
    return c.status != "PRESENT"


def decompose(matrix: WitnessMatrix, witness: str, sanskrit_witnesses: Sequence[str],
              cowitness: str | None = None, units: Iterable[str] | None = None) -> Decomposition:
    """Split each deviating unit of ``witness`` into exactly one of three classes.

    vorlage_explained      some Sanskrit witness also lacks/varies the unit (ABSENT/PARTIAL)
    shared_with_cowitness  Sanskrit unanimous, but the independent co-witness deviates too
    residual               Sanskrit unanimous and co-witness retains: only this witness deviates

    Units where the co-witness is LACUNA/UNALIGNED/NA are excluded from the
    shared/residual split and counted as ``residual`` only if no co-witness was
    requested at all (two-term decomposition, documented in the output).
    """
    dec = Decomposition(witness=witness, cowitness=cowitness, sanskrit_witnesses=list(sanskrit_witnesses))
    unit_ids = list(units) if units is not None else [u for u in matrix.ordered_units() if not u.startswith("+")]
    for u in unit_ids:
        c = matrix.get(u, witness)
        dev = _deviates(c)
        if dev is None:
            continue
        dec.n_counted += 1
        if not dev:
            continue
        dec.n_deviating += 1
        vorlage = False
        for s in sanskrit_witnesses:
            if s == matrix.reference:
                continue
            sc = matrix.get(u, s)
            if sc is not None and sc.status in ("ABSENT", "PARTIAL"):
                vorlage = True
                break
        if vorlage:
            label = "vorlage_explained"
        elif cowitness is not None:
            cw = _deviates(matrix.get(u, cowitness))
            if cw is None:
                continue  # insufficient co-witness evidence: not attributable
            label = "shared_with_cowitness" if cw else "residual"
        else:
            label = "residual"
        dec.counts[label] += 1
        dec.by_unit[u] = label
    return dec


# --------------------------------------------------------------------------- block bootstrap
def block_bootstrap(groups: dict[str, Sequence[float]], stat: Callable[[Sequence[float]], float],
                    n_boot: int = 1000, seed: int = 20260930, alpha: float = 0.05) -> dict:
    """Resample whole blocks (chapters) with replacement; return point estimate and CI.

    ``groups`` maps block id → observations in that block.
    """
    keys = [k for k, v in groups.items() if len(v) > 0]
    if not keys:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_blocks": 0}
    all_vals = [x for k in keys for x in groups[k]]
    est = stat(all_vals)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(keys) for _ in keys]
        vals = [x for k in sample for x in groups[k]]
        if vals:
            boots.append(stat(vals))
    boots.sort()
    lo = boots[int(math.floor(alpha / 2 * len(boots)))] if boots else float("nan")
    hi = boots[min(len(boots) - 1, int(math.ceil((1 - alpha / 2) * len(boots))) - 1)] if boots else float("nan")
    return {"estimate": est, "lo": lo, "hi": hi, "n_blocks": len(keys), "n_boot": len(boots)}


def rate(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def deviation_rate_by_chapter(matrix: WitnessMatrix, witness: str, status_set: Sequence[str] = ("ABSENT", "PARTIAL")) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for u in matrix.ordered_units():
        if u.startswith("+"):
            continue
        c = matrix.get(u, witness)
        if c is None or c.status not in COUNTED:
            continue
        groups.setdefault(matrix.units[u].chapter, []).append(1.0 if c.status in status_set else 0.0)
    return groups


# --------------------------------------------------------------------------- matched contrast
def matched_contrast(matrix: WitnessMatrix, witness: str, topics: dict[str, str],
                     sensitive: set[str], neutral_label: str = "neutral",
                     length_tolerance: float = 0.35, seed: int = 20260930) -> dict:
    """Greedy 1:1 matching of sensitive units to neutral units within the same chapter
    and reference kind, with |log length ratio| ≤ tolerance. Returns Δ = R_sens − R_neut
    on the matched pairs, with chapter-block bootstrap CI.
    """
    rng = random.Random(seed)
    pool: dict[tuple[str, str], list[str]] = {}
    for u, t in topics.items():
        if t == neutral_label and u in matrix.units:
            pool.setdefault((matrix.units[u].chapter, matrix.units[u].kind), []).append(u)
    for k in pool:
        rng.shuffle(pool[k])
    pairs: list[tuple[str, str]] = []
    for u, t in topics.items():
        if t not in sensitive or u not in matrix.units:
            continue
        key = (matrix.units[u].chapter, matrix.units[u].kind)
        cands = pool.get(key, [])
        lu = matrix.units[u].length
        for i, v in enumerate(cands):
            if abs(math.log(max(1, matrix.units[v].length) / max(1, lu))) <= length_tolerance:
                pairs.append((u, v))
                cands.pop(i)
                break
    diffs: dict[str, list[float]] = {}
    for u, v in pairs:
        du, dv = _deviates(matrix.get(u, witness)), _deviates(matrix.get(v, witness))
        if du is None or dv is None:
            continue
        diffs.setdefault(matrix.units[u].chapter, []).append(float(du) - float(dv))
    ci = block_bootstrap(diffs, rate, seed=seed)
    ci["n_pairs"] = sum(len(v) for v in diffs.values())
    return ci


# --------------------------------------------------------------------------- witness distance
def witness_distance(matrix: WitnessMatrix, w1: str, w2: str, weights: dict[str, float] | None = None) -> float | None:
    """docs/02 §7: status disagreement + L1 over numeric dims, on jointly countable units."""
    wts = {"status": 1.0, **{d: 1.0 for d in DIMS}}
    if weights:
        wts.update(weights)
    total, n = 0.0, 0
    for u in matrix.ordered_units():
        if u.startswith("+"):
            continue
        a, b = matrix.get(u, w1), matrix.get(u, w2)
        if a is None or b is None or a.status not in COUNTED or b.status not in COUNTED:
            continue
        d = wts["status"] * (1.0 if a.status != b.status else 0.0)
        for dim in DIMS:
            x, y = getattr(a, dim), getattr(b, dim)
            if x is not None and y is not None:
                d += wts[dim] * abs(x - y)
        total += d
        n += 1
    return total / n if n else None


def distance_matrix(matrix: WitnessMatrix, witnesses: Sequence[str] | None = None) -> tuple[list[str], list[list[float | None]]]:
    ws = list(witnesses or matrix.witnesses)
    D = [[0.0 if i == j else witness_distance(matrix, a, b) for j, b in enumerate(ws)] for i, a in enumerate(ws)]
    return ws, D


def average_linkage_newick(labels: Sequence[str], D: Sequence[Sequence[float | None]]) -> str:
    """UPGMA on a small distance matrix → Newick string (None distances treated as max)."""
    n = len(labels)
    finite = [x for row in D for x in row if x is not None]
    big = (max(finite) if finite else 1.0) * 2 + 1
    d = {(i, j): (D[i][j] if D[i][j] is not None else big) for i in range(n) for j in range(n) if i < j}
    clusters: dict[int, tuple[str, int, float]] = {i: (labels[i], 1, 0.0) for i in range(n)}
    nxt = n
    while len(clusters) > 1:
        (i, j), dij = min(d.items(), key=lambda kv: kv[1])
        ni, nj = clusters[i], clusters[j]
        h = dij / 2
        newick = f"({ni[0]}:{max(0.0, h - ni[2]):.4f},{nj[0]}:{max(0.0, h - nj[2]):.4f})"
        size = ni[1] + nj[1]
        for k in list(clusters):
            if k in (i, j):
                continue
            dik = d[(min(i, k), max(i, k))]
            djk = d[(min(j, k), max(j, k))]
            d[(min(nxt, k), max(nxt, k))] = (dik * ni[1] + djk * nj[1]) / size
        for key in [key for key in d if i in key or j in key]:
            del d[key]
        del clusters[i], clusters[j]
        clusters[nxt] = (newick, size, h)
        nxt += 1
    return next(iter(clusters.values()))[0] + ";"


# --------------------------------------------------------------------------- order statistics
def kendall_tau_distance(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    """Normalised number of discordant pairs between two rankings of the same items."""
    n = len(order_a)
    if n < 2:
        return 0.0
    pos_b = {x: i for i, x in enumerate(order_b)}
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pos_b[order_a[i]] - pos_b[order_a[j]]) > 0:
                disc += 1
    return disc / (n * (n - 1) / 2)


def summarize_dim(matrix: WitnessMatrix, witness: str, dim: str) -> dict:
    vals = [getattr(c, dim) for (u, w), c in matrix.cells.items()
            if w == witness and c.status in ("PRESENT", "PARTIAL") and getattr(c, dim) is not None and not u.startswith("+")]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals),
            "p90": sorted(vals)[int(0.9 * (len(vals) - 1))]}
