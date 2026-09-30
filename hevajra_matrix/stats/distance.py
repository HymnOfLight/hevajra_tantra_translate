"""Witness distances and an average-linkage (UPGMA) tree, descriptive only.

Kept from v0.2 with minimal changes (impl decisions): the distance between two witnesses is
the mean, over reference units countable in both, of an outcome-class disagreement (via
``matrix.status``) plus the L1 difference of the descriptive dimensions d_len, d_ord, d_lit.
A tree over two witnesses says nothing, so ``distance_matrix`` needs at least three.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..core.types import Cell
from ..matrix.status import cell_outcome

DIMS = ("d_len", "d_ord", "d_lit")
MIN_WITNESSES = 3


def _by_unit(cells: Iterable[Cell], witness: str) -> dict[str, Cell]:
    return {c.unit_id: c for c in cells if c.witness == witness and not c.unit_id.startswith("+")}


def witness_distance(
    cells: Iterable[Cell], w1: str, w2: str, weights: Mapping[str, float] | None = None,
    ref_kinds: Mapping[str, str] | None = None,
) -> float | None:
    """Mean per-unit distance over jointly countable units; None if there are none.

    ``weights`` may override "outcome" and any of ``DIMS`` (default 1.0 each). A missing
    dimension value on either side contributes nothing for that unit.
    """
    cells = list(cells)
    wts = {"outcome": 1.0, **{d: 1.0 for d in DIMS}, **(weights or {})}
    kinds = ref_kinds or {}
    a_cells, b_cells = _by_unit(cells, w1), _by_unit(cells, w2)
    total, n = 0.0, 0
    for unit in sorted(set(a_cells) & set(b_cells)):
        a, b = a_cells[unit], b_cells[unit]
        oa, ob = cell_outcome(a, kinds.get(unit, "")), cell_outcome(b, kinds.get(unit, ""))
        if oa is None or ob is None:
            continue
        d = wts["outcome"] * (oa is not ob)
        for dim in DIMS:
            x, y = getattr(a, dim), getattr(b, dim)
            if x is not None and y is not None:
                d += wts[dim] * abs(x - y)
        total += d
        n += 1
    return total / n if n else None


def distance_matrix(
    cells: Iterable[Cell], witnesses: Sequence[str], ref_kinds: Mapping[str, str] | None = None,
) -> tuple[list[str], list[list[float | None]]]:
    """Symmetric witness distance matrix (labels, rows); refuses fewer than three witnesses."""
    ws = list(witnesses)
    if len(ws) < MIN_WITNESSES:
        raise ValueError(f"witness distances need >= {MIN_WITNESSES} witnesses, got {len(ws)}")
    cells = list(cells)
    rows: list[list[float | None]] = [[0.0] * len(ws) for _ in ws]
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            rows[i][j] = rows[j][i] = witness_distance(cells, ws[i], ws[j], ref_kinds=ref_kinds)
    return ws, rows


def average_linkage_newick(labels: Sequence[str], dist: Sequence[Sequence[float | None]]) -> str:
    """UPGMA on a small distance matrix -> Newick string (None distances treated as max * 2 + 1)."""
    n = len(labels)
    finite = [x for row in dist for x in row if x is not None]
    big = (max(finite) if finite else 1.0) * 2 + 1
    d = {(i, j): (dist[i][j] if dist[i][j] is not None else big) for i in range(n) for j in range(n) if i < j}
    clusters: dict[int, tuple[str, int, float]] = {i: (labels[i], 1, 0.0) for i in range(n)}
    nxt = n
    while len(clusters) > 1:
        (i, j), dij = min(d.items(), key=lambda kv: (kv[1], kv[0]))
        ni, nj = clusters[i], clusters[j]
        h = dij / 2
        newick = f"({ni[0]}:{max(0.0, h - ni[2]):.4f},{nj[0]}:{max(0.0, h - nj[2]):.4f})"
        size = ni[1] + nj[1]
        for k in clusters:
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
