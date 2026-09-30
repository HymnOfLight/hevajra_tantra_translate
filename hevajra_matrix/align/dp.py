"""Chapter-gated monotone alignment with NULL beads (B0 baseline).

Dynamic programme over two segment sequences of one reference chapter and
the corresponding witness chapter. Beads allowed: 1:1, 1:0, 0:1, 1:2, 2:1,
2:2, 1:3, 3:1. Cost = length term (Gale–Church) + anchor term + bead prior.

The similarity backend is pluggable: the default uses cross-lingual anchors
(terms, names, numerals) only; an embedding backend can be dropped in
without touching the DP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from . import normalize
from .anchors import AnchorLexicon, anchor_similarity
from .segments import Segment


class SimilarityBackend(Protocol):
    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        """Return similarity in [0, 1] for a candidate bead."""


class AnchorBackend:
    def __init__(self, lexicon: AnchorLexicon) -> None:
        self.lex = lexicon
        self._cache: dict[str, set[str]] = {}

    def anchors(self, s: Segment) -> set[str]:
        key = f"{s.witness}|{s.seg_id}"
        if key not in self._cache:
            self._cache[key] = self.lex.extract(s.text, s.lang)
        return self._cache[key]

    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        a: set[str] = set()
        b: set[str] = set()
        for s in ref:
            a |= self.anchors(s)
        for s in wit:
            b |= self.anchors(s)
        return anchor_similarity(a, b)

    def has_anchors(self, segs: Sequence[Segment]) -> bool:
        return any(self.anchors(s) for s in segs)


@dataclass
class AlignParams:
    # Gale–Church style priors (negative log probabilities)
    # Defaults were chosen on a small sweep over four chapters of T0892 ↔ Toh 417–418
    # (anchor agreement vs. NULL count); they must be re-tuned on the pilot gold
    # alignment and then frozen in the pre-registration.
    prior_11: float = 0.0
    prior_null: float = 3.0      # 1:0 and 0:1
    prior_12: float = 2.4        # 1:2 and 2:1
    prior_22: float = 4.5
    prior_13: float = 4.0        # 1:3 and 3:1
    length_weight: float = 1.0
    length_var: float = 6.8      # Gale–Church s² (per reference length unit)
    anchor_weight: float = 8.0   # reward for shared anchors
    anchor_conflict: float = 2.0 # penalty when both sides carry anchors but share none
    beads: tuple[tuple[int, int], ...] = ((1, 1), (1, 0), (0, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))


@dataclass
class Bead:
    ref: list[int]
    wit: list[int]
    cost: float
    similarity: float

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ref), len(self.wit)


def _lengths(segs: Sequence[Segment]) -> list[int]:
    return [max(1, normalize.length(s.text, s.lang)) for s in segs]


def expected_ratio(ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
    lr, lw = sum(_lengths(ref)), sum(_lengths(wit))
    return (lw / lr) if lr else 1.0


def length_cost(lr: int, lw: int, c: float, s2: float) -> float:
    if lr == 0 or lw == 0:
        return 0.0
    mean = c * lr
    delta = (lw - mean) / math.sqrt(s2 * lr)
    return 0.5 * delta * delta


def align_chapter(ref: Sequence[Segment], wit: Sequence[Segment], backend: SimilarityBackend,
                  params: AlignParams | None = None, ratio: float | None = None) -> list[Bead]:
    """Align one reference chapter to one witness chapter; returns beads in order."""
    p = params or AlignParams()
    n, m = len(ref), len(wit)
    if n == 0 and m == 0:
        return []
    c = ratio if ratio is not None else expected_ratio(ref, wit)
    lr, lw = _lengths(ref), _lengths(wit)
    pref = [0] + [sum(lr[:i + 1]) for i in range(n)]
    pwit = [0] + [sum(lw[:j + 1]) for j in range(m)]
    prior = {(1, 1): p.prior_11, (1, 0): p.prior_null, (0, 1): p.prior_null,
             (1, 2): p.prior_12, (2, 1): p.prior_12, (2, 2): p.prior_22,
             (1, 3): p.prior_13, (3, 1): p.prior_13}
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    has_anchors = getattr(backend, "has_anchors", None)
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] == INF:
                continue
            for di, dj in p.beads:
                ii, jj = i + di, j + dj
                if ii > n or jj > m:
                    continue
                cost = prior[(di, dj)]
                sim = 0.0
                if di and dj:
                    cost += p.length_weight * length_cost(pref[ii] - pref[i], pwit[jj] - pwit[j], c, p.length_var)
                    rs, ws = ref[i:ii], wit[j:jj]
                    sim = backend.score(rs, ws)
                    cost -= p.anchor_weight * sim
                    if sim == 0.0 and has_anchors and has_anchors(rs) and has_anchors(ws):
                        cost += p.anchor_conflict
                total = D[i][j] + cost
                if total < D[ii][jj]:
                    D[ii][jj] = total
                    back[ii][jj] = (di, dj, sim)
    beads: list[Bead] = []
    i, j = n, m
    while (i, j) != (0, 0):
        step = back[i][j]
        if step is None:
            raise RuntimeError("alignment backtrace failed")
        di, dj, sim = step
        beads.append(Bead(ref=list(range(i - di, i)), wit=list(range(j - dj, j)),
                          cost=D[i][j] - D[i - di][j - dj], similarity=sim))
        i, j = i - di, j - dj
    beads.reverse()
    return beads


def alignment_summary(beads: list[Bead]) -> dict:
    shapes: dict[str, int] = {}
    for b in beads:
        k = f"{b.shape[0]}:{b.shape[1]}"
        shapes[k] = shapes.get(k, 0) + 1
    return {"beads": len(beads), "shapes": shapes,
            "ref_null": sum(1 for b in beads if b.shape[1] == 0),
            "wit_null": sum(1 for b in beads if b.shape[0] == 0)}
