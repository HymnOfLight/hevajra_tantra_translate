"""Monotone bead alignment with NULL beads: the content-free and lexical baselines.

A dynamic programme over the reference and witness segments of one chapter group
(Gale & Church 1993, extended by a similarity term). Beads: 1:1, 1:0, 0:1, 1:2, 2:1,
2:2, 1:3, 3:1. The cost of a bead is

    prior(shape)
    + length_weight * 0.5 * ((l_wit - c * l_ref) / sqrt(length_var * l_ref)) ** 2
    - anchor_weight * similarity.score(ref run, wit run)
    + anchor_conflict                 if similarity.conflict(ref run, wit run)

where the length and similarity terms apply only to beads with both sides non-empty,
lengths are ``core.textnorm.length`` (at least 1 per segment) and ``c`` is the ratio of
total witness length to total reference length within the group. With
``ZeroSimilarity`` this is the length-only control P1; with ``AnchorSimilarity`` it is
the baseline B0 (synthesis 5.3).

The DP is monotone by construction, so it cannot represent transpositions: it exists
only as a baseline for the Claude instrument, never to build the matrix.

Output is one ``Alignment`` of the shared type:
    n:m bead with n, m >= 1   one link per reference unit, relation ``equivalent``, every
                              unit of the bead carrying the same ``wit_ids``
    1:0 bead                  ``no_counterpart`` link with no witness ids
    0:1 bead                  witness-only link (``ref_id`` None), kind ``addition``
Links are emitted in bead order, so witness-only links sit between their neighbours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Collection, Iterator, Mapping, Sequence

from ..config import ConfigError, from_mapping
from ..core.textnorm import length
from ..core.types import Alignment, Link, Relation, Segment, WitnessOnlyKind
from .similarity import Similarity

BEADS: tuple[tuple[int, int], ...] = ((1, 1), (1, 0), (0, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
SOURCE_ZERO = "dp:zero"       # P1: ZeroSimilarity
SOURCE_ANCHOR = "dp:anchor"   # B0: AnchorSimilarity

# A chapter group: reference chapter keys (Segment.chapter) and witness-local chapter keys
# (Segment.local_chapter) that are aligned as one sequence pair.
ChapterGroup = tuple[Collection[str], Collection[str]]


@dataclass(frozen=True)
class DPParams:
    """Bead priors (negative log probabilities) and cost weights; ``run.yaml`` section ``align``.

    The defaults are those of v0.2, chosen on a small sweep over four chapters; they are
    to be re-tuned on dev gold and frozen in the pre-registration.
    """

    prior_11: float = 0.0
    prior_null: float = 3.0        # 1:0 and 0:1
    prior_12: float = 2.4          # 1:2 and 2:1
    prior_22: float = 4.5
    prior_13: float = 4.0          # 1:3 and 3:1
    length_weight: float = 1.0
    length_var: float = 6.8        # Gale-Church variance per unit of reference length
    anchor_weight: float = 8.0     # reward per unit of similarity
    anchor_conflict: float = 2.0   # penalty when the similarity reports a conflict

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ConfigError(f"align.{name} must be a finite number, got {value!r}")
        if self.length_var <= 0:
            raise ConfigError("align.length_var must be positive")
        if min(self.length_weight, self.anchor_weight, self.anchor_conflict) < 0:
            raise ConfigError("align weights must be non-negative")

    @classmethod
    def from_config(cls, run: Mapping[str, Any]) -> "DPParams":
        """Build from the parsed ``run.yaml`` (``Settings.run``); unknown keys raise."""
        return from_mapping(cls, run.get("align"), "run.yaml: align")

    def prior(self, shape: tuple[int, int]) -> float:
        di, dj = shape
        if (di, dj) == (1, 1):
            return self.prior_11
        if di == 0 or dj == 0:
            return self.prior_null
        if (di, dj) == (2, 2):
            return self.prior_22
        return self.prior_12 if max(di, dj) == 2 else self.prior_13


@dataclass(frozen=True)
class Bead:
    """Indices of the reference and witness segments joined by one bead."""

    ref: tuple[int, ...]
    wit: tuple[int, ...]


# --------------------------------------------------------------------------- the DP
def beads(ref: Sequence[Segment], wit: Sequence[Segment], sim: Similarity, params: DPParams) -> list[Bead]:
    """Minimum-cost bead sequence covering both segment lists in order.

    Ties go to the bead listed first in ``BEADS``, so the result is deterministic.
    """
    n, m = len(ref), len(wit)
    lr = [max(1, length(s.text, s.lang)) for s in ref]
    lw = [max(1, length(s.text, s.lang)) for s in wit]
    ratio = sum(lw) / sum(lr) if lr and lw else 1.0
    pr, pw = _prefix(lr), _prefix(lw)
    priors = {shape: params.prior(shape) for shape in BEADS}
    inf = math.inf
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    step: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            best, best_shape = cost[i][j], step[i][j]
            for shape in BEADS:
                di, dj = shape
                if di > i or dj > j or cost[i - di][j - dj] == inf:
                    continue
                total = cost[i - di][j - dj] + priors[shape]
                if di and dj:
                    total += params.length_weight * _length_cost(pr[i] - pr[i - di], pw[j] - pw[j - dj],
                                                                 ratio, params.length_var)
                    rs, ws = ref[i - di:i], wit[j - dj:j]
                    total -= params.anchor_weight * sim.score(rs, ws)
                    if params.anchor_conflict and sim.conflict(rs, ws):
                        total += params.anchor_conflict
                if total < best:
                    best, best_shape = total, shape
            cost[i][j], step[i][j] = best, best_shape
    return _backtrace(step, n, m)


def _prefix(values: Sequence[int]) -> list[int]:
    out = [0]
    for v in values:
        out.append(out[-1] + v)
    return out


def _length_cost(l_ref: int, l_wit: int, ratio: float, var: float) -> float:
    delta = (l_wit - ratio * l_ref) / math.sqrt(var * l_ref)
    return 0.5 * delta * delta


def _backtrace(step: Sequence[Sequence[tuple[int, int] | None]], n: int, m: int) -> list[Bead]:
    out: list[Bead] = []
    i, j = n, m
    while (i, j) != (0, 0):
        shape = step[i][j]
        if shape is None:  # unreachable: 1:0 and 0:1 beads reach every cell
            raise RuntimeError(f"DP backtrace failed at ({i}, {j})")
        di, dj = shape
        out.append(Bead(ref=tuple(range(i - di, i)), wit=tuple(range(j - dj, j))))
        i, j = i - di, j - dj
    out.reverse()
    return out


# --------------------------------------------------------------------------- to Alignment
def links(ref: Sequence[Segment], wit: Sequence[Segment], bead_list: Sequence[Bead], source: str) -> Iterator[Link]:
    """Links of a bead sequence, in bead order (see the module docstring)."""
    for bead in bead_list:
        wit_ids = tuple(wit[j].id for j in bead.wit)
        if not bead.ref:
            yield Link(ref_id=None, wit_ids=wit_ids, relation=WitnessOnlyKind.ADDITION, source=source)
            continue
        relation = Relation.EQUIVALENT if wit_ids else Relation.NO_COUNTERPART
        for i in bead.ref:
            yield Link(ref_id=ref[i].id, wit_ids=wit_ids, relation=relation, source=source)


def align(ref: Sequence[Segment], wit: Sequence[Segment], sim: Similarity, params: DPParams, *,
          source: str, reference: str, witness: str) -> Alignment:
    """Align one pair of segment sequences (one chapter group), in the order given."""
    return Alignment(source=source, reference=reference, witness=witness,
                     links=tuple(links(ref, wit, beads(ref, wit, sim, params), source)))


def align_groups(ref: Sequence[Segment], wit: Sequence[Segment], groups: Sequence[ChapterGroup],
                 sim: Similarity, params: DPParams, *, source: str, reference: str, witness: str) -> Alignment:
    """Align each chapter group independently and concatenate the links in group order.

    A group selects the reference segments whose ``chapter`` and the witness segments whose
    ``local_chapter`` is among its keys, keeping the input order. Groups are computed by
    the caller (from the concordance); a key may occur in one group only, so no segment
    is aligned twice. Segments outside every group get no link. Pass alignable segments
    only (content kinds; not notes, heads, colophons or paratext).
    """
    ref_seen: set[str] = set()
    wit_seen: set[str] = set()
    for ref_keys, wit_keys in groups:
        for keys, seen, side in ((ref_keys, ref_seen, "reference"), (wit_keys, wit_seen, "witness")):
            if isinstance(keys, str):
                raise TypeError(f"{side} chapter keys must be a collection of strings, got the string {keys!r}")
            repeated = seen & set(keys)
            if repeated:
                raise ValueError(f"{side} chapter key(s) {sorted(repeated)} occur in more than one group")
            seen.update(keys)
    out: list[Link] = []
    for ref_keys, wit_keys in groups:
        ref_keys, wit_keys = frozenset(ref_keys), frozenset(wit_keys)
        rs = [s for s in ref if s.chapter in ref_keys]
        ws = [s for s in wit if s.local_chapter in wit_keys]
        out.extend(links(rs, ws, beads(rs, ws, sim, params), source))
    return Alignment(source=source, reference=reference, witness=witness, links=tuple(out))
