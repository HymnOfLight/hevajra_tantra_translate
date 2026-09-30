"""Similarity of a candidate bead for the DP baselines (``align.dp``).

A bead joins a run of reference segments to a run of witness segments. The DP adds
``-anchor_weight * score`` to the cost of a bead and ``+anchor_conflict`` when the
similarity reports a conflict, so a similarity only has to answer two questions:

    score(ref, wit)     how much content evidence the two sides share, in [0, 1]
    conflict(ref, wit)  whether both sides carry evidence and none of it is shared

Two implementations exist, which is what justifies the protocol:

    ZeroSimilarity    always 0, never a conflict: the DP then decides by lengths and
                      bead priors alone (control P1, "length only")
    AnchorSimilarity  IDF-weighted Jaccard overlap of anchor sets (baseline B0)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import AbstractSet, Collection, Iterable, Mapping, Protocol, Sequence

from ..core.types import Segment
from .anchors import AnchorLexicon, extract


class Similarity(Protocol):
    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        """Shared content evidence of a bead, in [0, 1]; both sides are non-empty."""

    def conflict(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> bool:
        """True when both sides carry evidence and none of it is shared."""


@dataclass(frozen=True)
class ZeroSimilarity:
    """No content signal at all: the length-only control P1."""

    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        return 0.0

    def conflict(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> bool:
        return False


def idf_weights(bags: Iterable[Collection[str]]) -> dict[str, float]:
    """Smoothed inverse document frequency of every key: ``ln((1 + N) / (1 + df)) + 1``.

    ``N`` is the number of bags (segments, empty ones included) and ``df`` the number of
    bags containing the key. Every weight is at least 1, so a key present in every
    segment still counts a little, and a key in one segment of 5,000 weighs about 9.
    """
    bags = list(bags)
    df = Counter(key for bag in bags for key in set(bag))
    n = len(bags)
    return {key: math.log((1 + n) / (1 + count)) + 1.0 for key, count in df.items()}


def weighted_jaccard(a: AbstractSet[str], b: AbstractSet[str], weight: Mapping[str, float],
                     default: float = 1.0) -> float:
    """Sum of weights of shared keys over sum of weights of all keys (0.0 if nothing is shared).

    A key missing from ``weight`` weighs ``default``.
    """
    shared = a & b
    if not shared:
        return 0.0
    return sum(weight.get(k, default) for k in shared) / sum(weight.get(k, default) for k in a | b)


class AnchorSimilarity:
    """Baseline B0: IDF-weighted Jaccard overlap of the anchor sets of the two sides.

    The anchor set of a side is the union over its segments (``anchors.extract``).
    Weights come from ``idf``; a key it does not list (a segment outside the fitted set)
    gets the largest listed weight, i.e. it is treated as the rarest known key.

    Anchor sets are memoised per (language, text). ``extract`` is pure, so the memo
    changes speed only, never results; it is why this class is not a frozen dataclass.
    """

    def __init__(self, lexicon: AnchorLexicon, idf: Mapping[str, float]) -> None:
        self.lexicon = lexicon
        self.idf: Mapping[str, float] = MappingProxyType(dict(idf))
        self.unseen_weight = max(self.idf.values(), default=1.0)
        self._bags: dict[tuple[str, str], frozenset[str]] = {}

    @classmethod
    def fit(cls, lexicon: AnchorLexicon, segments: Iterable[Segment]) -> "AnchorSimilarity":
        """Weights from the document frequencies over ``segments``: pass every segment of
        both witnesses that is about to be aligned, so that a key frequent on either side
        is down-weighted."""
        segments = list(segments)
        bags = [extract(s.text, s.lang, lexicon) for s in segments]
        sim = cls(lexicon, idf_weights(bags))
        sim._bags.update(((s.lang, s.text), bag) for s, bag in zip(segments, bags))
        return sim

    def bag(self, segment: Segment) -> frozenset[str]:
        key = (segment.lang, segment.text)
        found = self._bags.get(key)
        if found is None:
            found = self._bags[key] = extract(segment.text, segment.lang, self.lexicon)
        return found

    def side(self, segments: Sequence[Segment]) -> frozenset[str]:
        """Anchor set of one side of a bead (the union over its segments)."""
        if len(segments) == 1:
            return self.bag(segments[0])
        return frozenset().union(*(self.bag(s) for s in segments))

    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        return weighted_jaccard(self.side(ref), self.side(wit), self.idf, self.unseen_weight)

    def conflict(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> bool:
        a, b = self.side(ref), self.side(wit)
        return bool(a) and bool(b) and a.isdisjoint(b)
