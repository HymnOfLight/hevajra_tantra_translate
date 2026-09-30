"""Replicate consensus (k = 3 by default), evidence grades and replicate agreement.

Per reference unit (synthesis 4.2, 4.3)

    vote       each replicate contributes the outcome class of its verified link
               (``matrix.status.outcome_class``) or "unresolved"; a class, or
               "unresolved", needs at least two thirds of the k replicates
    relation   the most frequent relation among the majority replicates; ties are broken
               by ``matrix.status.RELATION_PRIORITY``
    link       the link of one majority replicate carrying that relation: the one whose
               set of linked segments has the highest mean Jaccard similarity to the other
               replicates' sets (the medoid; the earliest replicate on a tie). Taking the
               whole link from one replicate keeps relation, segments, polarity flag and
               quotes mutually consistent. Its flags are the union of every replicate's
               flags for the unit, so no finding is lost.
    UNALIGNED  a majority of unresolved replicates gives their most frequent reason (a
               unit refused by all replicates stays "refused:<category>"); no majority at
               all gives "no_majority"
    grade      X for UNALIGNED; B when all k replicates give the same class and no flag
               other than ``corroborated`` is set; C otherwise

Witness-only material is decided per witness segment by the same two-thirds rule (kind
by majority, ties in ``WitnessOnlyKind`` order) and never for a segment the consensus
links to a reference unit. Each such segment becomes its own record keyed by its orphan
row id ``+<segment id>``; the replicates' grouping and quotes stay in the replicate files.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from types import MappingProxyType
from typing import Mapping, NamedTuple, Sequence

from ..core.ids import orphan_row_id, sort_key
from ..core.types import (
    REASON_NO_MAJORITY,
    REASON_UNASSESSED,
    Alignment,
    Grade,
    Link,
    OutcomeClass,
    WitnessOnlyKind,
)
from ..matrix.status import RELATION_PRIORITY, outcome_class
from .verify import FLAG_CORROBORATED, Collation

SOURCE_CONSENSUS = "claude:consensus"
BENIGN_FLAGS = frozenset({FLAG_CORROBORATED})
UNRESOLVED = "unresolved"


class Consensus(NamedTuple):
    """Unpacks as ``alignment, grades, reasons``.

    ``grades``   reference unit id or orphan row id -> grade
    ``reasons``  reference unit id -> UNALIGNED reason (units without a consensus link)
    """

    alignment: Alignment
    grades: Mapping[str, Grade]
    reasons: Mapping[str, str]


def has_majority(votes: int, k: int) -> bool:
    """At least two thirds of the k replicates (2 of 3, 2 of 2, 1 of 1)."""
    return 3 * votes >= 2 * k


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Jaccard similarity; two empty sets (two "no counterpart" readings) agree fully."""
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def consensus(replicates: Sequence[Collation], ref_kinds: Mapping[str, str],
              source: str = SOURCE_CONSENSUS) -> Consensus:
    """Consensus of verified replicates over the units of ``ref_kinds``.

    ``ref_kinds`` maps every reference unit id to its segment kind, in reference order;
    a unit no replicate mentions is UNALIGNED(unassessed).
    """
    _check(replicates, ref_kinds)
    k = len(replicates)
    by_ref = [r.alignment.by_ref() for r in replicates]
    links: list[Link] = []
    grades: dict[str, Grade] = {}
    reasons: dict[str, str] = {}
    for uid, kind in ref_kinds.items():
        readings = [m.get(uid) for m in by_ref]
        classes = [None if link is None else outcome_class(link.relation, link.polarity_flip, kind)
                   for link in readings]
        votes = Counter(c for c in classes if c is not None)
        top = votes.most_common(1)
        if top and has_majority(top[0][1], k):
            link, unanimous = _majority_link(readings, classes, top[0][0], source), top[0][1] == k
            links.append(link)
            grades[uid] = Grade.B if unanimous and link.flags <= BENIGN_FLAGS else Grade.C
            continue
        failed = [r.unresolved.get(uid, REASON_UNASSESSED) for r, c in zip(replicates, classes) if c is None]
        reasons[uid] = Counter(failed).most_common(1)[0][0] if has_majority(len(failed), k) else REASON_NO_MAJORITY
        grades[uid] = Grade.X
    witness_only = _witness_only(replicates, {i for link in links for i in link.wit_ids}, source)
    for link, grade in witness_only:
        grades[orphan_row_id(link.wit_ids[0])] = grade
    first = replicates[0].alignment
    alignment = Alignment(source, first.reference, first.witness,
                          tuple(links) + tuple(link for link, _ in witness_only))
    return Consensus(alignment, MappingProxyType(grades), MappingProxyType(reasons))


def _majority_link(readings: Sequence[Link | None], classes: Sequence[OutcomeClass | None],
                   top: OutcomeClass, source: str) -> Link:
    members = [link for link, c in zip(readings, classes) if c == top and link is not None]
    counts = Counter(link.relation for link in members)
    best = max(counts.values())
    relation = min((rel for rel, n in counts.items() if n == best), key=RELATION_PRIORITY.index)
    resolved = [frozenset(link.wit_ids) for link in readings if link is not None]

    def centrality(link: Link) -> float:
        own = frozenset(link.wit_ids)
        others = list(resolved)
        others.remove(own)
        return sum(jaccard(own, o) for o in others) / len(others) if others else 1.0

    candidates = [link for link in members if link.relation == relation]
    medoid = max(candidates, key=centrality)          # max keeps the earliest on a tie
    flags = frozenset().union(*(link.flags for link in readings if link is not None))
    return replace(medoid, flags=flags, source=source)


def _witness_only(replicates: Sequence[Collation], linked: set[str], source: str) -> list[tuple[Link, Grade]]:
    """Per-segment witness-only records with their grades, in segment order."""
    k = len(replicates)
    kinds: dict[str, list[WitnessOnlyKind]] = {}
    flags: dict[str, set[str]] = {}
    for rep in replicates:
        seen: set[str] = set()
        for link in rep.alignment.witness_only():
            for sid in link.wit_ids:
                if sid not in seen:
                    seen.add(sid)
                    kinds.setdefault(sid, []).append(WitnessOnlyKind(link.relation))
                    flags.setdefault(sid, set()).update(link.flags)
    order = list(WitnessOnlyKind)
    out = []
    for sid in sorted(kinds, key=sort_key):
        votes = kinds[sid]
        if sid in linked or not has_majority(len(votes), k):
            continue
        counts = Counter(votes)
        best = max(counts.values())
        kind = min((x for x, n in counts.items() if n == best), key=order.index)
        unanimous = len(votes) == k and len(counts) == 1 and flags[sid] <= BENIGN_FLAGS
        out.append((Link(None, (sid,), kind, flags=frozenset(flags[sid]), source=source),
                    Grade.B if unanimous else Grade.C))
    return out


def _check(replicates: Sequence[Collation], ref_kinds: Mapping[str, str]) -> None:
    if not replicates:
        raise ValueError("consensus needs at least one replicate")
    pairs = {(r.alignment.reference, r.alignment.witness) for r in replicates}
    if len(pairs) != 1:
        raise ValueError(f"replicates collate different text pairs: {sorted(pairs)}")
    unknown = sorted({u for r in replicates for u in (*r.alignment.by_ref(), *r.unresolved)} - set(ref_kinds))
    if unknown:
        raise ValueError(f"replicates mention units missing from ref_kinds: {unknown[:5]}")


# --------------------------------------------------------------------------- agreement
@dataclass(frozen=True)
class Agreement:
    """Replicate agreement (gate G2 reads ``fleiss_kappa``).

    ``units``              units resolved by every replicate (the kappa items)
    ``fleiss_kappa``       over outcome classes; None with fewer than 2 replicates, no
                           complete unit, or a single class used throughout (undefined)
    ``mean_link_jaccard``  mean pairwise Jaccard of linked-segment sets over the units
                           resolved by at least two replicates
    ``class_jaccard``      per outcome class, mean pairwise Jaccard of the replicates' unit
                           sets in that class: e.g. whether the replicates call the SAME
                           units ABSENT, not only the same number of them
    """

    replicates: int
    units: int
    fleiss_kappa: float | None
    mean_link_jaccard: float | None
    class_jaccard: Mapping[str, float | None]


def agreement(replicates: Sequence[Collation], ref_kinds: Mapping[str, str]) -> Agreement:
    _check(replicates, ref_kinds)
    by_ref = [r.alignment.by_ref() for r in replicates]
    classes = {uid: [None if uid not in m else outcome_class(m[uid].relation, m[uid].polarity_flip, kind)
                     for m in by_ref] for uid, kind in ref_kinds.items()}
    complete = [[c.value for c in cs] for cs in classes.values() if all(c is not None for c in cs)]
    link_sims = [jaccard(frozenset(a.wit_ids), frozenset(b.wit_ids))
                 for uid in ref_kinds for a, b in combinations([m[uid] for m in by_ref if uid in m], 2)]
    per_class: dict[str, float | None] = {}
    for c in OutcomeClass:
        sets = [frozenset(uid for uid, cs in classes.items() if cs[r] == c) for r in range(len(replicates))]
        sims = [jaccard(a, b) for a, b in combinations(sets, 2) if a or b]
        per_class[c.value] = sum(sims) / len(sims) if sims else None
    return Agreement(
        replicates=len(replicates), units=len(complete),
        fleiss_kappa=fleiss_kappa(complete) if len(replicates) > 1 else None,
        mean_link_jaccard=sum(link_sims) / len(link_sims) if link_sims else None,
        class_jaccard=MappingProxyType(per_class),
    )


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float | None:
    """Fleiss' kappa for items each rated by the same number (>= 2) of raters.

    None when there is no item or chance agreement is 1 (every rating in one category).
    """
    if not ratings:
        return None
    k = len(ratings[0])
    if k < 2 or any(len(r) != k for r in ratings):
        raise ValueError("every item needs the same number (>= 2) of ratings")
    n = len(ratings)
    totals: Counter[str] = Counter()
    observed = 0.0
    for item in ratings:
        counts = Counter(item)
        totals.update(counts)
        observed += (sum(c * c for c in counts.values()) - k) / (k * (k - 1))
    p_bar = observed / n
    p_e = sum((t / (n * k)) ** 2 for t in totals.values())
    return None if p_e == 1 else (p_bar - p_e) / (1 - p_e)
