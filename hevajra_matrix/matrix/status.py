"""Relation -> status -> outcome: the only place a cell's status is derived.

Length never enters. A deviation requires a textual relation established by a verified
machine proposal or by a human verdict:

    relation                                         status   outcome       D_any D_cov
    equivalent, paraphrase, expanded                 PRESENT  nondev        0     0
    transliterated (reference kind is mantra)        PRESENT  nondev        0     0
    transliterated (reference kind is not mantra)    PRESENT  dev_present   1     0
    generalised, substitution, category_name_omitted PRESENT  dev_present   1     0
    reversal, or any relation with polarity_flip     PRESENT  dev_present   1     0
    abridged                                         PARTIAL  partial       1     1
    no_counterpart                                   ABSENT   absent        1     1

UNALIGNED, LACUNA and NA cells have no outcome and are excluded from denominators.
"""

from __future__ import annotations

from typing import Literal

from ..core.types import COUNTED_STATUSES, Cell, OutcomeClass, Relation, Status

Outcome = Literal["any", "cov"]

RELATION_STATUS: dict[Relation, Status] = {
    Relation.EQUIVALENT: Status.PRESENT,
    Relation.PARAPHRASE: Status.PRESENT,
    Relation.EXPANDED: Status.PRESENT,
    Relation.GENERALISED: Status.PRESENT,
    Relation.SUBSTITUTION: Status.PRESENT,
    Relation.REVERSAL: Status.PRESENT,
    Relation.CATEGORY_NAME_OMITTED: Status.PRESENT,
    Relation.TRANSLITERATED: Status.PRESENT,
    Relation.ABRIDGED: Status.PARTIAL,
    Relation.NO_COUNTERPART: Status.ABSENT,
}

_REWRITING = frozenset({
    Relation.GENERALISED,
    Relation.SUBSTITUTION,
    Relation.REVERSAL,
    Relation.CATEGORY_NAME_OMITTED,
})

# Fixed tie-break order used when replicates disagree on the relation within a
# majority outcome class (most informative first).
RELATION_PRIORITY: tuple[Relation, ...] = (
    Relation.REVERSAL,
    Relation.SUBSTITUTION,
    Relation.CATEGORY_NAME_OMITTED,
    Relation.ABRIDGED,
    Relation.GENERALISED,
    Relation.TRANSLITERATED,
    Relation.PARAPHRASE,
    Relation.EXPANDED,
    Relation.EQUIVALENT,
    Relation.NO_COUNTERPART,
)


def status_of(relation: Relation) -> Status:
    return RELATION_STATUS[relation]


def outcome_class(relation: Relation, polarity_flip: bool = False, ref_kind: str = "") -> OutcomeClass:
    """Outcome of one relation; ``ref_kind`` is the reference segment kind (e.g. "mantra")."""
    if relation is Relation.NO_COUNTERPART:
        return OutcomeClass.ABSENT
    if relation is Relation.ABRIDGED:
        return OutcomeClass.PARTIAL
    if polarity_flip or relation in _REWRITING:
        return OutcomeClass.DEV_PRESENT
    if relation is Relation.TRANSLITERATED and ref_kind != "mantra":
        return OutcomeClass.DEV_PRESENT
    return OutcomeClass.NONDEV


def deviates(outcome: OutcomeClass, which: Outcome = "any") -> bool:
    """D_any: any deviation; D_cov: loss of coverage only (PARTIAL or ABSENT)."""
    if which == "cov":
        return outcome in (OutcomeClass.PARTIAL, OutcomeClass.ABSENT)
    return outcome is not OutcomeClass.NONDEV


def cell_outcome(cell: Cell, ref_kind: str = "") -> OutcomeClass | None:
    """Outcome of a matrix cell, or None when the cell is not countable."""
    if cell.status not in COUNTED_STATUSES or cell.relation is None:
        return None
    return outcome_class(Relation(cell.relation), cell.polarity_flip, ref_kind)


def cell_deviates(cell: Cell, which: Outcome = "any", ref_kind: str = "") -> bool | None:
    outcome = cell_outcome(cell, ref_kind)
    return None if outcome is None else deviates(outcome, which)
