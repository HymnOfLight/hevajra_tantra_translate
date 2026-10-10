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

Conventions (collate.v2: a standing rendering, a name or syllable rendered by sound) never
enter ``outcome_class``: voting, strata and grades do not see them. They only enter the
sensitivity variant D_any_conv (``any_conv``), which also counts a counted unit that carries
a convention as deviating; it is never a primary outcome (``PRIMARY_OUTCOMES``).

    relation (any) + a convention                    status and outcome from the relation;
                                                     D_any, D_cov unchanged; D_any_conv = 1
"""

from __future__ import annotations

from typing import Literal

from ..core.types import COUNTED_STATUSES, Cell, Convention, OutcomeClass, Relation, Status, convention_of

Outcome = Literal["any", "cov", "any_conv"]
PRIMARY_OUTCOMES: tuple[str, ...] = ("any", "cov")    # what a preregistration may name as primary
OUTCOMES: tuple[str, ...] = ("any", "cov", "any_conv")

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


def deviates(outcome: OutcomeClass, which: Outcome = "any", convention: bool | None = None) -> bool:
    """D_any: any deviation; D_cov: loss of coverage only (PARTIAL or ABSENT); D_any_conv:
    any deviation, or a convention on the unit (``convention`` must then be given, so a
    caller that cannot know conventions fails instead of computing D_any under that name)."""
    if which == "cov":
        return outcome in (OutcomeClass.PARTIAL, OutcomeClass.ABSENT)
    if which == "any":
        return outcome is not OutcomeClass.NONDEV
    if which == "any_conv":
        if convention is None:
            raise ValueError("D_any_conv needs the unit's convention")
        return outcome is not OutcomeClass.NONDEV or convention
    raise ValueError(f"unknown outcome {which!r}; expected one of {OUTCOMES}")


def has_convention(flags: frozenset[str] | set[str]) -> bool:
    """Whether a link's or cell's flags carry a convention (collate.v2)."""
    return convention_of(flags) is not Convention.NONE


def cell_outcome(cell: Cell, ref_kind: str = "") -> OutcomeClass | None:
    """Outcome of a matrix cell, or None when the cell is not countable."""
    if cell.status not in COUNTED_STATUSES or cell.relation is None:
        return None
    return outcome_class(Relation(cell.relation), cell.polarity_flip, ref_kind)


def cell_deviates(cell: Cell, which: Outcome = "any", ref_kind: str = "") -> bool | None:
    outcome = cell_outcome(cell, ref_kind)
    return None if outcome is None else deviates(outcome, which, has_convention(cell.flags))
