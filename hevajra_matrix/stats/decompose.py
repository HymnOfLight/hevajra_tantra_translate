"""E3: where do the Chinese deviations come from? (synthesis 6.4, critique A2 and A3)

Every deviating unit (D = 1) gets exactly one class, the first that applies:

    attested_vorlage   a translator note on the host segment states what the Sanskrit
                       manuscript had (note class ``vorlage_statement``)
    vorlage_ms         at least one informative Sanskrit *manuscript* lacks or varies the unit
                       (printed editions are not independent and are never passed in here)
    insufficient       fewer than ``m_min`` informative manuscripts, or the co-witness is not
                       countable at this unit
    shared             the co-witness deviates too; labelled ``shared_revised`` while the
                       co-witness is the revised Derge (then shared is a lower bound)
    residual           nothing above explains the deviation

The check order puts ``insufficient`` before ``shared`` because a missing manuscript or
co-witness reading cannot support either of the two remaining classes; the report prints the
classes in ``CLASS_ORDER``. Class counts always sum to the number of deviating units (asserted).

``attested_translator`` (note class ``substitution_instruction``) is translator-side evidence:
it is a flag on residual units and never moves a unit out of the residual (A3).

Excess over chance (A2). For an indicator I (A: a manuscript lacks the unit; T: the co-witness
deviates) ``excess`` prints the observed co-occurrence O, its expectation under independence
E = sum_c n_c p_c q_c (chapter-stratified when chapters are given), O - E, and

    phi = (P(I | D=1) - P(I | D=0)) / (1 - P(I | D=0))

the excess fraction over the independence base rate. The circular-shift null and
per-manuscript odds ratios are deferred until manuscript columns exist.

Current state: the reference is Derge, which is also the only co-witness, and there is no
manuscript column, so ``e3`` returns ``NOT_ESTIMABLE: G4: no manuscript column; reference is
the co-witness``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Container, Mapping, Sequence

from ..core.types import Estimate

VORLAGE_NOTE = "vorlage_statement"
TRANSLATOR_NOTE = "substitution_instruction"
INFORMATIVE = frozenset({"present", "absent", "variant"})   # readings that say something
LACKS_OR_VARIES = frozenset({"absent", "variant"})
CLASS_ORDER = ("attested_vorlage", "vorlage_ms", "shared", "residual", "insufficient")
REVISED_LABEL = "shared_revised"


@dataclass(frozen=True)
class UnitEvidence:
    """What is known about one countable reference unit of the witness under study.

    ``manuscripts``: manuscript id -> present | absent | variant | illegible | not_collated.
    ``cowitness_deviates``: None when the co-witness cell is not countable.
    """

    unit_id: str
    deviates: bool
    note_classes: frozenset[str] = frozenset()
    manuscripts: Mapping[str, str] = field(default_factory=dict)
    cowitness_deviates: bool | None = None
    chapter: str = ""

    @property
    def informative(self) -> int:
        return sum(r in INFORMATIVE for r in self.manuscripts.values())

    @property
    def ms_lacks(self) -> bool:
        return any(r in LACKS_OR_VARIES for r in self.manuscripts.values())


@dataclass(frozen=True)
class Decomposition:
    counts: Mapping[str, int]              # keys in CLASS_ORDER, "shared" renamed when revised
    by_unit: Mapping[str, str]
    translator_flagged: tuple[str, ...]    # residual units with a substitution_instruction note
    n_deviating: int
    cowitness_revised: bool


def shared_label(cowitness_revised: bool) -> str:
    return REVISED_LABEL if cowitness_revised else "shared"


def classify(unit: UnitEvidence, m_min: int = 2) -> str:
    """Class of one deviating unit (with the plain "shared" label)."""
    if not unit.deviates:
        raise ValueError(f"{unit.unit_id}: only deviating units are decomposed")
    if VORLAGE_NOTE in unit.note_classes:
        return "attested_vorlage"
    if unit.ms_lacks:
        return "vorlage_ms"
    if unit.informative < m_min or unit.cowitness_deviates is None:
        return "insufficient"
    return "shared" if unit.cowitness_deviates else "residual"


def decompose(units: Sequence[UnitEvidence], m_min: int = 2, cowitness_revised: bool = True) -> Decomposition:
    """Exhaustive, ordered classification of the deviating units; non-deviating units are skipped."""
    label = shared_label(cowitness_revised)
    counts = {(label if c == "shared" else c): 0 for c in CLASS_ORDER}
    by_unit: dict[str, str] = {}
    flagged: list[str] = []
    for unit in units:
        if not unit.deviates:
            continue
        cls = classify(unit, m_min)
        if cls == "residual" and TRANSLATOR_NOTE in unit.note_classes:
            flagged.append(unit.unit_id)
        cls = label if cls == "shared" else cls
        counts[cls] += 1
        by_unit[unit.unit_id] = cls
    n_dev = sum(u.deviates for u in units)
    assert sum(counts.values()) == n_dev == len(by_unit), "decomposition classes must be exhaustive"
    return Decomposition(MappingProxyType(counts), MappingProxyType(by_unit), tuple(flagged), n_dev, cowitness_revised)


def excess_fraction(case_rate: float, base_rate: float) -> float | None:
    """(a - b) / (1 - b): share of cases explained beyond the base rate; None if b == 1."""
    return None if base_rate >= 1.0 else (case_rate - base_rate) / (1.0 - base_rate)


@dataclass(frozen=True)
class Excess:
    """Co-occurrence of D = 1 with an indicator I, against independence."""

    n: int
    observed: int               # O: units with D = 1 and I = 1
    expected: float             # E = sum_c n_c p_c q_c
    case_rate: float | None     # P(I | D = 1)
    base_rate: float | None     # P(I | D = 0)
    phi: float | None

    @property
    def excess(self) -> float:
        """O - E."""
        return self.observed - self.expected


def excess(dev: Mapping[str, bool], ind: Mapping[str, bool], chapter_of: Mapping[str, str] | None = None) -> Excess:
    """O, E, O - E and phi over the units present in both mappings."""
    units = [u for u in dev if u in ind]
    strata: dict[str, list[str]] = {}
    for u in units:
        strata.setdefault(chapter_of[u] if chapter_of else "", []).append(u)
    expected = 0.0
    for members in strata.values():
        n_c = len(members)
        p = sum(dev[u] for u in members) / n_c
        q = sum(ind[u] for u in members) / n_c
        expected += n_c * p * q
    cases = [ind[u] for u in units if dev[u]]
    controls = [ind[u] for u in units if not dev[u]]
    case_rate = sum(cases) / len(cases) if cases else None
    base_rate = sum(controls) / len(controls) if controls else None
    phi = excess_fraction(case_rate, base_rate) if case_rate is not None and base_rate is not None else None
    return Excess(len(units), sum(cases), expected, case_rate, base_rate, phi)


def _chapters(units: Sequence[UnitEvidence]) -> dict[str, str]:
    return {u.unit_id: u.chapter for u in units}


def vorlage_excess(units: Sequence[UnitEvidence], m_min: int = 2) -> Excess:
    """I = A: some informative manuscript lacks or varies the unit (units with enough evidence)."""
    usable = [u for u in units if u.ms_lacks or u.informative >= m_min]
    return excess({u.unit_id: u.deviates for u in usable}, {u.unit_id: u.ms_lacks for u in usable}, _chapters(units))


def shared_excess(units: Sequence[UnitEvidence], m_min: int = 2) -> Excess:
    """I = T: the co-witness deviates, among units whose manuscripts all retain the unit (A = 0)."""
    usable = [u for u in units if u.informative >= m_min and not u.ms_lacks and u.cowitness_deviates is not None]
    return excess(
        {u.unit_id: u.deviates for u in usable}, {u.unit_id: bool(u.cowitness_deviates) for u in usable},
        _chapters(units),
    )


def not_estimable_reason(manuscripts: Sequence[str], reference: str, cowitness: str | None) -> str | None:
    """Gate G4 for E3: a manuscript column and a co-witness distinct from the reference."""
    problems = []
    if not manuscripts:
        problems.append("no manuscript column")
    if cowitness is None:
        problems.append("no co-witness")
    elif cowitness == reference:
        problems.append("reference is the co-witness")
    return "G4: " + "; ".join(problems) if problems else None


def bound_unverified_shared(units: Sequence[UnitEvidence], verified: Container[str]) -> list[UnitEvidence]:
    """A2 sensitivity: unverified deviating units count as not shared (lower bound on shared)."""
    return [
        replace(u, cowitness_deviates=False) if u.deviates and u.cowitness_deviates and u.unit_id not in verified else u
        for u in units
    ]


@dataclass(frozen=True)
class E3Result:
    decomposition: Decomposition
    vorlage: Excess             # I = some informative manuscript lacks or varies the unit
    shared: Excess              # I = co-witness deviates, among units with no manuscript evidence

    def estimates(self, scope: str = "") -> tuple[Estimate, Estimate]:
        """phi_V and phi_S as point estimates (their null intervals are deferred)."""
        return (_phi("E3.phi_V", self.vorlage, scope), _phi("E3.phi_S", self.shared, scope))


def _phi(name: str, ex: Excess, scope: str) -> Estimate:
    if ex.phi is None:
        return Estimate.missing(name, "no cases or no controls", scope)
    return Estimate(name=name, point=ex.phi, lo=None, hi=None, n=ex.n, scope=scope,
                    sources=("independence base rate P(I | D = 0)",))


def e3(
    units: Sequence[UnitEvidence], *, manuscripts: Sequence[str], reference: str, cowitness: str | None,
    m_min: int = 2, cowitness_revised: bool = True, scope: str = "",
) -> E3Result | Estimate:
    """The E3 decomposition with its excess statistics, or ``Estimate.missing`` when G4 fails."""
    reason = not_estimable_reason(manuscripts, reference, cowitness)
    if reason:
        return Estimate.missing("E3", reason, scope)
    return E3Result(
        decomposition=decompose(units, m_min, cowitness_revised),
        vorlage=vorlage_excess(units, m_min),
        shared=shared_excess(units, m_min),
    )
