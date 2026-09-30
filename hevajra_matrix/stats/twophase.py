"""Two-phase estimation: machine strata, human verification, posterior-predictive imputation.

One mechanism serves every estimand (synthesis 6.2). Phase 1 is the machine matrix; every
countable unit falls into a stratum fixed at sampling time (``stratum_of``). Phase 2 is a
uniform sample of each stratum's unverified pool, verified by humans (``verify`` and
``audit`` verdicts). ``draws`` then completes the outcome vector M times:

    verified units                       fixed at their human outcome in every draw
    unverified units of stratum h        p_h ~ Dirichlet(x_h + 1/2)  (Jeffreys prior)
                                         each unit ~ Categorical(p_h), independently

where x_h counts the outcome classes among the *sampled* verdicts of stratum h. Drawing each
unit from Categorical(p_h) is the same as Multinomial(N_h - n_h, p_h) counts assigned at
random within the stratum, and it gives per-unit draws for the contrast (E4). A stratum whose
units are all verified (a census) contributes no variance by construction. Units that stay
UNALIGNED after human resolution are left out of every draw; ``manski`` bounds them.

Assumption (stated, see ``contrast.misclassification_table`` for the check): within a stratum,
machine error does not depend on chapter or unit length.

Verdict columns (critique A1): ``column="final"`` is primary; ``column="blind"`` repeats the
estimate with the verdicts made before the annotator saw the machine output, which is the
pre-registered automation-bias sensitivity. ``revision_rate`` reports blind -> final changes.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from ..core.types import COUNTED_STATUSES, Cell, Estimate, Grade, OutcomeClass, Relation, Status, Verdict
from ..matrix.status import cell_outcome, deviates, outcome_class

Column = Literal["final", "blind"]

OUTCOMES: tuple[OutcomeClass, ...] = (
    OutcomeClass.NONDEV, OutcomeClass.DEV_PRESENT, OutcomeClass.PARTIAL, OutcomeClass.ABSENT,
)
UNRESOLVED = "unresolved"          # UNALIGNED cells: never imputed, bounded by ``manski``
EXCLUDED = "excluded"              # LACUNA / NA cells: outside every denominator
SAMPLE_TASKS = frozenset({"verify", "audit"})   # verdicts drawn by the phase-2 probability sample
FLAG_RELOCATION = "relocation"     # same value as collate.verify.FLAG_RELOCATION

# Verification classes of machine positives (config/preregistration.yaml ``verification``).
# ``witness_only`` claims are orphan rows, not reference units, so they never reach here;
# grade X cells are UNALIGNED and therefore "unresolved" until a human resolves them.
_SENSITIVE = "sensitive"

SOURCES = (
    "machine status error: two-phase verification + posterior predictive",
    "calibration uncertainty: Dirichlet(x_h + 1/2) draws per stratum",
    "Claude stochasticity: grade is a stratum",
    "differential error by topic: topic group is a stratum",
)


def topic_stratum(topic_group: str) -> str:
    """Two-level topic factor of the strata: "sensitive" or "other"."""
    return _SENSITIVE if topic_group == _SENSITIVE else "other"


def positive_class(cell: Cell, ref_kind: str = "") -> str:
    """Verification class of a machine-positive cell (census classes win over sampled ones)."""
    relation = Relation(cell.relation) if cell.relation else None
    if relation is Relation.NO_COUNTERPART:
        return "absent"
    if cell.polarity_flip or relation is Relation.REVERSAL:
        return "reversal"
    if relation in (Relation.SUBSTITUTION, Relation.CATEGORY_NAME_OMITTED):
        return str(relation)
    if FLAG_RELOCATION in cell.flags:
        return "relocated"
    if relation in (Relation.ABRIDGED, Relation.GENERALISED):
        return str(relation)
    if relation is Relation.TRANSLITERATED and ref_kind != "mantra":
        return "transliterated_prose"
    raise ValueError(f"{cell.unit_id}: relation {cell.relation!r} is not a machine positive")


def stratum_of(cell: Cell, topic_group: str, ref_kind: str = "") -> str:
    """Sampling stratum of one *machine* cell (call it before human verdicts are applied).

    ``pos:<class>:<topic>`` machine positive (census or sampled class x topic group)
    ``neg:<grade>:<topic>`` machine negative, grade B or C x topic group
    ``unresolved``          UNALIGNED (includes every grade X cell)
    ``excluded``            LACUNA or NA
    """
    if cell.status is Status.UNALIGNED:
        return UNRESOLVED
    if cell.status not in COUNTED_STATUSES:
        return EXCLUDED
    if cell.grade not in (Grade.B, Grade.C):
        raise ValueError(f"{cell.unit_id}: grade {cell.grade} on a countable cell; strata use machine cells")
    outcome = cell_outcome(cell, ref_kind)
    if outcome is None:
        raise ValueError(f"{cell.unit_id}: countable cell without a relation")
    topic = topic_stratum(topic_group)
    if deviates(outcome):
        return f"pos:{positive_class(cell, ref_kind)}:{topic}"
    return f"neg:{cell.grade}:{topic}"


def verdict_relation(verdict: Verdict, column: Column) -> str:
    """The relation a verdict records in ``column``; "" when that column is empty.

    A verdict without a reveal stage (gold) has only a blind decision, which is then also its
    final one.
    """
    if column == "final":
        return verdict.final_relation or verdict.blind_relation
    if column == "blind":
        return verdict.blind_relation
    raise ValueError(f"column must be 'final' or 'blind', not {column!r}")


def verdict_outcome(verdict: Verdict, column: Column, ref_kind: str = "") -> OutcomeClass | None:
    """Outcome class of a verdict, or None when the column holds no relation (e.g. "lacuna")."""
    value = verdict_relation(verdict, column)
    try:
        relation = Relation(value)
    except ValueError:
        return None
    return outcome_class(relation, verdict.polarity_flip, ref_kind)


def _latest(verdicts: Iterable[Verdict], column: Column) -> dict[str, Verdict]:
    """Per unit, the verdict that decides it: gold first, then the latest date, then list order."""
    best: dict[str, Verdict] = {}
    for v in verdicts:
        if not verdict_relation(v, column):
            continue
        old = best.get(v.unit_id)
        if old is None or _rank(v, column) >= _rank(old, column):
            best[v.unit_id] = v
    return best


def _rank(v: Verdict, column: Column) -> tuple[bool, str]:
    date = v.blind_date if column == "blind" else (v.final_date or v.blind_date)
    return (v.task == "gold", date)


def known_outcomes(
    cells: Iterable[Cell], verdicts: Iterable[Verdict], column: Column = "final",
    ref_kinds: Mapping[str, str] | None = None,
) -> dict[str, OutcomeClass]:
    """Human-verified outcome per unit: the "Delta from human-verified status only" option.

    A unit is verified by a usable verdict in ``column``, or by a grade A cell from gold (gold
    is blind by construction, so it counts for both columns) or, for the final column, by any
    grade A cell. Callers pass only verdicts whose fingerprint still matches.
    """
    kinds = ref_kinds or {}
    out: dict[str, OutcomeClass] = {}
    cells = list(cells)
    units = {cell.unit_id for cell in cells if cell.status not in (Status.LACUNA, Status.NA)}
    for cell in cells:
        if cell.grade is Grade.A and (column == "final" or cell.source.startswith("gold:")):
            outcome = cell_outcome(cell, kinds.get(cell.unit_id, ""))
            if outcome is not None:
                out[cell.unit_id] = outcome
    for unit, v in _latest(verdicts, column).items():
        outcome = verdict_outcome(v, column, kinds.get(unit, ""))
        if outcome is not None and unit in units:
            out[unit] = outcome
    return out


def sample_counts(
    verdicts: Iterable[Verdict], strata: Mapping[str, str], column: Column = "final",
    ref_kinds: Mapping[str, str] | None = None,
) -> dict[str, dict[OutcomeClass, int]]:
    """x_h: outcome counts per stratum among the phase-2 *sample* verdicts.

    Gold and resolve verdicts fix their units but are not a probability sample of any stratum,
    so they never calibrate one. The stratum recorded on the verdict (fixed at sampling time)
    wins over the current ``strata`` mapping.
    """
    kinds = ref_kinds or {}
    sample = [v for v in verdicts if v.task in SAMPLE_TASKS]
    counts: dict[str, dict[OutcomeClass, int]] = {}
    for unit, v in _latest(sample, column).items():
        outcome = verdict_outcome(v, column, kinds.get(unit, ""))
        stratum = v.stratum or strata.get(unit)
        if outcome is None or stratum is None:
            continue
        row = counts.setdefault(stratum, {o: 0 for o in OUTCOMES})
        row[outcome] += 1
    return counts


def dirichlet(alpha: Sequence[float], rng: random.Random) -> list[float]:
    """One Dirichlet draw via normalised Gamma variates."""
    g = [rng.gammavariate(a, 1.0) for a in alpha]
    total = sum(g)
    return [x / total for x in g]


def draws(
    cells: Iterable[Cell], verdicts: Sequence[Verdict], strata: Mapping[str, str],
    n_draws: int, seed: int, column: Column = "final", ref_kinds: Mapping[str, str] | None = None,
) -> list[dict[str, OutcomeClass]]:
    """M completed outcome vectors (unit id -> outcome class) under the two-phase posterior.

    ``cells`` is the built matrix (verdicts applied); ``strata`` maps every countable unit to
    its sampling stratum, computed with ``stratum_of`` on the *machine* cells. Unverified units
    in ``unresolved`` or ``excluded`` strata, and LACUNA/NA cells, are absent from every draw.
    """
    cells = list(cells)
    known = known_outcomes(cells, verdicts, column, ref_kinds)
    x = sample_counts(verdicts, strata, column, ref_kinds)
    pools: dict[str, list[str]] = {}
    for cell in cells:
        unit = cell.unit_id
        if unit in known or cell.status not in COUNTED_STATUSES:
            continue                   # verified, or UNALIGNED / LACUNA / NA: never imputed
        stratum = strata.get(unit)
        if stratum is None:
            raise ValueError(f"{unit}: unverified unit without a sampling stratum")
        if stratum not in (UNRESOLVED, EXCLUDED):
            pools.setdefault(stratum, []).append(unit)
    rng = random.Random(seed)
    out: list[dict[str, OutcomeClass]] = []
    for _ in range(n_draws):
        vector = dict(known)
        for stratum in sorted(pools):
            row = x.get(stratum, {})
            p = dirichlet([row.get(o, 0) + 0.5 for o in OUTCOMES], rng)
            cum = list(itertools.accumulate(p))
            for unit in pools[stratum]:
                vector[unit] = rng.choices(OUTCOMES, cum_weights=cum)[0]
        out.append(vector)
    return out


def indicator(outcome: OutcomeClass, which: str) -> bool:
    """D for ``which``: "any" (D_any), "cov" (D_cov), or one outcome class (e.g. "absent")."""
    if which in ("any", "cov"):
        return deviates(outcome, which)  # type: ignore[arg-type]
    return outcome is OutcomeClass(which)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, ``q`` in [0, 100]."""
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of an empty sequence")
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarise(values: Sequence[float], name: str, n: int, scope: str, sources: tuple[str, ...]) -> Estimate:
    """Mean of the draws with the 2.5-97.5 percentile interval."""
    return Estimate(
        name=name, point=sum(values) / len(values), lo=percentile(values, 2.5), hi=percentile(values, 97.5),
        n=n, scope=scope, sources=sources,
    )


def prevalence(
    draw_list: Sequence[Mapping[str, OutcomeClass]], outcome: str = "any",
    weights: Mapping[str, float] | None = None, name: str = "E1", scope: str = "",
) -> Estimate:
    """Finite-population share of units with the outcome (E1), or length-weighted (E2).

    With ``weights`` (e.g. syllables per unit) this is sum(l_u D_u) / sum(l_u). No chapter
    bootstrap: the text is a census (synthesis 6.1).
    """
    values: list[float] = []
    for vector in draw_list:
        num = den = 0.0
        for unit, cls in vector.items():
            w = 1.0 if weights is None else float(weights[unit])
            den += w
            num += w * indicator(cls, outcome)
        if den > 0:
            values.append(num / den)
    if not values:
        return Estimate.missing(name, "no countable units", scope)
    sources = SOURCES + (("segmentation: syllable-weighted",) if weights is not None else ())
    return summarise(values, name, len(draw_list[0]), scope, sources)


def unit_means(draw_list: Sequence[Mapping[str, OutcomeClass]], outcome: str = "any") -> dict[str, float]:
    """Posterior mean of D per unit (the input of the permutation test and matching)."""
    totals: dict[str, float] = {}
    for vector in draw_list:
        for unit, cls in vector.items():
            totals[unit] = totals.get(unit, 0.0) + indicator(cls, outcome)
    return {unit: t / len(draw_list) for unit, t in totals.items()}


def manski(
    cells: Iterable[Cell], outcome: str = "any", ref_kinds: Mapping[str, str] | None = None,
) -> tuple[float, float]:
    """Bounds on the share when every still-UNALIGNED unit deviates (hi) or none does (lo).

    ``cells`` is the built matrix (human resolutions applied). LACUNA and NA stay excluded.
    Report the bounds as flagged when ``hi - lo`` exceeds ``gates.g3.max_manski_width``.
    """
    kinds = ref_kinds or {}
    n = d = u = 0
    for cell in cells:
        if cell.status is Status.UNALIGNED:
            u += 1
            continue
        cls = cell_outcome(cell, kinds.get(cell.unit_id, ""))
        if cls is None:
            continue
        n += 1
        d += indicator(cls, outcome)
    total = n + u
    if total == 0:
        raise ValueError("no countable or unresolved units")
    return d / total, (d + u) / total


@dataclass(frozen=True)
class Revision:
    """Blind -> final changes among verdicts with both columns, in one stratum."""

    stratum: str
    n: int
    relation_changed: int
    outcome_changed: int

    @property
    def rate(self) -> float | None:
        """Share whose outcome class changed at reveal (the automation-bias exposure)."""
        return self.outcome_changed / self.n if self.n else None


def revision_rate(verdicts: Iterable[Verdict], ref_kinds: Mapping[str, str] | None = None) -> dict[str, Revision]:
    """Per verdict stratum (critique A1): how often the reveal changed the blind decision."""
    kinds = ref_kinds or {}
    acc: dict[str, list[int]] = {}
    for v in verdicts:
        if not (v.blind_relation and v.final_relation):
            continue
        row = acc.setdefault(v.stratum or "unstratified", [0, 0, 0])
        row[0] += 1
        row[1] += v.blind_relation != v.final_relation
        kind = kinds.get(v.unit_id, "")
        row[2] += verdict_outcome(v, "blind", kind) != verdict_outcome(v, "final", kind)
    return {s: Revision(s, *row) for s, row in sorted(acc.items())}
