"""D_any_conv (``E1_any_conv``): the convention imputed jointly with the outcome class.

D_any_conv (collate.v2 sensitivity variant, E1 on the final column only) counts a unit when
its outcome class deviates OR it carries a convention (standing rendering or transcribed
name). In the two-phase design (``twophase``) the convention is imputed together with the
class: per frozen stratum h the phase-2 sample verdicts (``twophase.sample_verdicts``) are
counted over five categories

    NONDEV without a convention, NONDEV with a convention, DEV_PRESENT, PARTIAL, ABSENT

(under any_conv only the first is non-deviating). A verdict's convention follows
``review.verdicts.decision_convention``; "" (not annotated, e.g. a ``?`` left on the sheet)
counts as none, here as in the built matrix, and ``summary`` reports how many sample verdicts
and how many units decided by gold or resolve leave it so. The posterior predictive is
Dirichlet(x_h + prior) with the prior of ``twophase.draws`` unchanged: the Jeffreys 1/2 of
NONDEV stays on NONDEV without a convention and the convention category gets no pseudo-count.
By the aggregation property of the Dirichlet this equals

    the class draw of ``twophase.draws`` (NONDEV with weight x_h,NONDEV + 1/2), then
    q_h ~ Beta(c_h, x_h,NONDEV - c_h + 1/2) per draw (c_h: NONDEV verdicts with a convention),
    and each unit drawn NONDEV carries a convention with probability q_h,

so ``convention_draws`` completes the E1_any draws instead of drawing classes anew: the class
marginal is exactly the E1_any posterior (prior P(D) unchanged), and a stratum whose NONDEV
sample verdicts carry no convention imputes none (q_h = 0).

Verified units (``twophase.known_outcomes``) keep their human class and human convention:
the deciding verdict's, or the grade A cell's convention flag (built from the human decision).
Machine conventions never enter, just as machine classes do not: an unverified unit's
convention comes from its stratum's sample. E1_any_conv is therefore defined for a collate.v1
run too (conventions enter through the human verdicts only), and on a collate.v2 run the
machine convention errors are calibrated by the sample like the class errors. It equals
E1_any draw by draw whenever no non-deviating verified unit and no non-deviating phase-2
sample verdict records a convention, whatever the machine cells say (in particular when no
human verdict records one); otherwise it is larger. The Manski bounds of any_conv read the
built cells instead (``twophase.manski``) and are computed only for a collate.v2 run
(``pipeline.results``).
"""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

from ..core.types import Cell, Convention, Grade, OutcomeClass, Verdict
from ..matrix.status import has_convention
from ..review.verdicts import decision_convention
from . import twophase


def verdict_convention(verdict: Verdict) -> bool:
    """Whether the verdict's final decision records a convention ("" and none: no)."""
    return Convention(decision_convention(verdict) or Convention.NONE.value) is not Convention.NONE


def known_conventions(
    cells: Iterable[Cell], verdicts: Sequence[Verdict], ref_kinds: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    """Human convention of every verified unit of the final column (the units of
    ``twophase.known_outcomes``): the deciding verdict's, else the grade A cell's flag."""
    kinds = ref_kinds or {}
    cells = list(cells)
    known = twophase.known_outcomes(cells, verdicts, "final", kinds)
    out = {c.unit_id: has_convention(c.flags) for c in cells if c.unit_id in known and c.grade is Grade.A}
    for unit, v in twophase.latest_verdicts(verdicts, "final").items():
        if unit in known and twophase.verdict_outcome(v, "final", kinds.get(unit, "")) is not None:
            out[unit] = verdict_convention(v)
    return {unit: out.get(unit, False) for unit in known}


def convention_counts(
    verdicts: Sequence[Verdict], strata: Mapping[str, str], ref_kinds: Mapping[str, str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Per stratum, (NONDEV sample verdicts without a convention, NONDEV ones with one)."""
    out: dict[str, list[int]] = {}
    for stratum, v, outcome in twophase.sample_verdicts(verdicts, strata, "final", ref_kinds):
        if outcome is OutcomeClass.NONDEV:
            out.setdefault(stratum, [0, 0])[verdict_convention(v)] += 1
    return {s: (row[0], row[1]) for s, row in out.items()}


def _beta(a: float, b: float, rng: random.Random) -> float:
    """Beta(a, b) with a = 0 read as the point mass at 0 (no convention in the sample)."""
    return 0.0 if a == 0 else rng.betavariate(a, b)


def convention_draws(
    cells: Iterable[Cell], verdicts: Sequence[Verdict], strata: Mapping[str, str],
    draw_list: Sequence[Mapping[str, OutcomeClass]], seed: int, ref_kinds: Mapping[str, str] | None = None,
) -> list[dict[str, bool]]:
    """One unit -> convention map per draw of ``draw_list`` (the final-column ``twophase.draws``
    with the Jeffreys prior, computed from the same cells, verdicts and strata).

    Verified units carry their human convention (``known_conventions``); an imputed unit drawn
    NONDEV carries one with its stratum's q_h ~ Beta(c_h, x_h,NONDEV - c_h + 1/2), drawn once
    per draw and stratum; an imputed unit drawn deviating counts under D_any_conv anyway and
    is given none. ``seed`` drives a stream of its own, so the class draws stay those of E1_any.
    """
    cells = list(cells)
    known = known_conventions(cells, verdicts, ref_kinds)
    counts = convention_counts(verdicts, strata, ref_kinds)
    rng = random.Random(f"any_conv:{seed}")
    out: list[dict[str, bool]] = []
    for vector in draw_list:
        share: dict[str, float] = {}
        conv: dict[str, bool] = {}
        for unit in sorted(vector):
            if unit in known:
                conv[unit] = known[unit]
                continue
            if vector[unit] is not OutcomeClass.NONDEV:
                conv[unit] = False
                continue
            stratum = strata[unit]
            if stratum not in share:
                without, with_ = counts.get(stratum, (0, 0))
                share[stratum] = _beta(with_, without + twophase.PRIOR_JEFFREYS[0], rng)
            conv[unit] = rng.random() < share[stratum]
        out.append(conv)
    return out


def summary(
    cells: Iterable[Cell], verdicts: Sequence[Verdict], strata: Mapping[str, str],
    ref_kinds: Mapping[str, str] | None = None, gold: Sequence[Verdict] = (),
) -> dict[str, int]:
    """What the any_conv imputation rests on (``stats/details.json: any_conv``).

    ``verdicts``                phase-2 sample verdicts that calibrate a stratum
    ``unannotated``             of them, with no annotated convention ("" counted as none)
    ``with_convention``         of them, recording a convention
    ``nondev_with_convention``  of them, non-deviating and recording a convention
    ``verified_with_convention`` verified units carrying a human convention
    ``verified_unannotated``    verified units whose deciding verdict is gold or resolve (not
                                a sample verdict) with no annotated convention (counted as none)
    ``verified_nondev_unannotated``  of them, non-deviating (the only verified units whose
                                annotation could move E1_any_conv)
    E1_any_conv equals E1_any when ``nondev_with_convention`` and ``verified_with_convention``
    are 0. ``gold`` are the gold verdicts behind the grade A gold cells; they enter the last two
    counts only, and the deciding verdict is the one ``known_conventions`` reads (a usable
    verdict of ``verdicts`` over the gold cell).
    """
    cells = list(cells)
    kinds = ref_kinds or {}
    used = twophase.sample_verdicts(verdicts, strata, "final", kinds)
    known = twophase.known_outcomes(cells, verdicts, "final", kinds)
    deciding = twophase.latest_verdicts(gold, "final")
    for unit, v in twophase.latest_verdicts(verdicts, "final").items():
        if twophase.verdict_outcome(v, "final", kinds.get(unit, "")) is not None:
            deciding[unit] = v
    unannotated = [known[u] for u, v in deciding.items()
                   if u in known and v.task not in twophase.SAMPLE_TASKS and not decision_convention(v)]
    return {"verdicts": len(used),
            "unannotated": sum(1 for _, v, _ in used if not decision_convention(v)),
            "with_convention": sum(1 for _, v, _ in used if verdict_convention(v)),
            "nondev_with_convention": sum(1 for _, v, o in used if o is OutcomeClass.NONDEV and verdict_convention(v)),
            "verified_with_convention": sum(known_conventions(cells, verdicts, kinds).values()),
            "verified_unannotated": len(unannotated),
            "verified_nondev_unannotated": sum(1 for o in unannotated if o is OutcomeClass.NONDEV)}
