"""E4 for the stats stage: Delta with its pre-registered diagnostics, and the MDE on real labels.

Beside the two-phase-corrected Delta (``stats.contrast.delta``) the stage reports what
synthesis 6.3 and critique B13 pre-register:

    permutation_p         topic labels permuted within strata (the placebo topic)
    equivalent_within_margin  TOST with margin = the pre-registered MDE (``stats.mde``)
    overlap               strata with both exposures and the units dropped (B13)
    naive / attenuation   Delta on the machine labels, and corrected minus naive
    matched_rd            greedy 1:1 caliper matching within chapter x reference kind
    misclassification     verified machine error by chapter and by length tertile, per
                          imputation stratum (checks the within-stratum independence assumption)
    negative_control      Delta of frame vs neutral units (expected about 0)

``mde_on_labels`` re-runs ``stats.power.simulate_delta`` on the real topic labels (synthesis
6.5): the smallest effect on a grid detected with 80% power. Gate G4 uses the larger of it
and the preregistered MDE once ``stats/power.json`` exists.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from ..core.textnorm import length
from ..core.types import COUNTED_STATUSES, Cell, Estimate, OutcomeClass, Status, Verdict
from ..matrix.status import cell_outcome
from ..stats import contrast, power, twophase
from .context import Texts
from .store import estimate_to_dict

TARGET, CONTROL, NEGATIVE_CONTROL = "sensitive", "neutral", "frame"
MDE_GRID = tuple(round(0.02 * k, 2) for k in range(1, 16))     # 0.02 .. 0.30
TARGET_POWER = 0.80
N_SIM = 200


def exposure(cells: Sequence[Cell], groups: Mapping[str, str], target: str = TARGET) -> dict[str, bool]:
    """Unit -> exposed (``target`` group) or not (neutral); every other unit is left out, as are
    uniform-across-list and LACUNA cells."""
    excluded = {c.unit_id for c in cells if "uniform_across_list" in c.flags or c.status is Status.LACUNA}
    return {u: g == target for u, g in groups.items() if g in (target, CONTROL) and u not in excluded}


def tost_margin(prereg: Mapping[str, Any]) -> tuple[float, str | None]:
    """The TOST margin is the pre-registered MDE (synthesis 6.3); a differing legacy
    ``stats.tost_margin`` is reported and ignored."""
    params = prereg.get("stats") or {}
    mde, legacy = params.get("mde"), params.get("tost_margin")
    margin = float(mde if mde is not None else legacy if legacy is not None else 0.1)
    if mde is not None and legacy is not None and float(legacy) != float(mde):
        return margin, (f"preregistration.yaml: stats.tost_margin {legacy} differs from stats.mde {mde}; "
                        "the TOST margin is the pre-registered MDE")
    return margin, None


def machine_indicator(machine: Sequence[Cell], ref_kinds: Mapping[str, str], outcome: str) -> dict[str, float]:
    """Naive D per unit from the machine cells (countable cells only)."""
    out = {}
    for c in machine:
        cls = cell_outcome(c, ref_kinds.get(c.unit_id, "")) if c.status in COUNTED_STATUSES else None
        if isinstance(cls, OutcomeClass):
            out[c.unit_id] = float(twophase.indicator(cls, outcome))
    return out


def contrast_with_diagnostics(texts: Texts, cells: Sequence[Cell], machine: Sequence[Cell], groups: Mapping[str, str],
                              verdicts: Sequence[Verdict], draws: Sequence[Mapping[str, OutcomeClass]], outcome: str,
                              seed: int, n_perm: int, margin: float, scope: str) -> tuple[Estimate, dict[str, Any]]:
    """Delta and every diagnostic listed in the module docstring."""
    exposed = exposure(cells, groups)
    all_lengths = {s.id: float(length(s.text, s.lang)) for s in texts.units}
    tertiles = contrast.tertile_strata(all_lengths, texts.chapter_of)
    strata = contrast.tertile_strata({u: all_lengths[u] for u in exposed if u in all_lengths}, texts.chapter_of)
    estimate = contrast.delta(draws, exposed, strata, texts.chapter_of, seed, outcome, scope=scope)
    means = twophase.unit_means(draws, outcome)
    naive = contrast.stratified_rd(machine_indicator(machine, texts.ref_kinds, outcome), exposed, strata)
    group = {u: f"{texts.chapter_of[u]}|{texts.ref_kinds.get(u, '')}" for u in exposed if u in texts.chapter_of}
    matched, pairs = contrast.matched_rd(means, exposed, group, all_lengths, seed=seed)
    table = contrast.misclassification_table(verdicts, texts.chapter_of,
                                             {u: t.rsplit("|", 1)[-1] for u, t in tertiles.items()},
                                             "final", outcome, texts.ref_kinds)
    frame = exposure(cells, groups, NEGATIVE_CONTROL)
    frame_strata = contrast.tertile_strata({u: all_lengths[u] for u in frame if u in all_lengths}, texts.chapter_of)
    negative = contrast.delta(draws, frame, frame_strata, texts.chapter_of, seed, outcome,
                              name="E4_frame_vs_neutral", scope=scope)
    corrected = estimate.point
    return estimate, {
        "permutation_p": contrast.permutation_p(means, exposed, strata, n_perm, seed),
        "equivalent_within_margin": contrast.tost(estimate, margin), "margin": margin,
        "overlap": asdict(contrast.overlap_diagnostics(exposed, strata)),
        "naive": naive, "attenuation": None if corrected is None or naive is None else corrected - naive,
        "matched_rd": {"difference": matched, "pairs": pairs},
        "misclassification": [{**asdict(r), "rate": r.rate} for r in table],
        "negative_control": estimate_to_dict(negative),
    }


def mde_on_labels(texts: Texts, machine: Sequence[Cell], groups: Mapping[str, str], outcome: str,
                  seed: int, n_sim: int = N_SIM) -> dict[str, Any]:
    """The smallest Delta on ``MDE_GRID`` detected with ``TARGET_POWER`` by the stratified test,
    for the real chapters, exposure shares and machine base rate (None: none on the grid)."""
    exposed = exposure(machine, groups)
    sizes: dict[str, list[int]] = {}
    for u, e in exposed.items():
        chapter = texts.chapter_of.get(u)
        if chapter is not None:
            row = sizes.setdefault(chapter, [0, 0])
            row[0] += 1
            row[1] += e
    chapters = sorted(sizes)
    y = machine_indicator(machine, texts.ref_kinds, outcome)
    control = [y[u] for u, e in exposed.items() if not e and u in y]
    p0 = min(0.9, max(0.02, sum(control) / len(control))) if control else 0.2
    found: float | None = None
    curve: dict[str, float] = {}
    for d in MDE_GRID:
        achieved = power.simulate_delta([sizes[c][0] for c in chapters], [sizes[c][1] / sizes[c][0] for c in chapters],
                                        p0, d, n_sim=n_sim, seed=seed)
        curve[f"{d:.2f}"] = achieved
        if achieved >= TARGET_POWER:
            found = d
            break
    return {"mde_on_labels": found, "target_power": TARGET_POWER, "p0": p0, "chapters": len(chapters),
            "units": len(exposed), "exposed": sum(exposed.values()), "power_curve": curve, "n_sim": n_sim}
