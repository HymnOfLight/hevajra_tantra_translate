"""E3 for ``stats`` and gate G4: per-manuscript Sanskrit readings -> the decomposition.

Inputs (synthesis 6.4, critique A2/A3):

    readings     ``data/reference/readings/<chapter>.tsv`` (``ingest.sanskrit.load_readings``),
                 keyed by reference unit id (Snellgrove ids; Derge segment ids while the Derge
                 is the provisional reference)
    manuscripts  only ids listed in ``data/registry/sa_manuscripts.yaml`` are manuscript
                 columns; a reading under an edition's witness id is never counted (editions
                 are not independent witnesses) and is listed in ``editions_ignored``; any other
                 id is an error
    co-witness   the other aligned witness: the Derge for the Chinese and the Chinese for the
                 Derge under the Sanskrit reference. Under the Derge reference there is none (the
                 reference would be its own co-witness). A revised co-witness (the Derge) gives
                 the label ``shared_revised``
    m_min        ``preregistration.yaml: gates.g4.min_manuscripts_per_unit``

The deviation indicator is the primary outcome of the final matrix (human decisions applied).
A2: the units entering E3 must be verified or bounded, so beside the decomposition as
measured the stage reports the bound in which every shared unit not human-verified in both
columns (grade A) counts as not shared (``decompose.bound_unverified_shared``).

When G4 fails (``E3Inputs.reason``) E3 is NOT_ESTIMABLE with that reason; when manuscript
readings exist the decomposition is still written to ``stats/details.json`` as descriptive
detail (with no co-witness every non-Vorlage unit is ``insufficient``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.types import Cell, Estimate, Grade
from ..evaluation.gate import GateSpecs
from ..ingest.sanskrit import ReadingTable, load_readings
from ..matrix.status import cell_deviates
from ..registry import load_manuscripts, load_witnesses
from ..review.verdicts import is_orphan
from ..stats import decompose
from .context import RunContext, StageError, Texts
from .store import read_cells

READINGS_DIR = "readings"
MANUSCRIPTS_FILE = Path("registry") / "sa_manuscripts.yaml"
WITNESSES_FILE = Path("registry") / "witnesses.yaml"


@dataclass(frozen=True)
class E3Inputs:
    readings: ReadingTable | None
    manuscripts: tuple[str, ...]          # manuscript columns: registry manuscripts with readings
    editions_ignored: tuple[str, ...]     # reading columns naming an edition (never counted)
    cowitness: str | None
    cowitness_cells: tuple[Cell, ...]
    cowitness_revised: bool
    reason: str | None                    # the G4 reason, None when E3 is estimable


def readings_dir(ctx: RunContext) -> Path:
    return ctx.settings.path("reference") / READINGS_DIR


def e3_inputs(ctx: RunContext, texts: Texts) -> E3Inputs:
    """Readings, manuscript columns, co-witness and the G4 verdict for this run's witness."""
    directory = readings_dir(ctx)
    table = None
    if directory.is_dir() and any(directory.glob("*.tsv")):
        try:
            table = load_readings(directory)
        except ValueError as exc:
            raise StageError(str(exc)) from exc
    registry = load_manuscripts(ctx.data_dir / MANUSCRIPTS_FILE)
    witnesses = load_witnesses(ctx.data_dir / WITNESSES_FILE)
    named = sorted(table.manuscripts()) if table else []
    unknown = [m for m in named if m not in registry and m not in witnesses]
    if unknown:
        raise StageError(f"{directory}: manuscript id(s) {unknown} are not in {MANUSCRIPTS_FILE} "
                         "(add the manuscript there; editions use their witness id and are never counted)")
    manuscripts = tuple(m for m in named if m in registry)
    editions = tuple(m for m in named if m not in registry)
    others = [w for w in ctx.witnesses if w != texts.witness_id]
    cowitness = others[0] if ctx.sanskrit_mode and others else (None if ctx.sanskrit_mode else texts.reference_id)
    reason = decompose.not_estimable_reason(manuscripts, texts.reference_id, cowitness)
    if editions and not manuscripts:
        reason = f"{reason} (the readings of {', '.join(editions)} are editions, never manuscripts)"
    cells: tuple[Cell, ...] = ()
    if cowitness is not None and cowitness != texts.reference_id:
        path = ctx.for_witness(cowitness).path("matrix", "cells.jsonl")
        if path.is_file():
            cells = tuple(c for c in read_cells(path) if not is_orphan(c.unit_id))
        elif reason is None:
            reason = f"G4: the co-witness {cowitness} has no matrix yet (run `hevajra-matrix build`)"
    revised = cowitness in witnesses and witnesses[cowitness].independence == "revised"
    return E3Inputs(table, manuscripts, editions, cowitness, cells, revised, reason)


def unit_evidence(texts: Texts, cells: Sequence[Cell], inputs: E3Inputs, outcome: str) -> list[decompose.UnitEvidence]:
    """One ``UnitEvidence`` per countable unit (uncountable cells are left out)."""
    hosts: dict[str, set[str]] = {}
    for s in texts.witness:
        if s.kind == "note" and s.extra.get("host"):
            hosts.setdefault(s.extra["host"], set()).add(s.extra.get("note_class", ""))
    cow = {c.unit_id: c for c in inputs.cowitness_cells}
    columns = set(inputs.manuscripts)
    out = []
    for c in cells:
        kind = texts.ref_kinds.get(c.unit_id, "")
        d = cell_deviates(c, outcome, kind)          # type: ignore[arg-type]
        if d is None:
            continue
        other = cow.get(c.unit_id)
        readings = inputs.readings.for_unit(c.unit_id) if inputs.readings else ()
        out.append(decompose.UnitEvidence(
            unit_id=c.unit_id, deviates=d, note_classes=frozenset(x for w in c.wit_ids for x in hosts.get(w, ())),
            manuscripts={r.ms: r.status for r in readings if r.ms in columns},
            cowitness_deviates=None if other is None else cell_deviates(other, outcome, kind),  # type: ignore[arg-type]
            chapter=texts.chapter_of.get(c.unit_id, "")))
    return out


def verified_units(cells: Sequence[Cell], inputs: E3Inputs) -> set[str]:
    """Units decided by a human (grade A) in the witness column and, when there is one, the
    co-witness column too."""
    cow = {c.unit_id: c.grade for c in inputs.cowitness_cells}
    return {c.unit_id for c in cells if c.grade is Grade.A
            and (not inputs.cowitness_cells or cow.get(c.unit_id) is Grade.A)}


def e3_stats(ctx: RunContext, texts: Texts, cells: Sequence[Cell], outcome: str,
             scope: str) -> tuple[dict[str, Estimate], dict[str, Any]]:
    """E3 estimates (``E3``, ``E3.phi_V``, ``E3.phi_S``) and ``stats/details.json: e3``."""
    inputs = e3_inputs(ctx, texts)
    m_min = GateSpecs.from_prereg(ctx.settings.prereg).g4.min_manuscripts_per_unit
    known = set(texts.ref_kinds)
    read_units = sorted(inputs.readings.by_unit) if inputs.readings else []
    details: dict[str, Any] = {
        "manuscripts": list(inputs.manuscripts), "editions_ignored": list(inputs.editions_ignored),
        "cowitness": inputs.cowitness, "cowitness_revised": inputs.cowitness_revised, "m_min": m_min,
        "readings_units": len(read_units), "unknown_units": [u for u in read_units if u not in known],
        "estimable": inputs.reason is None, "reason": inputs.reason}
    if details["unknown_units"]:
        print(f"stats E3: WARNING {len(details['unknown_units'])} reading unit id(s) are not reference units of "
              f"{texts.reference_id}, e.g. {details['unknown_units'][:3]}")
    if not inputs.manuscripts:
        return {"E3": Estimate.missing("E3", str(inputs.reason), scope)}, details
    units = unit_evidence(texts, cells, inputs, outcome)
    verified = verified_units(cells, inputs)
    bounded = decompose.bound_unverified_shared(units, verified)
    measured = decompose.decompose(units, m_min, inputs.cowitness_revised)
    bound = decompose.decompose(bounded, m_min, inputs.cowitness_revised)
    vorlage = decompose.vorlage_excess(units, m_min)
    shared, shared_bound = decompose.shared_excess(units, m_min), decompose.shared_excess(bounded, m_min)
    details.update({
        "units": len(units), "n_deviating": measured.n_deviating, "counts": dict(measured.counts),
        "by_unit": dict(measured.by_unit),
        "translator_flagged": list(measured.translator_flagged),
        "verified_deviating": sum(1 for u in units if u.deviates and u.unit_id in verified),
        "vorlage": excess_dict(vorlage), "shared": excess_dict(shared),
        "bound": {"counts": dict(bound.counts), "by_unit": dict(bound.by_unit), "shared": excess_dict(shared_bound),
                  "unverified_shared": sum(1 for a, b in zip(units, bounded)
                                           if a.cowitness_deviates != b.cowitness_deviates)}})
    if inputs.reason:
        return {"E3": Estimate.missing("E3", inputs.reason, scope)}, details
    result = decompose.E3Result(measured, vorlage, shared)
    phi_v, phi_s = result.estimates(scope)
    estimate = Estimate("E3", float(measured.n_deviating), None, None, len(units), scope,
                        ("decomposition counts, O/E/O-E and phi: stats/details.json e3", "A2 bound beside it"))
    return {"E3": estimate, "E3.phi_V": phi_v, "E3.phi_S": phi_s}, details


def excess_dict(e: decompose.Excess) -> dict[str, Any]:
    return {"n": e.n, "observed": e.observed, "expected": e.expected, "excess": e.excess,
            "case_rate": e.case_rate, "base_rate": e.base_rate, "phi": e.phi}


def e3_reason(ctx: RunContext, texts: Texts) -> str | None:
    """The G4 reason for E3 (``evaluation.gate.Calibration.e3_not_estimable``), or None."""
    return e3_inputs(ctx, texts).reason
