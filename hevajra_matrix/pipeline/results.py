"""Claude-free stages that turn cells and verdicts into numbers and the report: stats, report."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..collate.consensus import SOURCE_CONSENSUS
from ..core.io import read_jsonl
from ..core.textnorm import length
from ..core.types import Cell, Estimate, Grade, Relation, Status
from ..evaluation.gate import GateReport, GateSpecs
from ..report import markdown, svg
from ..review import sampling
from ..review.verdicts import is_orphan
from ..stats import decompose, distance, twophase
from . import e3 as e3_stage
from . import e4 as e4_stage
from . import human_data as ann
from .context import RunContext, Texts, load_texts, record_stage, scope_lines
from .instrument import read_consensus
from .measure import control_alignments, read_built_cells, read_gate
from .store import estimate_from_dict, estimate_to_dict, read_cells, sentinel_result_from_dict

OUTCOMES = ("any", "cov", "absent", "partial")
NO_PHASE2 = "G3: no phase-2 verification or audit verdict yet"
PRIOR_SUFFIX = "_prior"    # E1_<primary>_prior: the primary E1 with a prior symmetric in D
TRANSLATOR_NOTE_CLASSES = (decompose.VORLAGE_NOTE, decompose.TRANSLATOR_NOTE)


# --------------------------------------------------------------------------- stats
def stats(ctx: RunContext) -> dict[str, Estimate]:
    """E1, E2 (two-phase draws, final and blind columns), Manski bounds, E3, E4, E5, and the
    witness distances (UPGMA) once three texts share the reference.

    E3 comes from ``pipeline.e3`` (manuscript readings, co-witness, A2 bound).

    ``E1_<primary>_prior`` repeats E1 of the primary outcome with ``twophase.PRIOR_SYMMETRIC_D``
    (the spec's Jeffreys prior leans 3/4 toward deviation in sparse strata). A column whose
    draws would impute a stratum without any phase-2 sample is NOT_ESTIMABLE, not a prior.

    Numbers are computed only from phase-2 verdicts: without any, E1/E2 are NOT_ESTIMABLE
    rather than prior-only draws. Whether a number may be printed is the report's job.
    """
    texts = load_texts(ctx)
    cells = [c for c in read_built_cells(ctx) if not is_orphan(c.unit_id)]
    orphans = [c for c in read_built_cells(ctx) if is_orphan(c.unit_id)]
    notes = sampling.note_rows(orphans, texts.wit_kinds)     # classified at ingest: outside E5
    orphans = [c for c in orphans if c.unit_id not in notes]
    machine = [c for c in read_built_cells(ctx, machine=True) if not is_orphan(c.unit_id)]
    topics = ann.load_topics(ctx, texts)
    verdicts = ann.current_verdicts(ann.review_verdicts(ctx, texts), texts)
    strata = ann.sampling_strata(ctx, texts.witness_id, sampling.machine_strata(machine, topics.groups(),
                                                                                texts.ref_kinds))
    params = ctx.settings.prereg.get("stats") or {}
    n_draws, seed = int(params.get("n_draws", 500)), int(params.get("seed", 0))
    scope = "; ".join(scope_lines(ctx))
    primary = str(ctx.settings.prereg.get("primary_outcome", "any"))
    est: dict[str, Estimate] = {}
    extra: dict[str, Any] = {}
    draws: dict[str, list] = {}
    if any(v.task in twophase.SAMPLE_TASKS for v in verdicts):
        weights = {s.id: float(length(s.text, s.lang)) for s in texts.units}
        for column in ("final", "blind"):
            suffix = "" if column == "final" else "_blind"
            names = [f"E1_{o}{suffix}" for o in (OUTCOMES if column == "final" else ("any", "cov"))]
            names.append(f"E2_any{suffix}")
            try:
                draws[column] = twophase.draws(cells, verdicts, strata, n_draws, seed, column, texts.ref_kinds)
            except twophase.UncalibratedStrata as err:
                extra.setdefault("uncalibrated", {})[column] = err.strata
                est.update({name: Estimate.missing(name, str(err), scope) for name in names})
                continue
            for outcome in OUTCOMES if column == "final" else ("any", "cov"):
                est[f"E1_{outcome}{suffix}"] = twophase.prevalence(draws[column], outcome, name=f"E1_{outcome}{suffix}",
                                                                   scope=scope)
            est[f"E2_any{suffix}"] = twophase.prevalence(draws[column], "any", weights, name=f"E2_any{suffix}", scope=scope)
        if "final" in draws:      # prior sensitivity: Dirichlet symmetric in D instead of Jeffreys
            sym = twophase.draws(cells, verdicts, strata, n_draws, seed, "final", texts.ref_kinds, prior="symmetric_d")
            name = f"E1_{primary}{PRIOR_SUFFIX}"
            est[name] = twophase.prevalence(sym, primary, name=name, scope=scope)
        extra["revision"] = {s: vars(r) for s, r in twophase.revision_rate(verdicts, texts.ref_kinds).items()}
    else:
        for name in ("E1_any", "E1_cov", "E1_absent", "E1_partial", "E2_any"):
            est[name] = Estimate.missing(name, NO_PHASE2, scope)
    if cells:
        extra["manski"] = {o: list(twophase.manski(cells, o, texts.ref_kinds)) for o in ("any", "cov")}
    e3, extra["e3"] = e3_stage.e3_stats(ctx, texts, cells, primary, scope)
    est.update(e3)
    tree = witness_tree(ctx, texts, cells)
    if tree:
        extra["distance"] = tree
    no_draws = est[f"E1_{primary}"].not_estimable if f"E1_{primary}" in est else None   # why "final" has no draws
    est["E4"], extra["e4"] = _e4(ctx, texts, cells, machine, topics, verdicts, draws.get("final"), primary, seed,
                                 scope, no_draws or NO_PHASE2)
    est["E5"] = _e5(orphans, texts, scope)
    extra["ingest_notes"] = {"witness_only_rows": len(notes)}   # descriptive: notes classified at ingest
    out = ctx.path("stats")
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimates.json").write_text(json.dumps({k: estimate_to_dict(v) for k, v in est.items()}, indent=1,
                                                   sort_keys=True) + "\n", encoding="utf-8")
    (out / "details.json").write_text(json.dumps(extra, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    record_stage(ctx, "stats")
    for name, e in est.items():
        print(f"stats {name}: {'NOT_ESTIMABLE: ' + e.not_estimable if e.not_estimable else 'computed'}")
    return est


def _e4(ctx: RunContext, texts: Texts, cells: list[Cell], machine: list[Cell], topics: ann.Topics,
        verdicts: list, draws: list | None, outcome: str, seed: int, scope: str,
        no_draws: str = NO_PHASE2) -> tuple[Estimate, dict[str, Any]]:
    """Delta (synthesis 6.3) with its diagnostics (``pipeline.e4``) when G4 allows it and draws
    exist; otherwise the reason (``no_draws`` says why there are no draws: no phase-2 verdict,
    or a stratum without a phase-2 sample). With complete topic labels the MDE is first
    simulated on them (``stats/power.json``); an MDE above the G4 maximum blocks E4 here as well."""
    prereg = ctx.settings.prereg
    margin, warning = e4_stage.tost_margin(prereg)
    if warning:
        print(f"stats E4: WARNING {warning}")
    power = None
    if topics.complete:
        power = e4_stage.mde_on_labels(texts, machine, topics.groups(), outcome, seed)
        _write_json(ctx.path("stats", "power.json"), power)
    gate = read_gate(ctx)
    reason = (gate.not_estimable.get("E4") if gate is not None else "G4: evaluate has not run")
    max_mde = GateSpecs.from_prereg(prereg).g4.max_mde
    if reason is None and power is not None and (power["mde_on_labels"] is None or power["mde_on_labels"] > max_mde):
        reason = f"G4: MDE on the real topic labels {power['mde_on_labels']} > {max_mde} (stats/power.json)"
    if reason is None and draws is None:
        reason = no_draws
    if reason:
        return Estimate.missing("E4", reason, scope), {"margin": margin, **({"power": power} if power else {})}
    n_perm = int((prereg.get("stats") or {}).get("n_permutations", 2000))
    estimate, extra = e4_stage.contrast_with_diagnostics(texts, cells, machine, topics.groups(), verdicts, draws,
                                                         outcome, seed, n_perm, margin, scope)
    return estimate, {**extra, **({"power": power} if power else {})}


def witness_tree(ctx: RunContext, texts: Texts, cells: list[Cell]) -> dict[str, Any] | None:
    """Witness distances and the UPGMA tree (descriptive) once three texts share the reference
    units: the Sanskrit reference itself (every unit retained, all dimensions 0), the Derge and
    the Chinese. None in Derge mode or before every witness has a matrix."""
    columns = {texts.witness_id: cells}
    for other in ctx.witnesses:
        path = ctx.for_witness(other).path("matrix", "cells.jsonl")
        if other not in columns:
            if not path.is_file():
                return None
            columns[other] = [c for c in read_cells(path) if not is_orphan(c.unit_id)]
    columns[texts.reference_id] = [Cell(u.id, texts.reference_id, Status.PRESENT, Grade.A, str(u.chapter),
                                        Relation.EQUIVALENT.value, d_len=0.0, d_ord=0.0, d_lit=0.0)
                                   for u in texts.units]
    if len(columns) < distance.MIN_WITNESSES:
        return None
    labels, rows = distance.distance_matrix([c for cs in columns.values() for c in cs], sorted(columns),
                                            texts.ref_kinds)
    return {"labels": labels, "matrix": rows, "newick": distance.average_linkage_newick(labels, rows)}


def _e5(orphans: list[Cell], texts: Texts, scope: str) -> Estimate:
    """Human-verified witness-only segments (count; ``n`` = their characters)."""
    verified = [c for c in orphans if c.grade.value == "A"]
    if not verified:
        return Estimate.missing("E5", "G3: no witness-only claim verified yet", scope)
    text = {s.id: s for s in texts.witness}
    chars = sum(length(text[c.unit_id[1:]].text, text[c.unit_id[1:]].lang) for c in verified if c.unit_id[1:] in text)
    return Estimate("E5", float(len(verified)), float(len(verified)), float(len(verified)), chars, scope,
                    ("human-verified census of witness-only claims",))


def translator_notes(texts: Texts) -> list[list[str]]:
    """Witness notes of the classes the decomposition cares about: (id, class, host)."""
    return [[s.id, s.extra.get("note_class", ""), s.extra.get("host", "")] for s in texts.witness
            if s.kind == "note" and s.extra.get("note_class") in TRANSLATOR_NOTE_CLASSES]


# --------------------------------------------------------------------------- report
def report_inputs(ctx: RunContext) -> markdown.ReportInputs:
    """Collect everything ``report.markdown.render`` prints from the run directory."""
    texts = load_texts(ctx)
    consensus = read_consensus(ctx)
    controls = control_alignments(ctx, required=False)
    cells_path = ctx.path("matrix", "cells.jsonl")
    cells = read_built_cells(ctx) if cells_path.is_file() else []
    scores_doc = _read_json(ctx.path("evaluation", "scores.json"))
    scores = {src: {k: estimate_from_dict(v) for k, v in s.get("intervals", {}).items()}
              for src, s in (scores_doc.get("sources") or {}).items()}
    sentinel_path = ctx.path("evaluation", "sentinels.jsonl")
    if not sentinel_path.is_file():
        sentinel_path = ctx.path("ingest", "sentinels.jsonl")
    sentinels = tuple(sentinel_result_from_dict(r) for r in read_jsonl(sentinel_path)) if sentinel_path.is_file() else ()
    est = {k: estimate_from_dict(v) for k, v in _read_json(ctx.path("stats", "estimates.json")).items()}
    details = _read_json(ctx.path("stats", "details.json"))
    return markdown.ReportInputs(
        run_id=ctx.run_dir.name, reference=texts.reference_id, witness=texts.witness_id,
        scope=tuple(scope_lines(ctx)),
        instrument_counts=markdown.alignment_counts(consensus, texts.ref_kinds) if consensus else None,
        control_counts={src: markdown.alignment_counts(a, texts.ref_kinds) for src, a in controls.items()
                        if src != SOURCE_CONSENSUS},
        human_verified=markdown.human_verified_counts(cells, texts.ref_kinds),
        sentinels=sentinels, gold_set=str(scores_doc.get("gold_set", "")), scores=scores,
        witness_only=tuple(markdown.WitnessOnlyRow(c.unit_id, c.chapter, c.relation or "", c.grade.value)
                           for c in cells if is_orphan(c.unit_id)),
        estimates=est, manski={k: (v[0], v[1]) for k, v in (details.get("manski") or {}).items()},
        max_manski_width=float(((ctx.settings.prereg.get("gates") or {}).get("g3") or {}).get("max_manski_width", 0.05)),
        notes=tuple(tuple(n) for n in translator_notes(texts)),
        revision=details.get("revision") or {}, e4_details=details.get("e4") or {},
        reliability={src: s["reliability"] for src, s in (scores_doc.get("sources") or {}).items()
                     if s.get("reliability")},
        e3_details=details.get("e3") or {},
        negation=_read_json(ctx.path("evaluation", "perturbations.json")).get("negation"))


def report(ctx: RunContext) -> Path:
    """Write ``summary.md`` and the SVG figures, gated by the level in ``evaluation/gate.json``."""
    gate = read_gate(ctx) or GateReport(level=0, confirmatory=False, passed={},
                                        reasons=("G0: evaluate has not run in this run directory",))
    stale = stale_gate_inputs(ctx)
    if stale:
        note = (f"report: evaluation/gate.json predates {', '.join(stale)}; the gates (G2, G3) may not reflect them: "
                "rerun `hevajra-matrix evaluate`")
        print(f"report: WARNING {note.removeprefix('report: ')}")
        gate = replace(gate, reasons=(*gate.reasons, note))
    inputs = report_inputs(ctx)
    path = ctx.path("summary.md")
    path.write_text(markdown.render(inputs, gate), encoding="utf-8")
    cells_path = ctx.path("matrix", "cells.jsonl")
    if cells_path.is_file():
        caption = markdown.UNVALIDATED if gate.level == 0 else f"report level {gate.level}"
        cells = read_built_cells(ctx)
        svg.status_strip(cells, ctx.path("status_strip.svg"), caption)
        svg.chapter_heatmap(cells, ctx.path("chapter_heatmap.svg"), caption, shade=gate.level > 0)
    record_stage(ctx, "report")
    print(f"report: {path} (level {gate.level})")
    return path


STALE_CHECKS = (("matrix", "cells.jsonl"), ("evaluation", "perturbations.json"))


def stale_gate_inputs(ctx: RunContext) -> list[str]:
    """Gate inputs written after ``evaluation/gate.json`` (e.g. build after new verdicts)."""
    gate = ctx.path("evaluation", "gate.json")
    if not gate.is_file():
        return []
    when = gate.stat().st_mtime_ns
    return ["/".join(parts) for parts in STALE_CHECKS
            if ctx.path(*parts).is_file() and ctx.path(*parts).stat().st_mtime_ns > when]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
