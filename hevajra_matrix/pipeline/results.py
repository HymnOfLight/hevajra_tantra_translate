"""Claude-free stages that turn cells and verdicts into numbers and the report: stats, report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..collate.consensus import SOURCE_CONSENSUS
from ..core.io import read_jsonl
from ..core.textnorm import length
from ..core.types import Cell, Estimate, Status
from ..evaluation.gate import GateReport
from ..report import markdown, svg
from ..review import sampling
from ..review.verdicts import is_orphan
from ..stats import contrast, decompose, twophase
from . import human_data as ann
from .context import RunContext, Texts, load_texts, record_stage
from .instrument import read_consensus
from .measure import control_alignments, read_built_cells, read_gate
from .store import estimate_from_dict, estimate_to_dict, sentinel_result_from_dict

OUTCOMES = ("any", "cov", "absent", "partial")
NO_PHASE2 = "G3: no phase-2 verification or audit verdict yet"
TRANSLATOR_NOTE_CLASSES = (decompose.VORLAGE_NOTE, decompose.TRANSLATOR_NOTE)


# --------------------------------------------------------------------------- stats
def stats(ctx: RunContext) -> dict[str, Estimate]:
    """E1, E2 (two-phase draws, final and blind columns), Manski bounds, E3, E4, E5.

    Numbers are computed only from phase-2 verdicts: without any, E1/E2 are NOT_ESTIMABLE
    rather than prior-only draws. Whether a number may be printed is the report's job.
    """
    texts = load_texts(ctx)
    cells = [c for c in read_built_cells(ctx) if not is_orphan(c.unit_id)]
    orphans = [c for c in read_built_cells(ctx) if is_orphan(c.unit_id)]
    machine = [c for c in read_built_cells(ctx, machine=True) if not is_orphan(c.unit_id)]
    topics = ann.load_topics(ctx, texts)
    verdicts = ann.current_verdicts(ann.review_verdicts(ctx, texts), texts)
    strata = sampling.machine_strata(machine, topics.groups(), texts.ref_kinds)
    params = ctx.settings.prereg.get("stats") or {}
    n_draws, seed = int(params.get("n_draws", 500)), int(params.get("seed", 0))
    scope = "; ".join(str(s) for s in ctx.settings.prereg.get("scope") or [])
    primary = str(ctx.settings.prereg.get("primary_outcome", "any"))
    est: dict[str, Estimate] = {}
    extra: dict[str, Any] = {}
    draws: dict[str, list] = {}
    if any(v.task in twophase.SAMPLE_TASKS for v in verdicts):
        weights = {s.id: float(length(s.text, s.lang)) for s in texts.units}
        for column in ("final", "blind"):
            draws[column] = twophase.draws(cells, verdicts, strata, n_draws, seed, column, texts.ref_kinds)
            suffix = "" if column == "final" else "_blind"
            for outcome in OUTCOMES if column == "final" else ("any", "cov"):
                est[f"E1_{outcome}{suffix}"] = twophase.prevalence(draws[column], outcome, name=f"E1_{outcome}{suffix}",
                                                                   scope=scope)
            est[f"E2_any{suffix}"] = twophase.prevalence(draws[column], "any", weights, name=f"E2_any{suffix}", scope=scope)
        extra["revision"] = {s: vars(r) for s, r in twophase.revision_rate(verdicts, texts.ref_kinds).items()}
    else:
        for name in ("E1_any", "E1_cov", "E1_absent", "E1_partial", "E2_any"):
            est[name] = Estimate.missing(name, NO_PHASE2, scope)
    if cells:
        extra["manski"] = {o: list(twophase.manski(cells, o, texts.ref_kinds)) for o in ("any", "cov")}
    est["E3"] = decompose.e3((), manuscripts=(), reference=texts.reference_id, cowitness=texts.reference_id,
                             scope=scope)   # the Derge reference is also the only co-witness: NOT_ESTIMABLE
    est["E4"], extra["e4"] = _e4(ctx, texts, cells, topics, draws.get("final"), primary, seed, scope)
    est["E5"] = _e5(orphans, texts, scope)
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


def _e4(ctx: RunContext, texts: Texts, cells: list[Cell], topics: ann.Topics, draws: list | None, outcome: str,
        seed: int, scope: str) -> tuple[Estimate, dict[str, Any]]:
    """Delta (synthesis 6.3) when G4 allows it and draws exist; otherwise the reason."""
    gate = read_gate(ctx)
    reason = (gate.not_estimable.get("E4") if gate is not None else "G4: evaluate has not run")
    if reason is None and draws is None:
        reason = NO_PHASE2
    if reason:
        return Estimate.missing("E4", reason, scope), {}
    groups = topics.groups()
    excluded = {c.unit_id for c in cells if "uniform_across_list" in c.flags or c.status is Status.LACUNA}
    exposure = {u: g == "sensitive" for u, g in groups.items() if g in ("sensitive", "neutral") and u not in excluded}
    lengths = {s.id: float(length(s.text, s.lang)) for s in texts.units if s.id in exposure}
    strata = contrast.tertile_strata(lengths, texts.chapter_of)
    estimate = contrast.delta(draws, exposure, strata, texts.chapter_of, seed, outcome, scope=scope)
    means = twophase.unit_means(draws, outcome)
    n_perm = int((ctx.settings.prereg.get("stats") or {}).get("n_permutations", 2000))
    margin = float((ctx.settings.prereg.get("stats") or {}).get("tost_margin", 0.1))
    overlap = contrast.overlap_diagnostics(exposure, strata)
    return estimate, {"permutation_p": contrast.permutation_p(means, exposure, strata, n_perm, seed),
                      "equivalent_within_margin": contrast.tost(estimate, margin), "margin": margin,
                      "overlap": asdict(overlap)}


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
        scope=tuple(str(s) for s in ctx.settings.prereg.get("scope") or ()),
        instrument_counts=markdown.alignment_counts(consensus, texts.ref_kinds) if consensus else None,
        control_counts={src: markdown.alignment_counts(a, texts.ref_kinds) for src, a in controls.items()
                        if src != SOURCE_CONSENSUS},
        human_verified=markdown.human_verified_counts(cells, texts.ref_kinds),
        sentinels=sentinels, gold_set=str(scores_doc.get("gold_set", "")), scores=scores,
        witness_only=tuple(markdown.WitnessOnlyRow(c.unit_id, c.chapter, c.relation or "", c.grade.value)
                           for c in cells if is_orphan(c.unit_id)),
        estimates=est, manski={k: (v[0], v[1]) for k, v in (details.get("manski") or {}).items()},
        max_manski_width=float(((ctx.settings.prereg.get("gates") or {}).get("g3") or {}).get("max_manski_width", 0.05)),
        notes=tuple(tuple(n) for n in translator_notes(texts)))


def report(ctx: RunContext) -> Path:
    """Write ``summary.md`` and the SVG figures, gated by the level in ``evaluation/gate.json``."""
    gate = read_gate(ctx) or GateReport(level=0, confirmatory=False, passed={},
                                        reasons=("G0: evaluate has not run in this run directory",))
    inputs = report_inputs(ctx)
    path = ctx.path("summary.md")
    path.write_text(markdown.render(inputs, gate), encoding="utf-8")
    cells_path = ctx.path("matrix", "cells.jsonl")
    if cells_path.is_file():
        caption = markdown.UNVALIDATED if gate.level == 0 else f"report level {gate.level}"
        cells = read_built_cells(ctx)
        svg.status_strip(cells, ctx.path("status_strip.svg"), caption)
        svg.chapter_heatmap(cells, ctx.path("chapter_heatmap.svg"), caption)
    record_stage(ctx, "report")
    print(f"report: {path} (level {gate.level})")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

