"""Pipeline stages: plain functions ``stage(ctx, ...)`` that communicate through files.

There is no DAG engine (synthesis 2). Each stage reads what its predecessors wrote into
the run directory and writes its own files; ``run`` chains the main stages. A missing
input raises ``StageError`` naming the command that produces it.

Run directory ``runs/<UTC timestamp>-<git short hash>/``::

    manifest.json               version, git commit, config and input sha256, instrument
                                digests, served models, cache hits/misses, stages run
    llm_audit.jsonl             one line per LLM call (no text)
    ingest/                     segments_<witness>.jsonl, footnotes.jsonl, variants.csv,
                                witness_meta.json, report.json, g0.json, sentinels.jsonl
    alignments/                 dp_zero.jsonl (P1), dp_anchor.jsonl (B0), external_<name>.jsonl,
                                claude.r<k>.jsonl (replicates), claude.jsonl (consensus),
                                shuffled.jsonl (P2)
    collation/                  r<k>.json (UNALIGNED reasons, diagnostics, hints, overlap),
                                replicates.json (incl. request keys), consensus.json, integrity.json,
                                diagnostics.jsonl, dry_run.json
    matrix/                     cells.csv, units.csv, wide_status.csv, stale_verdicts.csv,
                                cells.jsonl (human decisions applied), machine_cells.jsonl
    evaluation/                 scores.json, gate.json, sentinels.jsonl (the gating evaluation: Claude on
                                test gold), scores.<set>[_baselines].json (every evaluation), perturbations.json
    topics/prelabels.jsonl      T3 hints
    review/                     the sheets (licensed text; never committed); review plans and their
                                frozen strata are committed under data/annotations/verdicts/<witness>/
                                (older runs kept plan_<batch>.csv, strata_<batch>.json here)
    stats/                      estimates.json, details.json (Manski bounds, revision rates, E4 tests and
                                diagnostics), power.json (MDE on the real topic labels)
    components/                 components.jsonl, rendering_profile.csv, diagnostics.jsonl
    experiments/overattribution/  trials.jsonl, responses.jsonl, results.json, coding sheet (main phase;
                                the pilot's own files under pilot/)
    summary.md, status_strip.svg, chapter_heatmap.svg
    witnesses/<witness>/        in Sanskrit mode (``RunContext.sanskrit_mode``): the same per-witness
                                layout (alignments/ ... summary.md) for the Derge, aligned to the
                                Sanskrit reference like T0892; ingest/, the manifest, the audit log,
                                topics/ and experiments/ stay shared at the top

Modules: ``context`` (RunContext, run directories, ``make_client``, text loading), ``store``
(JSON forms), ``texts`` (fetch, ingest, baselines), ``instrument`` (collate, perturb,
components, claude-check), ``measure`` (build, evaluate), ``results`` (stats, report), ``e3``
(E3: manuscript readings, co-witness, A2 bound), ``e4``
(the E4 diagnostics and the MDE on real labels),
``review`` (topics, sample, review sheets), ``human_data`` (committed annotations), ``experiment``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .context import RunContext, StageError, has_api_key, latest_run_dir, make_client, new_run_dir
from .experiment import experiment_plan, experiment_run, experiment_score
from .instrument import CONSENSUS_FILE, claude_check, collate, components_stage, perturb, plan_collation
from .measure import build, evaluate, test_scoring_refused
from .results import report, stats
from .review import review_export, review_import, review_status, sample, topics_prelabel
from .texts import baselines, baselines_import, fetch, ingest

__all__ = [
    "RunContext", "StageError", "baselines", "baselines_import", "build", "claude_check", "collate",
    "components_stage", "evaluate", "experiment_plan", "experiment_run", "experiment_score", "fetch", "ingest",
    "latest_run_dir", "make_client", "new_run_dir", "perturb", "plan_collation", "report", "review_export",
    "review_import", "review_status", "run", "sample", "stats", "topics_prelabel", "witness_contexts",
]


def witness_contexts(ctx: RunContext) -> Iterator[RunContext]:
    """``ctx.each_witness()``, announcing the witness when the run aligns several."""
    contexts = ctx.each_witness()
    for one in contexts:
        if len(contexts) > 1:
            print(f"--- witness {one.witness} (reference {one.reference})")
        yield one


def run(ctx: RunContext, raw_dir: Path | None = None, gold_set: str = "test") -> int:
    """ingest -> baselines -> collate -> build -> evaluate -> stats -> report, as far as the
    data and credentials allow; returns the report level of the target witness.

    Every stage after ingest runs for each aligned witness in turn (the Derge too when the
    Sanskrit reference is present), stage by stage, so that ``stats`` of one witness finds
    the matrix of the other (its co-witness in E3).

    Collate is skipped (with a message) when there is no API key and the run is not
    offline; build and stats need a collation and are skipped without one. Evaluate and
    report always run, so a run without Claude still yields the baseline scores and a
    level-0 report. Before ``prereg freeze`` the consensus is never scored on test gold:
    with test gold present, ``run`` then evaluates dev gold (non-gating) and goes on.
    """
    ingest(ctx, raw_dir)
    for one in witness_contexts(ctx):
        baselines(one)
    if ctx.offline or has_api_key():
        for one in witness_contexts(ctx):
            collate(one)
    else:
        print("collate: skipped (no ANTHROPIC_API_KEY and not --offline); the report shows the controls only")
    for one in witness_contexts(ctx):
        if one.path("collation", "replicates.json").is_file():
            build(one)
    levels = {}
    for one in witness_contexts(ctx):
        chosen = gold_set
        if (chosen == "test" and one.path("alignments", CONSENSUS_FILE).is_file()
                and test_scoring_refused(one)):
            print("evaluate: the preregistration is not frozen, so the Claude consensus is not scored on test "
                  "gold; scoring dev gold instead (run `hevajra-matrix prereg freeze`, then evaluate test gold)")
            chosen = "dev"
        levels[one.witness] = evaluate(one, gold_set=chosen).level
    for one in witness_contexts(ctx):
        if one.path("matrix", "cells.jsonl").is_file():
            stats(one)
        else:
            print("stats: skipped (no matrix: collate and build have not run)")
    for one in witness_contexts(ctx):
        report(one)
    return levels[ctx.witness]
