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
                                replicates.json, consensus.json, integrity.json,
                                diagnostics.jsonl, dry_run.json
    matrix/                     cells.csv, units.csv, wide_status.csv, stale_verdicts.csv,
                                cells.jsonl (human decisions applied), machine_cells.jsonl
    evaluation/                 scores.json, gate.json, sentinels.jsonl, perturbations.json
    topics/prelabels.jsonl      T3 hints
    review/                     plan_<batch>.csv and the sheets (licensed text; never committed)
    stats/                      estimates.json, details.json (Manski bounds, revision rates, E4 tests)
    components/                 components.jsonl, rendering_profile.csv, diagnostics.jsonl
    experiments/overattribution/  trials.jsonl, responses.jsonl, results.json, coding sheet
    summary.md, status_strip.svg, chapter_heatmap.svg

Modules: ``context`` (RunContext, run directories, ``make_client``, text loading), ``store``
(JSON forms), ``texts`` (fetch, ingest, baselines), ``instrument`` (collate, perturb,
components, claude-check), ``measure`` (build, evaluate), ``results`` (stats, report),
``review`` (topics, sample, review sheets), ``human_data`` (committed annotations), ``experiment``.
"""

from __future__ import annotations

from pathlib import Path

from .context import RunContext, StageError, has_api_key, latest_run_dir, make_client, new_run_dir
from .experiment import experiment_plan, experiment_run, experiment_score
from .instrument import claude_check, collate, components_stage, perturb, plan_collation
from .measure import build, evaluate
from .results import report, stats
from .review import review_export, review_import, review_status, sample, topics_prelabel
from .texts import baselines, baselines_import, fetch, ingest

__all__ = [
    "RunContext", "StageError", "baselines", "baselines_import", "build", "claude_check", "collate",
    "components_stage", "evaluate", "experiment_plan", "experiment_run", "experiment_score", "fetch", "ingest",
    "latest_run_dir", "make_client", "new_run_dir", "perturb", "plan_collation", "report", "review_export",
    "review_import", "review_status", "run", "sample", "stats", "topics_prelabel",
]


def run(ctx: RunContext, raw_dir: Path | None = None, gold_set: str = "test") -> int:
    """ingest -> baselines -> collate -> build -> evaluate -> stats -> report, as far as the
    data and credentials allow; returns the report level.

    Collate is skipped (with a message) when there is no API key and the run is not
    offline; build and stats need a collation and are skipped without one. Evaluate and
    report always run, so a run without Claude still yields the baseline scores and a
    level-0 report.
    """
    ingest(ctx, raw_dir)
    baselines(ctx)
    if ctx.offline or has_api_key():
        collate(ctx)
    else:
        print("collate: skipped (no ANTHROPIC_API_KEY and not --offline); the report shows the controls only")
    if ctx.path("collation", "replicates.json").is_file():
        build(ctx)
    gate = evaluate(ctx, gold_set=gold_set)
    if ctx.path("matrix", "cells.jsonl").is_file():
        stats(ctx)
    else:
        print("stats: skipped (no matrix: collate and build have not run)")
    report(ctx)
    return gate.level
