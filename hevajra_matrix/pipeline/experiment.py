"""The over-attribution experiment as three stages (synthesis 8; critique B12):

    plan    validate the item bank, write the seeded trial schedule (trials.jsonl)
    run     T4 subject and T5 scorer calls for every trial (responses.jsonl; fallback off)
    score   analysis: H1/H2 with Holm, exploratory H3, refusal bounds, scorer agreement
            with the human codes, and the blind human coding sheet (results.json)

Outputs live under ``<run_dir>/experiments/overattribution/``; they hold model output and
are never committed. The item bank and human codes are committed under
``data/experiments/overattribution/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ..core.io import read_csv, read_jsonl, write_jsonl
from ..experiments.overattribution import analysis, design, lexical, run, score
from .context import RunContext, StageError, has_api_key, load_texts, make_client, record_stage, require

EXPERIMENT = ("experiments", "overattribution")


def _bank(ctx: RunContext) -> Path:
    return ctx.data_dir / "experiments" / "overattribution"


def _schedule(ctx: RunContext, phase: str) -> list[design.Trial]:
    params = design.ExperimentParams.from_prereg(ctx.settings.prereg)
    items = design.items_in_phase(design.load_items(_bank(ctx) / "items.csv"), phase)
    if not items:
        raise StageError(f"experiment: no {phase} item in {_bank(ctx) / 'items.csv'} (fill the item bank first)")
    return design.trials(items, params.conditions, params.replicates, params.seed)


def experiment_plan(ctx: RunContext, phase: str = "main") -> int:
    """Validate the bank and write the schedule; report shortfalls against the prereg size."""
    params = design.ExperimentParams.from_prereg(ctx.settings.prereg)
    shortfalls = design.bank_shortfalls(design.load_items(_bank(ctx) / "items.csv"), params)
    schedule = _schedule(ctx, phase)
    write_jsonl(ctx.path(*EXPERIMENT, "trials.jsonl"),
                ({"trial_id": t.trial_id, "item_id": t.item_id, "arm": t.arm, "condition": t.condition,
                  "evidence": t.evidence, "replicate": t.replicate, "order": t.order} for t in schedule))
    record_stage(ctx, "experiment-plan")
    print(f"experiment plan: {len(schedule)} {phase} trials" + (f"; bank short: {'; '.join(shortfalls)}"
                                                                   if shortfalls else ""))
    return len(schedule)


def experiment_run(ctx: RunContext, phase: str = "main") -> int:
    """Run every trial of the schedule (subject, then scorer; a seeded 20% scored twice)."""
    if not ctx.offline and not has_api_key():
        raise StageError("experiment run needs ANTHROPIC_API_KEY, or --offline with a filled cache")
    texts = load_texts(ctx)
    schedule = _schedule(ctx, phase)
    llm = ctx.settings.llm
    subject_s = run.ExperimentTaskSettings.from_config(llm, run.SUBJECT_TASK)
    scorer_s = run.ExperimentTaskSettings.from_config(llm, run.SCORER_TASK)
    seed = design.ExperimentParams.from_prereg(ctx.settings.prereg).seed
    double = design.double_score_ids([t.trial_id for t in schedule], scorer_s.double_score_fraction, seed)
    outcomes = run.run_trials(schedule, run.segment_resolver(texts.units, texts.witness), make_client(ctx), subject_s,
                              scorer_s, design.load_evidence(_bank(ctx) / "evidence.yaml"),
                              lexical.load_motive_lexicon(ctx.data_dir), double)
    write_jsonl(ctx.path(*EXPERIMENT, "responses.jsonl"), (score.outcome_record(o) for o in outcomes))
    record_stage(ctx, "experiment-run")
    print(f"experiment run: {len(outcomes)} trials, {sum(1 for o in outcomes if o.measured)} measured")
    return len(outcomes)


def outcome_from_record(record: dict[str, Any]) -> score.TrialOutcome:
    """Inverse of ``score.outcome_record``."""
    names = {f.name for f in fields(score.TrialOutcome)}
    values = {k: v for k, v in record.items() if k in names}
    values["flags"] = frozenset(values.get("flags") or ())
    values["stances"] = dict(values.get("stances") or {})
    return score.TrialOutcome(**values)


def experiment_score(ctx: RunContext, n_boot: int = 10000) -> dict[str, Any]:
    """Analyse the responses; write results.json and the blind human coding sheet."""
    outcomes = [outcome_from_record(r) for r in read_jsonl(require(ctx.path(*EXPERIMENT, "responses.jsonl"),
                                                                   "experiment overattribution run"))]
    params = analysis.AnalysisParams.from_prereg(ctx.settings.prereg, n_boot=n_boot)
    codes_path = _bank(ctx) / "human_codes.csv"
    human = analysis.load_human_codes(codes_path) if codes_path.is_file() and read_csv(codes_path) else None
    results = analysis.analyse(outcomes, params, human)
    exp = ctx.settings.prereg.get("experiment") or {}
    sample = analysis.human_sample(outcomes, int(exp.get("human_coded_responses", 150)), params.seed)
    analysis.write_coding_sheet(ctx.path(*EXPERIMENT, "human_coding_sheet.csv"), outcomes, sample)
    payload = asdict(results)
    path = ctx.path(*EXPERIMENT, "results.json")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=list) + "\n", encoding="utf-8")
    record_stage(ctx, "experiment-score")
    for t in results.tests:
        if t.role in ("confirmatory", "exploratory"):
            print(f"experiment {t.name} ({t.role}): estimate {t.estimate} [{t.ci_low}, {t.ci_high}], "
                  f"p {t.p_value}, Holm {t.p_holm}")
    return payload
