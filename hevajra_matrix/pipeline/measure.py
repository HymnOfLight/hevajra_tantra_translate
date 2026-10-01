"""Claude-free stages that turn alignments into cells and scores: build and evaluate."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..align.placebo import SOURCE_SHUFFLED, shuffle_alignment
from ..collate.consensus import SOURCE_CONSENSUS, agreement, consensus
from ..collate.perturb import Rate
from ..core.io import git_commit, read_jsonl, write_jsonl
from ..core.types import REASON_UNASSESSED, REASON_VERIFICATION_FAILED, Alignment, Cell, OutcomeClass, Status, Verdict
from ..evaluation import gate as gates
from ..evaluation import gold as gold_sets
from ..evaluation import sentinels as sentinel_checks
from ..matrix.build import build_cells
from ..matrix.export import write_matrix
from ..matrix.status import cell_outcome
from ..review import sampling
from ..review.verdicts import is_orphan
from ..stats import twophase
from ..topics import human_agreement
from . import human_data as ann
from .context import AUDIT_LOG, RunContext, StageError, Texts, all_segments, load_texts, record_stage, require
from .e3 import e3_reason
from .instrument import CONSENSUS_FILE, collate_digest, read_replicates
from .store import (
    diagnostic_from_dict,
    estimate_to_dict,
    gate_from_dict,
    gate_to_dict,
    read_alignment,
    read_cells,
    sentinel_result_to_dict,
    write_alignment,
    write_cells,
)
from .texts import BASELINE_FILES, EXTERNAL_PREFIX, SENTINELS_FILE

SHUFFLED_FILE = "shuffled.jsonl"
TOPIC_BREAKDOWN = "refusal_rate:"     # metric keys per topic group (evaluation.scores.refusal_by_group)
_V4 = re.compile(r"\bV4\b")


# --------------------------------------------------------------------------- build
def build(ctx: RunContext) -> dict[str, int]:
    """Consensus of the replicates, P2, and the matrix with gold and verdicts applied.

    Writes ``alignments/claude.jsonl`` and ``shuffled.jsonl``, ``collation/consensus.json``
    and ``integrity.json``, and ``matrix/`` (the CSV exports, ``cells.jsonl`` with human
    decisions applied, and ``machine_cells.jsonl`` without them: sampling strata must be
    computed on machine cells).
    """
    texts = load_texts(ctx)
    replicates = read_replicates(ctx)
    cons = consensus(replicates, texts.ref_kinds)
    agree = agreement(replicates, texts.ref_kinds)
    write_alignment(ctx.path("alignments", CONSENSUS_FILE), cons.alignment)
    seed = int((ctx.settings.prereg.get("stats") or {}).get("seed", 0))
    write_alignment(ctx.path("alignments", SHUFFLED_FILE), shuffle_alignment(cons.alignment, texts.chapter_of, seed))
    quote_failures = sum(1 for r in replicates for d in r.diagnostics
                         if d.kind == REASON_VERIFICATION_FAILED and _V4.search(d.detail))
    n_units = len(texts.units)
    integrity = {"replicates": len(replicates), "units": n_units, "quote_failures": quote_failures,
                 "quote_failure_rate": quote_failures / (n_units * len(replicates)) if n_units else None,
                 "missing_units": sum(1 for r in cons.reasons.values() if r == REASON_UNASSESSED),
                 "fleiss_kappa": agree.fleiss_kappa, "mean_link_jaccard": agree.mean_link_jaccard,
                 "class_jaccard": dict(agree.class_jaccard)}
    integrity["missing_rate"] = integrity["missing_units"] / n_units if n_units else None
    _write_json(ctx.path("collation", "consensus.json"),
                {"grades": {k: str(v) for k, v in cons.grades.items()}, "reasons": dict(cons.reasons)})
    _write_json(ctx.path("collation", "integrity.json"), integrity)
    write_jsonl(ctx.path("collation", "diagnostics.jsonl"),
                ({"replicate": n + 1, "kind": d.kind, "ref_ids": list(d.ref_ids), "wit_ids": list(d.wit_ids),
                  "detail": d.detail} for n, r in enumerate(replicates) for d in r.diagnostics))
    groups = texts.groups()
    machine, machine_orphans, _ = build_cells(texts.units, cons.alignment, cons.grades, cons.reasons, (), (),
                                              texts.witness, groups)
    cells, orphans, stale = build_cells(texts.units, cons.alignment, cons.grades, cons.reasons,
                                        ann.gold_verdicts(ctx, texts), ann.review_verdicts(ctx, texts),
                                        texts.witness, groups)
    write_cells(ctx.path("matrix", "machine_cells.jsonl"), [*machine, *machine_orphans])
    write_cells(ctx.path("matrix", "cells.jsonl"), [*cells, *orphans])
    write_matrix(cells, orphans, stale, ctx.path("matrix"), reference_segments=texts.units)
    record_stage(ctx, "build")
    counts = {s.value: sum(1 for c in cells if c.status is s) for s in Status if s is not Status.NA}
    print(f"build: {len(cells)} cells {counts}, {len(orphans)} witness-only rows, {len(stale)} stale verdicts; "
          f"replicate Fleiss kappa {agree.fleiss_kappa}")
    return {**counts, "orphans": len(orphans), "stale": len(stale)}


def read_built_cells(ctx: RunContext, machine: bool = False) -> list[Cell]:
    name = "machine_cells.jsonl" if machine else "cells.jsonl"
    return read_cells(require(ctx.path("matrix", name), "build"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# --------------------------------------------------------------------------- evaluate
def control_alignments(ctx: RunContext, required: bool = True) -> dict[str, Alignment]:
    """P1, B0 and every imported external alignment of this run (source -> alignment).

    Missing baselines raise ``StageError`` when ``required``, else they are left out.
    """
    out = {}
    for source, name in BASELINE_FILES.items():
        path = ctx.path("alignments", name)
        if required or path.is_file():
            out[source] = read_alignment(require(path, "baselines"))
    for path in sorted(ctx.path("alignments").glob("external_*.jsonl")):
        alignment = read_alignment(path)
        if alignment.source.startswith(EXTERNAL_PREFIX):
            out[alignment.source] = alignment
    return out


def evaluate(ctx: RunContext, gold_set: str = "test", baselines_only: bool = False) -> gates.GateReport:
    """Score every aligner on gold, check sentinels, and evaluate the gates G0-G4.

    ``baselines_only`` scores the controls without any Claude output (critique A5: the
    baseline P/R report on dev gold exists before any call). Scoring the Claude consensus
    on test gold appends one record to the evaluation ledger (unless an identical scoring is
    already recorded), and is refused while the preregistration is not frozen (the protocol
    freezes before the test set is scored).

    Only that evaluation gates: it alone writes ``gate.json`` and ``scores.json`` once it
    exists. Dev-gold and baselines-only evaluations get G1 failed and level 0 (dev gold never
    gates), write ``scores.<set>[_baselines].json``, and never replace a gating gate.
    """
    if gold_set not in gold_sets.SETS[:2]:
        raise StageError(f"--gold must be dev or test, not {gold_set!r}")
    texts = load_texts(ctx)
    aligners = control_alignments(ctx)
    consensus_path = ctx.path("alignments", CONSENSUS_FILE)
    claude = None if baselines_only or not consensus_path.is_file() else read_alignment(consensus_path)
    if claude is not None:
        aligners[SOURCE_CONSENSUS] = claude
        aligners[SOURCE_SHUFFLED] = read_alignment(require(ctx.path("alignments", SHUFFLED_FILE), "build"))
    cells_path = ctx.path("matrix", "cells.jsonl")
    cells = read_cells(cells_path) if cells_path.is_file() and not baselines_only else None
    topics = ann.load_topics(ctx, texts)
    if gold_set == "test" and claude is not None and test_scoring_refused(ctx, texts):
        raise StageError("test gold is scored against Claude only after `hevajra-matrix prereg freeze`; "
                         "use --gold dev or --baselines-only until then")
    scores, human_kappa = _scores(ctx, texts, aligners, gold_set, topics.groups())
    integrity = _integrity(ctx, texts, claude, cells)
    digest = collate_digest(ctx, texts)
    prereg_sha = ctx.settings.shas.get("preregistration.yaml", "")
    ledger = ann.ledger_path(ctx)
    specs = gates.GateSpecs.from_prereg(ctx.settings.prereg)
    gating = gold_set == "test" and SOURCE_CONSENSUS in scores
    if gating:
        g1 = gates.check_g1(scores, human_kappa, specs.g1, specs.seed)
        record = gates.ledger_record(digest, prereg_sha, git_commit(ctx.root),
                                     {k: v.metrics() for k, v in scores.items()}, g1.passed)
        if already_ledgered(ledger, record):
            print("ledger: this test scoring (same instrument digest, preregistration and scores) is already "
                  "recorded; not appended again (re-gating after review does not score the test set anew)")
        else:
            gates.ledger_append(ledger, record)
    report = gates.evaluate(scores, human_kappa, integrity, _perturbations(ctx), _calibration(ctx, texts, cells, topics),
                            ctx.settings.prereg, digest, gates.ledger_summary(ledger, digest, prereg_sha))
    if not gating:
        report = not_gating(report, gold_set, baselines_only)
    out = ctx.path("evaluation")
    scores_doc = {
        "gold_set": gold_set, "baselines_only": baselines_only, "human_kappa": human_kappa,
        "sources": {src: {"counts": s.counts(), "metrics": s.metrics(),
                          "intervals": {k: estimate_to_dict(e) for k, e in
                                        gold_sets.interval_estimates(s, specs.g1.n_boot, specs.seed).items()},
                          "reliability": gold_sets.confidence_reliability(s)}
                    for src, s in scores.items()}}
    own = out / f"scores.{gold_set}{'_baselines' if baselines_only else ''}.json"     # every evaluation keeps its own
    _write_json(own, scores_doc)
    if not gating and _read_json(out / "gate.json").get("gating") is True:
        print(f"evaluate: gate.json and scores.json keep the test-gold Claude evaluation; this one never gates "
              f"(scores in {own})")
    else:
        _write_json(out / "scores.json", scores_doc)
        _write_json(out / "gate.json", {**gate_to_dict(report), "gating": gating, "gold_set": gold_set,
                                        "baselines_only": baselines_only})
        write_jsonl(out / "sentinels.jsonl",
                    (sentinel_result_to_dict(r) for rs in integrity.sentinels.values() for r in rs))
    record_stage(ctx, "evaluate")
    if not scores:
        print(f"evaluate: no {gold_set} gold for {texts.witness_id} yet "
              f"({ann.gold_dir(ctx, texts.witness_id) / (gold_set + '.csv')}); nothing scored")
    for src, s in scores.items():
        m = s.metrics()
        print(f"evaluate {gold_set} {src}: {s.counts()['units']} units, link F1 {_fmt(m['link_f1'])}, "
              f"status kappa {_fmt(m['status_kappa'])}")
    print(f"gates: {dict(report.passed)} -> report level {report.level}"
          f"{'' if report.confirmatory else ' (' + report.scope_note + ')'}")
    return report


def test_scoring_refused(ctx: RunContext, texts: Texts | None = None) -> bool:
    """True when test gold exists but the preregistration is not frozen yet: the Claude
    consensus may not be scored on it (``evaluate`` refuses; ``run`` falls back to dev)."""
    if ctx.settings.prereg.get("frozen") is True:
        return False
    return ann.load_gold(ctx, texts or load_texts(ctx), "test") is not None


def already_ledgered(ledger: Path, record: Mapping[str, Any]) -> bool:
    """True when the ledger holds a record of the same instrument digest, preregistration,
    scores and gate outcome: re-gating identical output on identical gold (e.g. to refresh
    G2/G3 after review) reveals nothing new about the test set and is not a new scoring.
    Breakdowns by topic group are left out of the comparison: they change when topic labels
    are added after the test scoring (the campaign order), not when the instrument or the
    gold does."""
    def key(r: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = {src: {k: v for k, v in (m or {}).items() if not k.startswith(TOPIC_BREAKDOWN)}
                   for src, m in (r.get("metrics") or {}).items()}
        return (r.get("instrument_digest"), r.get("prereg_sha256"), r.get("gate"),
                json.dumps(json.loads(json.dumps(metrics)), sort_keys=True))

    if not ledger.is_file():
        return False
    return any(key(r) == key(record) for r in read_jsonl(ledger))


def not_gating(report: gates.GateReport, gold_set: str, baselines_only: bool) -> gates.GateReport:
    """The gate of an evaluation that cannot gate: dev gold never gates (synthesis 5.4) and a
    baselines-only run has no instrument. G1 is replaced by that reason, the level is 0
    and the result is never confirmatory."""
    what = "baselines only, no Claude output" if baselines_only else f"{gold_set} gold"
    if gold_set == "test" and not baselines_only:
        what = "no Claude consensus scored on test gold"
    reason = f"G1: not a gating evaluation ({what}); only the Claude consensus scored on test gold gates"
    reasons = (reason,) + tuple(r for r in report.reasons if not r.startswith("G1:"))
    return replace(report, level=0, confirmatory=False, passed={**report.passed, "G1": False}, reasons=reasons,
                   scope_note=f"exploratory; not a gating evaluation ({what})")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"




def _scores(ctx: RunContext, texts: Texts, aligners: Mapping[str, Alignment], gold_set: str,
            topic_groups: Mapping[str, str]) -> tuple[dict[str, gold_sets.AlignmentScores], float | None]:
    gold = ann.load_gold(ctx, texts, gold_set)
    if gold is None:
        return {}, None
    consensus_meta = _read_json(ctx.path("collation", "consensus.json"))
    diagnostics_path = ctx.path("collation", "diagnostics.jsonl")
    diagnostics = [diagnostic_from_dict(d) for d in read_jsonl(diagnostics_path)] \
        if SOURCE_CONSENSUS in aligners and diagnostics_path.is_file() else []
    scores = {}
    for source, alignment in aligners.items():
        is_claude = source == SOURCE_CONSENSUS
        scores[source] = gold_sets.score(alignment, gold, ref_kinds=texts.ref_kinds,
                                         unresolved=consensus_meta.get("reasons") if is_claude else None,
                                         diagnostics=diagnostics if is_claude else (), topic_groups=topic_groups)
    second = ann.load_gold(ctx, texts, "test_second") if gold_set == "test" else None
    human = gold_sets.human_kappa(gold, second, texts.ref_kinds) if second is not None else None
    return scores, human


def _integrity(ctx: RunContext, texts: Texts, claude: Alignment | None, cells: list[Cell] | None) -> gates.Integrity:
    everything = all_segments(ctx)
    loaded = sentinel_checks.applicable(sentinel_checks.load(ctx.data_dir / SENTINELS_FILE),
                                        sentinel_checks.text_prefixes(texts.reference),
                                        sentinel_checks.text_prefixes(texts.witness))
    results = {"ingest": sentinel_checks.check(loaded, "ingest", everything)}
    if claude is not None:
        results["proposal"] = sentinel_checks.check(loaded, "proposal", everything, alignment=claude)
    if cells is not None:
        results["final"] = sentinel_checks.check(loaded, "final", everything, cells=cells)
    collation = _read_json(ctx.path("collation", "integrity.json")) if claude is not None else {}
    audit = [r for r in read_jsonl(ctx.path(AUDIT_LOG))] if ctx.path(AUDIT_LOG).is_file() else []
    # G2 looks only at the calls that fed the consensus: the requests of the latest collate
    # (``replicates.json: request_keys``), not perturbation or other collate-task calls.
    # Runs written before the keys were recorded fall back to every collate call.
    keys = _read_json(ctx.path("collation", "replicates.json")).get("request_keys")
    fed = None if keys is None else set(keys)
    measured = [r for r in audit if r.get("task") == "collate" and (fed is None or r.get("key") in fed)]
    substituted = sum(1 for r in measured if r.get("fallback_used") or r.get("served_model") != r.get("requested_model"))
    return gates.Integrity(
        ingest_reports=_read_json(require(ctx.path("ingest", "report.json"), "ingest")), sentinels=results,
        replicate_kappa=collation.get("fleiss_kappa"), quote_failure_rate=collation.get("quote_failure_rate"),
        missing_rate=collation.get("missing_rate"), substituted_calls=substituted if measured else None)


def _perturbations(ctx: RunContext) -> gates.Perturbations:
    found = _read_json(ctx.path("evaluation", "perturbations.json"))

    def rate(key: str) -> Rate | None:
        value = found.get(key)
        return Rate(int(value["hits"]), int(value["n"])) if value else None

    return gates.Perturbations(wrong_window=rate("wrong_window"), deletion=rate("deletion"), negation=rate("negation"))


def _calibration(ctx: RunContext, texts: Texts, cells: list[Cell] | None, topics: ann.Topics) -> gates.Calibration:
    """Two-phase facts for G3 and estimand facts for G4 (see ``evaluation.gate.Calibration``)."""
    verdicts = ann.current_verdicts(ann.review_verdicts(ctx, texts), texts)
    plan = ann.review_plans(ctx, texts)
    coverage = sampling.coverage(plan, verdicts)
    planned = {s: c["planned"] for s, c in coverage.items()}
    achieved = {s: c["final"] for s, c in coverage.items()}
    floor = gates.GateSpecs.from_prereg(ctx.settings.prereg).g3.min_audit_per_stratum
    census = {s for s in planned if all(i.inclusion_prob >= 1.0 for i in plan if i.stratum == s)}
    # A fully verified census of a stratum smaller than the floor meets it: draw_audit plans
    # min(floor, |U_h|) and a census has no sampling error (synthesis 6.2).
    audit = {s: max(n, floor) if s in census and n >= planned[s] else n
             for s, n in achieved.items() if s.startswith("neg:")}
    absent = [v for v in verdicts if v.stratum.startswith("pos:absent") and v.blind_relation]
    null_precision = sum(1 for v in absent if v.blind_relation == "no_counterpart") / len(absent) if absent else None
    units = [c for c in cells or [] if not is_orphan(c.unit_id)]
    unresolved = sum(1 for c in units if c.status is Status.UNALIGNED) if units else None
    width = None
    uncalibrated: dict[str, int] = {}
    null_recall = None
    if units:
        lo, hi = twophase.manski(units, ctx.settings.prereg.get("primary_outcome", "any"), texts.ref_kinds)
        width = hi - lo
        machine_path = ctx.path("matrix", "machine_cells.jsonl")
        if machine_path.is_file():
            machine = [c for c in read_cells(machine_path) if not is_orphan(c.unit_id)]
            strata = ann.sampling_strata(ctx, texts.witness_id,
                                         sampling.machine_strata(machine, topics.groups(), texts.ref_kinds))
            uncalibrated = twophase.uncalibrated_strata(units, verdicts, strata, "final", texts.ref_kinds)
            null_recall = deferred_null_recall(machine, units, verdicts, strata, texts.ref_kinds,
                                               ctx.settings.prereg.get("stats") or {})
    e3 = e3_reason(ctx, texts)
    agreement_topics = human_agreement(topics.labels)
    scorer = _read_json(ctx.path("experiments", "overattribution", "results.json")).get("scorer") or {}
    return gates.Calibration(
        planned=planned, achieved=achieved, audit=audit, null_precision=null_precision, null_recall=null_recall,
        unresolved_units=unresolved, manski_width=width, uncalibrated=uncalibrated,
        e3_not_estimable=e3.removeprefix("G4: ") if e3 else None, topic_labels_complete=topics.complete,
        topic_kappa=agreement_topics.group_kappa, mde=design_mde(ctx), scorer_kappa=scorer.get("kappa_y_over"))


def deferred_null_recall(machine: Sequence[Cell], cells: Sequence[Cell], verdicts: Sequence[Verdict],
                         strata: Mapping[str, str], ref_kinds: Mapping[str, str],
                         params: Mapping[str, Any]) -> float | None:
    """NULL recall from imputation (synthesis 5.5, G3): over the two-phase draws on the blind
    column, the share of units whose completed outcome is ABSENT that the machine marked
    ABSENT, averaged over draws. None without phase-2 sample verdicts or when a stratum has
    no sample to impute from (G3 then says so)."""
    if not any(v.task in twophase.SAMPLE_TASKS for v in verdicts):
        return None
    try:
        vectors = twophase.draws(cells, verdicts, strata, int(params.get("n_draws", 500)),
                                 int(params.get("seed", 0)), "blind", ref_kinds)
    except twophase.UncalibratedStrata:
        return None
    flagged = {c.unit_id for c in machine if cell_outcome(c, ref_kinds.get(c.unit_id, "")) is OutcomeClass.ABSENT}
    values = []
    for vector in vectors:
        absent = [u for u, cls in vector.items() if cls is OutcomeClass.ABSENT]
        if absent:
            values.append(sum(1 for u in absent if u in flagged) / len(absent))
    return sum(values) / len(values) if values else None


def design_mde(ctx: RunContext) -> float | None:
    """MDE for G4: the preregistered one, or the larger one simulated on the real topic labels
    (``stats/power.json``, written by the stats stage) once it exists; None when the
    simulation found no detectable effect within its grid. With ``stats.mde`` unset the
    simulated MDE is used alone (the maximum of the values present)."""
    mde = (ctx.settings.prereg.get("stats") or {}).get("mde")
    power = _read_json(ctx.path("stats", "power.json"))
    if "mde_on_labels" not in power:
        return mde
    computed = power["mde_on_labels"]
    if computed is None:
        return None
    return max(float(x) for x in (mde, computed) if x is not None)


def read_gate(ctx: RunContext) -> gates.GateReport | None:
    path = ctx.path("evaluation", "gate.json")
    return gate_from_dict(_read_json(path)) if path.is_file() else None

