"""Claude-free stages that turn alignments into cells and scores: build and evaluate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..align.placebo import SOURCE_SHUFFLED, shuffle_alignment
from ..collate.consensus import SOURCE_CONSENSUS, agreement, consensus
from ..collate.perturb import Rate
from ..core.io import git_commit, read_jsonl, write_jsonl
from ..core.types import REASON_UNASSESSED, REASON_VERIFICATION_FAILED, Alignment, Cell, Status
from ..evaluation import gate as gates
from ..evaluation import gold as gold_sets
from ..evaluation import sentinels as sentinel_checks
from ..matrix.build import build_cells
from ..matrix.export import write_matrix
from ..review import sampling
from ..review.verdicts import is_orphan
from ..stats import decompose, twophase
from ..topics import human_agreement
from . import human_data as ann
from .context import AUDIT_LOG, RunContext, StageError, Texts, load_texts, record_stage, require
from .instrument import CONSENSUS_FILE, read_replicates
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
    on test gold appends one record to the evaluation ledger, and is refused while the
    preregistration is not frozen (the protocol freezes before the test set is scored).
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
    if (gold_set == "test" and claude is not None and ctx.settings.prereg.get("frozen") is not True
            and ann.load_gold(ctx, texts, "test") is not None):
        raise StageError("test gold is scored against Claude only after `hevajra-matrix prereg freeze`; "
                         "use --gold dev or --baselines-only until then")
    scores, human_kappa = _scores(ctx, texts, aligners, gold_set, topics.groups())
    integrity = _integrity(ctx, texts, claude, cells)
    digest = _collate_digest(ctx)
    prereg_sha = ctx.settings.shas.get("preregistration.yaml", "")
    ledger = ann.ledger_path(ctx)
    specs = gates.GateSpecs.from_prereg(ctx.settings.prereg)
    if gold_set == "test" and SOURCE_CONSENSUS in scores:
        g1 = gates.check_g1(scores, human_kappa, specs.g1, specs.seed)
        gates.ledger_append(ledger, gates.ledger_record(digest, prereg_sha, git_commit(ctx.root),
                                                        {k: v.metrics() for k, v in scores.items()}, g1.passed))
    report = gates.evaluate(scores, human_kappa, integrity, _perturbations(ctx), _calibration(ctx, texts, cells, topics),
                            ctx.settings.prereg, digest, gates.ledger_summary(ledger, digest, prereg_sha))
    out = ctx.path("evaluation")
    _write_json(out / "scores.json", {
        "gold_set": gold_set, "baselines_only": baselines_only, "human_kappa": human_kappa,
        "sources": {src: {"counts": s.counts(), "metrics": s.metrics(),
                          "intervals": {k: estimate_to_dict(e) for k, e in
                                        gold_sets.interval_estimates(s, specs.g1.n_boot, specs.seed).items()}}
                    for src, s in scores.items()}})
    _write_json(out / "gate.json", gate_to_dict(report))
    write_jsonl(out / "sentinels.jsonl", (sentinel_result_to_dict(r) for rs in integrity.sentinels.values() for r in rs))
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


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _collate_digest(ctx: RunContext) -> str:
    from ..prereg import instrument_digests      # local import: prereg imports every task module
    return instrument_digests(ctx.settings)["collate"]


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
    everything = [*texts.reference, *texts.witness]
    loaded = sentinel_checks.load(ctx.data_dir / SENTINELS_FILE)
    results = {"ingest": sentinel_checks.check(loaded, "ingest", everything)}
    if claude is not None:
        results["proposal"] = sentinel_checks.check(loaded, "proposal", everything, alignment=claude)
    if cells is not None:
        results["final"] = sentinel_checks.check(loaded, "final", everything, cells=cells)
    collation = _read_json(ctx.path("collation", "integrity.json")) if claude is not None else {}
    audit = [r for r in read_jsonl(ctx.path(AUDIT_LOG))] if ctx.path(AUDIT_LOG).is_file() else []
    measured = [r for r in audit if r.get("task") == "collate"]
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
    coverage = sampling.coverage(ann.review_plans(ctx), verdicts)
    planned = {s: c["planned"] for s, c in coverage.items()}
    achieved = {s: c["final"] for s, c in coverage.items()}
    audit = {s: n for s, n in achieved.items() if s.startswith("neg:")}
    absent = [v for v in verdicts if v.stratum.startswith("pos:absent") and v.blind_relation]
    null_precision = sum(1 for v in absent if v.blind_relation == "no_counterpart") / len(absent) if absent else None
    units = [c for c in cells or [] if not is_orphan(c.unit_id)]
    unresolved = sum(1 for c in units if c.status is Status.UNALIGNED) if units else None
    width = None
    if units:
        lo, hi = twophase.manski(units, ctx.settings.prereg.get("primary_outcome", "any"), texts.ref_kinds)
        width = hi - lo
    e3 = decompose.not_estimable_reason((), texts.reference_id, texts.reference_id)
    agreement_topics = human_agreement(topics.labels)
    scorer = _read_json(ctx.path("experiments", "overattribution", "results.json")).get("scorer") or {}
    return gates.Calibration(
        planned=planned, achieved=achieved, audit=audit, null_precision=null_precision, null_recall=None,
        unresolved_units=unresolved, manski_width=width,
        e3_not_estimable=e3.removeprefix("G4: ") if e3 else None, topic_labels_complete=topics.complete,
        topic_kappa=agreement_topics.group_kappa, mde=(ctx.settings.prereg.get("stats") or {}).get("mde"),
        scorer_kappa=scorer.get("kappa_y_over"))


def read_gate(ctx: RunContext) -> gates.GateReport | None:
    path = ctx.path("evaluation", "gate.json")
    return gate_from_dict(_read_json(path)) if path.is_file() else None

