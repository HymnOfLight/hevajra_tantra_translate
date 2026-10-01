"""Command line: ``hevajra-matrix <command> [options]`` (also ``python -m hevajra_matrix``).

Every command takes ``--root`` (default: the directory above the current one that holds
``config/run.yaml``), ``--run-dir`` (default: a new run directory for ``ingest``, ``run``
and ``claude-check``, otherwise the latest one under ``runs/``) and ``--offline`` (answer
LLM calls from the response cache only). Exit status: 0 success, 1 a check failed,
2 a stage could not run (the message says why and what to run first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from . import pipeline
from .align.external import ExternalAlignmentError
from .config import ConfigError, Settings, find_root, load_settings
from .core.lexicon import LexiconError
from .evaluation.gold import GoldError
from .evaluation.sentinels import SentinelError
from .experiments.overattribution.analysis import AnalysisError
from .experiments.overattribution.design import DesignError
from .llm.client import LLMError
from .pipeline import RunContext, StageError
from .prereg import PreregError, freeze
from .registry import RegistryError
from .review.verdicts import VerdictError
from .topics import TopicError

NEW_RUN_COMMANDS = frozenset({"ingest", "run", "claude-check"})
# Errors whose message is meant for the researcher (a file, a line and what is wrong):
# printed without a traceback. Anything else is a bug and keeps its traceback.
USER_ERRORS = (StageError, PreregError, ConfigError, LLMError, GoldError, VerdictError, TopicError, DesignError,
               AnalysisError, SentinelError, RegistryError, LexiconError, ExternalAlignmentError)


def _common() -> argparse.ArgumentParser:
    """Options accepted before or after the command name."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", type=Path, default=argparse.SUPPRESS,
                   help="repository root (default: found from the current directory)")
    p.add_argument("--run-dir", type=Path, default=argparse.SUPPRESS,
                   help="run directory to use (default: new for ingest/run, else the latest)")
    p.add_argument("--offline", action="store_true", default=argparse.SUPPRESS,
                   help="answer LLM calls from the response cache only; never call the API")
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(prog="hevajra-matrix", parents=[common],
                                     description="Validated, evidence-graded collation of the Hevajratantra "
                                                 "witnesses (Claude proposes, code verifies, humans decide).")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, help_: str, parent: argparse._SubParsersAction = sub) -> argparse.ArgumentParser:
        return parent.add_parser(name, help=help_, description=help_, parents=[common])

    p = add("fetch", "Download the source texts into data/raw and record their sha256.")
    p.add_argument("--raw-dir", type=Path, help="target directory (default: run.yaml paths.raw)")
    p.add_argument("--force", action="store_true", help="download again even if the files exist")

    p = add("ingest", "Parse the raw texts into segments; check G0 and the ingest-stage sentinels.")
    p.add_argument("--raw-dir", type=Path, help="directory with the raw files (default: run.yaml paths.raw)")

    p = add("baselines", "Compute the P1 (length-only) and B0 (anchor) DP baselines; or import one.")
    bsub = p.add_subparsers(dest="action", metavar="<action>")
    q = add("import", "Import an external alignment TSV (e.g. MITRA-E) as a control.", bsub)
    q.add_argument("--tsv", type=Path, required=True, help="alignment TSV with our segment ids")
    q.add_argument("--name", required=True, help="control name, e.g. mitra-e")

    p = add("collate", "Run the Claude collator (T1) with replicates and verify every proposal.")
    p.add_argument("--chapters", help="comma-separated reference chapters, e.g. I.1,II.3 (default: all)")
    p.add_argument("--replicates", type=int, help="number of replicates (default: config/llm.yaml)")
    p.add_argument("--dry-run", action="store_true", help="project the cost and send nothing")

    add("build", "Build the consensus, the P2 placebo and the matrix with gold and verdicts applied.")

    p = add("evaluate", "Score aligners on gold, check sentinels and evaluate the gates G0-G4.")
    p.add_argument("--gold", choices=("dev", "test"), default="test", help="gold set to score (default: test)")
    p.add_argument("--baselines-only", action="store_true", help="score the controls only; no Claude output")

    add("perturb", "Run the P-wrong-window and P-deletion perturbations (gate G2).")
    add("stats", "Compute the estimands (two-phase draws, Manski bounds, E3, E4, E5).")
    add("report", "Write the level-gated summary.md and the SVG figures.")

    p = add("topics", "Topic pre-labels (T3) and the human topic sheets.")
    tsub = p.add_subparsers(dest="action", required=True, metavar="<action>")
    add("prelabel", "Pre-label every reference unit with the topic codebook (reference text only).", tsub)
    q = add("export", "Write the topic sheet (same as review export --task topics).", tsub)
    q.add_argument("--second", action="store_true", help="the blind second-coder sheet (no prelabel columns)")
    q = add("import", "Import a filled topic sheet (same as review import --task topics).", tsub)
    q.add_argument("--file", type=Path, required=True, help="the filled sheet")
    q.add_argument("--annotator", required=True, help="the coder")
    q.add_argument("--date", help="YYYY-MM-DD (default: today)")

    p = add("sample", "Draw test windows, the verification sample or the audit sample.")
    p.add_argument("kind", choices=("windows", "verification", "audit"), help="what to draw")
    p.add_argument("--batch", help="plan name (default: verify or audit)")
    p.add_argument("--force", action="store_true", help="redraw although a draw exists")

    p = add("review", "Export, import, reveal and track human review sheets.")
    rsub = p.add_subparsers(dest="action", required=True, metavar="<action>")
    q = add("export", "Write the sheets of one batch into the run's review/ directory.", rsub)
    q.add_argument("--task", required=True, choices=pipeline.review.EXPORT_TASKS)
    q.add_argument("--batch", help="batch (plan) name")
    q.add_argument("--set", dest="gold_set", help="gold set for --task gold: dev, test or test_second")
    q.add_argument("--hours", type=float, help="fill only this many hours of work, in priority order")
    q.add_argument("--competence", default="bo,zh", help="reviewer languages for --hours (default: bo,zh)")
    q.add_argument("--force", action="store_true", help="overwrite an existing sheet of this batch")
    for action, help_ in (("import", "Import a filled sheet into data/annotations/."),
                          ("reveal", "Import a filled reveal sheet (final decisions after seeing the machine).")):
        q = add(action, help_, rsub)
        if action == "import":
            q.add_argument("--task", required=True, choices=pipeline.review.IMPORT_TASKS)
        q.add_argument("--file", type=Path, required=True, help="the filled sheet")
        q.add_argument("--annotator", default="", help="who filled it")
        q.add_argument("--date", help="YYYY-MM-DD (default: today)")
        q.add_argument("--minutes", type=float, help="time spent, to measure the real review cost")
    add("status", "Print review coverage per stratum, gold rows and topic labels.", rsub)

    add("components", "Code components of selected pairs with T2 (descriptive only).")

    p = add("experiment", "The over-attribution experiment (Claude as subject).")
    esub = p.add_subparsers(dest="name", required=True, metavar="<experiment>")
    q = add("overattribution", "Plan, run or score the over-attribution experiment.", esub)
    q.add_argument("action", choices=("plan", "run", "score"))
    q.add_argument("--phase", choices=("main", "pilot"), default="main",
                   help="item phase; each phase has its own outputs and only main is confirmatory")

    add("claude-check", "Send one tiny live request: key, model and response path work.")

    p = add("prereg", "Freeze the preregistration with the current instrument digests.")
    psub = p.add_subparsers(dest="action", required=True, metavar="<action>")
    q = add("freeze", "Write instrument digests and frozen: true into config/preregistration.yaml.", psub)
    q.add_argument("--amend", help="reason for changing a frozen preregistration (appended to amendments)")

    p = add("run", "Run ingest, baselines, collate, build, evaluate, stats and report as far as possible.")
    p.add_argument("--raw-dir", type=Path, help="directory with the raw files (default: run.yaml paths.raw)")
    p.add_argument("--gold", choices=("dev", "test"), default="test", help="gold set to score (default: test)")
    return parser


# --------------------------------------------------------------------------- dispatch
def _context(args: argparse.Namespace, settings: Settings) -> RunContext:
    run_dir = getattr(args, "run_dir", None)
    if run_dir is None:
        run_dir = pipeline.new_run_dir(settings) if args.command in NEW_RUN_COMMANDS else \
            pipeline.latest_run_dir(settings)
        if run_dir is None:
            raise StageError("no run directory yet: run `hevajra-matrix ingest` first (or pass --run-dir)")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run directory: {run_dir}")
    return RunContext(settings.root, run_dir, settings, offline=bool(getattr(args, "offline", False)))


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    command, action = args.command, getattr(args, "action", None)
    if command == "fetch":
        pipeline.fetch(args.raw_dir or settings.path("raw"), force=args.force)
        return 0
    if command == "prereg":
        doc = freeze(settings.root, amend=args.amend)
        print(f"prereg: frozen with {len(doc['instrument_digests'])} instrument digests; "
              f"{len(doc['amendments'])} amendment(s)")
        return 0
    if command == "claude-check" and getattr(args, "offline", False):
        raise StageError("claude-check is a live check; it cannot run --offline")
    ctx = _context(args, settings)
    stages: dict[str, Callable[[], object]] = {
        "ingest": lambda: pipeline.ingest(ctx, args.raw_dir),
        "collate": lambda: pipeline.collate(ctx, args.chapters, args.replicates, args.dry_run),
        "build": lambda: pipeline.build(ctx),
        "evaluate": lambda: pipeline.evaluate(ctx, args.gold, args.baselines_only),
        "perturb": lambda: pipeline.perturb(ctx),
        "stats": lambda: pipeline.stats(ctx),
        "report": lambda: pipeline.report(ctx),
        "components": lambda: pipeline.components_stage(ctx),
        "sample": lambda: pipeline.sample(ctx, args.kind, args.batch, args.force),
    }
    if command in stages:
        stages[command]()
        return 0
    if command == "baselines":
        if action == "import":
            pipeline.baselines_import(ctx, args.tsv, args.name)
        else:
            pipeline.baselines(ctx)
        return 0
    if command == "topics":
        if action == "prelabel":
            pipeline.topics_prelabel(ctx)
        elif action == "export":
            pipeline.review_export(ctx, "topics_second" if args.second else "topics")
        else:
            second = args.file.name.endswith(".second.csv")
            pipeline.review_import(ctx, "topics_second" if second else "topics", args.file, args.annotator, args.date)
        return 0
    if command == "review":
        return _review(ctx, args)
    if command == "experiment":
        if args.action == "plan":
            pipeline.experiment_plan(ctx, args.phase)
        elif args.action == "run":
            pipeline.experiment_run(ctx, args.phase)
        else:
            pipeline.experiment_score(ctx, phase=args.phase)
        return 0
    if command == "claude-check":
        return 0 if pipeline.claude_check(ctx)["passed"] else 1
    if command == "run":
        pipeline.run(ctx, args.raw_dir, args.gold)
        return 0
    raise AssertionError(f"unhandled command {command}")


def _review(ctx: RunContext, args: argparse.Namespace) -> int:
    if args.action == "export":
        competence = [c.strip() for c in args.competence.split(",") if c.strip()]
        pipeline.review_export(ctx, args.task, args.batch, args.gold_set, args.hours, competence, args.force)
    elif args.action in ("import", "reveal"):
        task = "reveal" if args.action == "reveal" else args.task
        pipeline.review_import(ctx, task, args.file, args.annotator, args.date, args.minutes)
    else:
        pipeline.review_status(ctx)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = getattr(args, "root", None)
        settings = load_settings(Path(root).resolve() if root else find_root())
        return _dispatch(args, settings)
    except USER_ERRORS as exc:
        print(f"hevajra-matrix {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
