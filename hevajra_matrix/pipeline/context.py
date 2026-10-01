"""The run context, run directories, the composed LLM client and the shared text loader.

A run directory ``runs/<UTC timestamp>-<git short hash>/`` holds everything one run
produced; every stage reads its predecessors' files there and writes its own (layout in
``pipeline/__init__.py``). ``RunContext`` is the only thing a stage receives.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import Settings
from ..core.io import MANIFEST_NAME, git_commit, read_jsonl, write_manifest
from ..core.types import Segment
from ..ingest import CONTENT_KINDS
from ..llm.anthropic_client import AnthropicClient
from ..llm.audit import AuditedClient
from ..llm.cache import CachedClient
from ..llm.client import CacheMiss, LLMClient, LLMRequest, LLMResponse
from ..registry import Concordance, load_concordance
from .store import read_segments

RUN_NAME = re.compile(r"^\d{8}T\d{6}Z-[0-9a-z]+(-\d+)?$")
AUDIT_LOG = "llm_audit.jsonl"
SPEND_LEDGER = "llm_spend.jsonl"     # beside the shared response cache: the budget is cumulative
CONCORDANCE_FILE = Path("registry") / "concordance.yaml"


class StageError(RuntimeError):
    """A stage cannot run with what the run directory and the configuration hold.

    The message says what is missing and which command produces it; the CLI prints it
    and exits with status 2.
    """


@dataclass(frozen=True)
class RunContext:
    """What every stage receives: the repository root, the run directory, the settings,
    and whether LLM calls must be answered from the response cache only."""

    root: Path
    run_dir: Path
    settings: Settings
    offline: bool = False

    def path(self, *parts: str) -> Path:
        """A path inside the run directory."""
        return self.run_dir.joinpath(*parts)

    @property
    def data_dir(self) -> Path:
        return self.settings.path("data")

    @property
    def reference(self) -> str:
        return str(self.settings.run["witnesses"]["reference"])

    @property
    def witness(self) -> str:
        return str(self.settings.run["witnesses"]["target"])


# --------------------------------------------------------------------------- run directories
def new_run_dir(settings: Settings, now: datetime | None = None) -> Path:
    """A fresh ``runs/<UTC timestamp>-<git short hash>`` directory (created)."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    commit = (git_commit(settings.root) or "nogit")[:7]
    base = settings.path("runs") / f"{stamp}-{commit}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.mkdir(parents=True)
    return path


def latest_run_dir(settings: Settings) -> Path | None:
    """The most recent run directory (names sort by time), or None."""
    runs = settings.path("runs")
    if not runs.is_dir():
        return None
    names = sorted(p.name for p in runs.iterdir() if p.is_dir() and RUN_NAME.match(p.name))
    return runs / names[-1] if names else None


# --------------------------------------------------------------------------- LLM client
class _NoNetwork:
    """Inner client of an offline cache: it is never reached, because the offline cache
    raises ``CacheMiss`` first. Having it keeps the SDK client from ever being built."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise CacheMiss(f"offline: no network client for task {request.task!r}")


def anthropic_client(settings: Settings) -> AnthropicClient:
    """The SDK-backed client configured by ``config/llm.yaml`` (the SDK itself loads lazily)."""
    llm = settings.llm
    return AnthropicClient(model=str(llm.get("model")), max_retries=int(llm.get("max_retries", 4)),
                           timeout_s=float(llm.get("timeout_s", 900)))


def audited(ctx: RunContext, inner: LLMClient) -> AuditedClient:
    """``inner`` wrapped in the run's audit log and the configured spending cap.

    The cap is checked against the spend ledger ``<paths.cache>/llm_spend.jsonl``, which
    every run appends to, so starting a new run directory does not reset it.
    """
    llm = ctx.settings.llm
    fallback = llm.get("fallback_pricing_usd_per_mtok")
    return AuditedClient(inner, ctx.path(AUDIT_LOG), pricing=dict(llm.get("pricing_usd_per_mtok") or {}),
                         budget_usd=float(llm.get("budget_usd", 0)), run_id=ctx.run_dir.name,
                         ledger_path=ctx.settings.path("cache") / SPEND_LEDGER,
                         fallback_pricing=dict(fallback) if fallback else None)


def make_client(ctx: RunContext) -> AuditedClient:
    """``AuditedClient(CachedClient(AnthropicClient(...)))`` from ``config/llm.yaml``.

    The cache lives at ``run.yaml: paths.cache`` and is shared by all runs; the audit log
    is ``<run_dir>/llm_audit.jsonl``. Offline, the cache raises ``CacheMiss`` for anything
    not stored and no Anthropic client is constructed at all.
    """
    inner: LLMClient = _NoNetwork() if ctx.offline else anthropic_client(ctx.settings)
    return audited(ctx, CachedClient(inner, ctx.settings.path("cache"), offline=ctx.offline))


CREDENTIAL_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")   # what the SDK reads from the environment


def has_api_key() -> bool:
    """Whether live calls have credentials: an API key or a bearer auth token (the SDK takes
    either). Other SDK sources (profiles, workload identity) are not detected here."""
    return any(os.environ.get(name) for name in CREDENTIAL_VARIABLES)


def audit_summary(log_path: Path) -> dict[str, Any]:
    """Served models, cache hits and misses and estimated spend from an audit log."""
    records = read_jsonl(log_path) if log_path.is_file() else []
    served: dict[str, set[str]] = {}
    for r in records:
        served.setdefault(str(r.get("task")), set()).add(str(r.get("served_model")))
    return {
        "calls": len(records),
        "cache_hits": sum(1 for r in records if r.get("from_cache")),
        "cache_misses": sum(1 for r in records if not r.get("from_cache")),
        "served_models": {task: sorted(models) for task, models in sorted(served.items())},
        "substituted_calls": dict(Counter(str(r.get("task")) for r in records
                                          if r.get("fallback_used") or r.get("served_model") != r.get("requested_model"))),
        "est_usd_uncached": round(sum(float(r.get("est_usd") or 0) for r in records if not r.get("from_cache")), 4),
    }


# --------------------------------------------------------------------------- manifest
def record_stage(ctx: RunContext, stage: str, inputs: Mapping[str, Path] | None = None) -> Path:
    """Rewrite ``manifest.json`` after ``stage``: inputs so far, stages so far, instrument
    digests, and the LLM audit summary (served models, cache hits and misses)."""
    from ..prereg import instrument_digests      # local import: prereg imports the task modules

    old: dict[str, Any] = {}
    path = ctx.path(MANIFEST_NAME)
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
    known = {label: Path(rec["path"]) for label, rec in (old.get("inputs") or {}).items()}
    known.update(inputs or {})
    present = {label: p for label, p in known.items() if p.is_file()}
    stages = [*[s for s in old.get("stages") or [] if s != stage], stage]
    extra = {"run_id": ctx.run_dir.name, "offline": ctx.offline, "stages": stages,
             "instrument_digests": instrument_digests(ctx.settings), "llm": audit_summary(ctx.path(AUDIT_LOG))}
    return write_manifest(ctx.run_dir, ctx.settings, present, extra)


# --------------------------------------------------------------------------- texts
@dataclass(frozen=True)
class Texts:
    """The ingested texts of a run and the concordance that maps their chapters.

    ``reference`` and ``witness`` are whole texts (every kind, document order); reference
    segments carry their reference chapter (``Concordance.assign_reference_chapters``).
    """

    reference: tuple[Segment, ...]
    witness: tuple[Segment, ...]
    concordance: Concordance

    @cached_property
    def units(self) -> tuple[Segment, ...]:
        """Alignable reference units (the matrix rows), in reference order."""
        return tuple(s for s in self.reference if s.kind in CONTENT_KINDS and s.chapter)

    @cached_property
    def witness_content(self) -> tuple[Segment, ...]:
        return tuple(s for s in self.witness if s.kind in CONTENT_KINDS)

    @cached_property
    def ref_kinds(self) -> dict[str, str]:
        return {s.id: s.kind for s in self.units}

    @cached_property
    def wit_kinds(self) -> dict[str, str]:
        return {s.id: s.kind for s in self.witness}

    @cached_property
    def chapter_of(self) -> dict[str, str]:
        return {s.id: str(s.chapter) for s in self.units}

    @property
    def reference_id(self) -> str:
        return self.reference[0].witness

    @property
    def witness_id(self) -> str:
        return self.witness[0].witness

    def groups(self) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Chapter groups (reference chapters, witness local chapter) from the concordance:
        every witness chapter with the reference chapters it maps to (pin11 = I.11 + II.1)."""
        wit = self.witness_id
        return [(tuple(refs), (local,)) for local in self.concordance.local_order[wit]
                if (refs := self.concordance.refs_for_local(wit, local))]

    def units_by_chapter(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.units:
            out.setdefault(str(s.chapter), []).append(s.id)
        return out


def segments_file(ctx: RunContext, witness: str) -> Path:
    return ctx.path("ingest", f"segments_{witness}.jsonl")


def load_texts(ctx: RunContext) -> Texts:
    """The texts written by ``ingest`` into this run directory."""
    paths = [segments_file(ctx, w) for w in (ctx.reference, ctx.witness)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise StageError(f"{missing[0]} is missing: run `hevajra-matrix ingest` for this run directory first")
    reference, witness = (tuple(read_segments(p)) for p in paths)
    if not reference or not witness:
        raise StageError("an ingested text is empty; rerun ingest")
    return Texts(reference, witness, load_concordance(ctx.data_dir / CONCORDANCE_FILE))


def require(path: Path, producer: str) -> Path:
    """``path`` if it exists, else a ``StageError`` naming the command that writes it."""
    if not path.is_file():
        raise StageError(f"{path} is missing: run `hevajra-matrix {producer}` first")
    return path


def parse_chapters(value: str | None, known: Sequence[str]) -> tuple[str, ...] | None:
    """``"I.1,II.3"`` -> ("I.1", "II.3"); None means every chapter."""
    if not value:
        return None
    chapters = tuple(c.strip() for c in value.split(",") if c.strip())
    unknown = [c for c in chapters if c not in known]
    if unknown:
        raise StageError(f"unknown chapter(s) {unknown}; known: {', '.join(known)}")
    return chapters
