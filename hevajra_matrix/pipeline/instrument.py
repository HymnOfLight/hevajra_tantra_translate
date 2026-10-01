"""Stages that call Claude through the composed client: collate (and its dry-run cost
projection), the G2 perturbations, the T2 component coder and the connectivity check."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..collate import collator, components
from ..collate.merge import verify_all
from ..collate.perturb import Rate, deletion_recall, delete_segments, false_link_rate, wrong_window
from ..collate.verify import CheckLexicon, Collation, verify
from ..collate.windows import Window, WindowParams, plan
from ..core.ids import REF_CHAPTERS
from ..core.io import write_csv, write_jsonl
from ..core.types import Alignment
from ..llm.cache import CachedClient
from ..llm.check import build_check_request, run_check
from ..llm.client import CacheMiss, LLMError, LLMRequest
from ..ingest import CONTENT_KINDS
from .context import (
    RunContext,
    StageError,
    Texts,
    anthropic_client,
    audited,
    has_api_key,
    load_texts,
    make_client,
    parse_chapters,
    record_stage,
    require,
)
from .human_data import load_topics
from .store import (
    collation_from_dict,
    collation_to_dict,
    diagnostic_to_dict,
    read_alignment,
    write_alignment,
)

CONSENSUS_FILE = "claude.jsonl"
ASSUMED_OUTPUT_TOKENS = 30_000          # per T1 call; synthesis 3.5 expects 20-40k (thinking included)


# --------------------------------------------------------------------------- collation plan
@dataclass(frozen=True)
class CollationPlan:
    """Everything that determines the T1 requests of a run (pure; no client involved)."""

    windows: tuple[Window, ...]
    settings: collator.TaskSettings
    tags: tuple[str, ...]
    examples: collator.Examples
    template: str

    def requests(self) -> list[tuple[str, Window, LLMRequest]]:
        return collator.build_requests(self.windows, self.settings, self.tags, self.examples, self.template)


def plan_collation(ctx: RunContext, chapters: str | None = None, replicates: int | None = None,
                   texts: Texts | None = None) -> CollationPlan:
    """The windows (all chapters, or ``chapters`` like "I.1,II.3") and replicate tags."""
    texts = texts or load_texts(ctx)
    settings = collator.TaskSettings.from_config(ctx.settings.llm)
    windows = plan(texts.reference, texts.witness, texts.concordance, WindowParams.from_config(ctx.settings.run))
    wanted = parse_chapters(chapters, REF_CHAPTERS)
    if wanted is not None:
        windows = [w for w in windows if w.chapter in wanted]
    k = settings.replicates if replicates is None else replicates
    if k < 1:
        raise StageError("--replicates must be >= 1")
    return CollationPlan(tuple(windows), settings, collator.replicate_tags(k),
                         collator.load_examples(ctx.data_dir), collator.load_template())


def replicate_paths(ctx: RunContext, tag: str) -> tuple[Path, Path]:
    return ctx.path("alignments", f"claude.{tag}.jsonl"), ctx.path("collation", f"{tag}.json")


def read_replicates(ctx: RunContext) -> list[Collation]:
    """The verified replicates written by the latest ``collate`` (see ``collation/replicates.json``)."""
    index = require(ctx.path("collation", "replicates.json"), "collate")
    tags = json.loads(index.read_text(encoding="utf-8"))["tags"]
    out = []
    for tag in tags:
        a_path, c_path = replicate_paths(ctx, tag)
        out.append(collation_from_dict(read_alignment(a_path), json.loads(c_path.read_text(encoding="utf-8"))))
    return out


# --------------------------------------------------------------------------- collate
def collate(ctx: RunContext, chapters: str | None = None, replicates: int | None = None,
            dry_run: bool = False) -> dict[str, Any]:
    """Run T1 for every window and replicate, verify (V1-V11) and write each replicate.

    ``--chapters`` restricts this invocation; ``build`` reads what the latest invocation
    wrote, so assemble the whole text by rerunning without ``--chapters`` (cached windows
    cost nothing). ``dry_run`` sends nothing and prints the projected cost.
    """
    texts = load_texts(ctx)
    cplan = plan_collation(ctx, chapters, replicates, texts)
    if not cplan.windows:
        raise StageError("no window to collate")
    if dry_run:
        return dry_run_cost(ctx, cplan)
    if not ctx.offline and not has_api_key():
        raise StageError("collate needs ANTHROPIC_API_KEY, or --offline to replay the response cache "
                         f"({ctx.settings.path('cache')})")
    client = make_client(ctx)
    workers = int(ctx.settings.llm.get("workers", 1))
    try:
        parsed = collator.collate(cplan.windows, client, cplan.settings, cplan.tags, cplan.examples, cplan.template,
                                  workers=workers)
    except CacheMiss as exc:
        raise StageError(f"{exc}; the cache does not hold this collation (run online once, or check the prompt "
                         "and configuration have not changed)") from exc
    lexicon = CheckLexicon.load(ctx.data_dir, texts.witness)
    summary: dict[str, Any] = {"windows": len(cplan.windows), "tags": list(cplan.tags), "replicates": {}}
    for tag in cplan.tags:
        verified, overlap = verify_all(cplan.windows, parsed[tag], lexicon,
                                       collator.source_name(cplan.settings.model, tag))
        a_path, c_path = replicate_paths(ctx, tag)
        write_alignment(a_path, verified.alignment)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.write_text(json.dumps({**collation_to_dict(verified), "overlap": vars(overlap)}, ensure_ascii=False,
                                     indent=1, sort_keys=True) + "\n", encoding="utf-8")
        summary["replicates"][tag] = {"resolved": len(verified.alignment.by_ref()),
                                      "unresolved": len(verified.unresolved), "overlap_agreement": overlap.rate}
        print(f"collate {tag}: {len(verified.alignment.by_ref())} units resolved, {len(verified.unresolved)} "
              f"UNALIGNED, overlap agreement {overlap.rate if overlap.rate is None else round(overlap.rate, 3)}")
    ctx.path("collation", "replicates.json").write_text(
        json.dumps({"tags": list(cplan.tags), "chapters": sorted({w.chapter for w in cplan.windows}),
                    "windows": [w.key for w in cplan.windows],
                    "request_keys": sorted(r.key() for _, _, r in cplan.requests())}, indent=1) + "\n",
        encoding="utf-8")
    record_stage(ctx, "collate")
    return summary


def estimate_tokens(text: str) -> int:
    """Rough token count when no API key is available: 4 ASCII characters or 1 other
    character per token. Tibetan and Chinese tokenisation is unknown, so this errs high;
    measure chapter I.1 with a key before the first full run (synthesis 3.0)."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return math.ceil(ascii_chars / 4 + (len(text) - ascii_chars))


def project_cost(requests: Sequence[LLMRequest], pricing: Mapping[str, float], count: Callable[[LLMRequest], int],
                 cached: Callable[[LLMRequest], bool], output_tokens: int = ASSUMED_OUTPUT_TOKENS) -> dict[str, Any]:
    """Projected USD for ``requests`` (pure given ``count`` and ``cached``).

    Per uncached call: the static prefix (system + context) is written to the prompt cache
    by the first call of each prefix and read by the others; the body is plain input;
    ``output_tokens`` is an assumption. Replicates differ only in their cache tag, so each
    distinct content is counted once and multiplied.
    """
    counted: dict[str, int] = {}
    seen_prefix: set[tuple[str, str | None]] = set()
    total_in = usd = 0.0
    sent = 0
    for req in requests:
        if cached(req):
            continue
        content = (req.system, req.context, req.body)
        key = "\x1f".join(x or "" for x in content)
        if key not in counted:
            counted[key] = count(req)
        tokens = counted[key]
        prefix_chars = len(req.system) + len(req.context or "")
        prefix = tokens * prefix_chars / max(1, prefix_chars + len(req.body))
        first = (req.system, req.context) not in seen_prefix
        seen_prefix.add((req.system, req.context))
        usd += (prefix * (pricing["cache_write"] if first else pricing["cache_read"])
                + (tokens - prefix) * pricing["input"] + output_tokens * pricing["output"]) / 1e6
        total_in += tokens
        sent += 1
    return {"calls": len(requests), "cached_calls": len(requests) - sent, "calls_to_send": sent,
            "input_tokens": int(total_in), "assumed_output_tokens_per_call": output_tokens,
            "projected_usd": round(usd, 2)}


def dry_run_cost(ctx: RunContext, cplan: CollationPlan) -> dict[str, Any]:
    """Project the cost of ``cplan``; nothing is sent. Token counts come from the API's
    free ``count_tokens`` when a key is present (and not offline), else ``estimate_tokens``."""
    requests = [r for _, _, r in cplan.requests()]
    llm = ctx.settings.llm
    cache = CachedClient(None, ctx.settings.path("cache"), offline=True)   # type: ignore[arg-type]  # lookups only
    method, count = "heuristic", lambda r: estimate_tokens((r.system or "") + (r.context or "") + r.body)
    if has_api_key() and not ctx.offline:
        client = anthropic_client(ctx.settings)
        try:
            client.count_tokens(requests[0])
            method, count = "count_tokens", client.count_tokens
        except LLMError as exc:
            print(f"collate --dry-run: count_tokens failed ({exc}); using the character heuristic")
    result = project_cost(requests, dict(llm.get("pricing_usd_per_mtok") or {}), count, cache.is_cached)
    result.update({"token_count_method": method, "windows": len(cplan.windows), "replicates": len(cplan.tags),
                   "budget_usd": llm.get("budget_usd")})
    path = ctx.path("collation", "dry_run.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"collate --dry-run: {result['calls']} calls ({result['cached_calls']} cached), "
          f"{result['input_tokens']} input tokens ({method}), assumed {result['assumed_output_tokens_per_call']} "
          f"output tokens per call -> projected ${result['projected_usd']} (budget ${llm.get('budget_usd')}); "
          "nothing was sent")
    return result


# --------------------------------------------------------------------------- perturbations (G2)
def dev_chapters(prereg: Mapping[str, Any]) -> list[str]:
    regions = (prereg.get("gold") or {}).get("dev_regions") or []
    return list(dict.fromkeys(str(r["chapter"]) for r in regions if isinstance(r, Mapping) and "chapter" in r))


def far_chapter(window: Window, order: Sequence[str]) -> str:
    """The witness chapter farthest (in chapter order) from the window's core chapters."""
    core = [order.index(c) for c in window.core_locals if c in order]
    present = {s.local_chapter for s in window.text}
    candidates = [c for c in order if c in present and c not in window.core_locals]
    if not core or not candidates:
        raise StageError(f"window {window.key}: no witness chapter outside its core window")
    return max(candidates, key=lambda c: (min(abs(order.index(c) - i) for i in core), -order.index(c)))


def perturb(ctx: RunContext, fraction: float = 0.05, seed: int | None = None) -> dict[str, Any]:
    """P-wrong-window and P-deletion (G2) on the windows of the dev-gold chapters.

    Expected counterparts for P-deletion come from this run's consensus
    (``alignments/claude.jsonl``). P-negation needs negated gold pairs and is not run here.
    """
    texts = load_texts(ctx)
    expected = read_alignment(require(ctx.path("alignments", CONSENSUS_FILE), "build"))
    seed = int((ctx.settings.prereg.get("stats") or {}).get("seed", 0)) if seed is None else seed
    base = plan_collation(ctx, ",".join(dev_chapters(ctx.settings.prereg)) or None, 1, texts)
    order = texts.concordance.local_order[texts.witness_id]
    wrong = [wrong_window(w, [far_chapter(w, order)]) for w in base.windows]
    deleted = [delete_segments(w, fraction, seed) for w in base.windows
               if any(s.id in w.core and s.kind in CONTENT_KINDS for s in w.text)]
    windows = [w for w, _ in wrong] + [w for w, _ in deleted]
    if not ctx.offline and not has_api_key():
        raise StageError("perturb needs ANTHROPIC_API_KEY, or --offline with a filled cache")
    parsed = collator.collate(windows, make_client(ctx), base.settings, ("r1",), base.examples, base.template,
                              workers=int(ctx.settings.llm.get("workers", 1)))["r1"]
    lexicon = CheckLexicon.load(ctx.data_dir, texts.witness)
    results = [verify(p, w, lexicon, collator.source_name(base.settings.model, "perturb"))
               for w, p in zip(windows, parsed)]
    ww = _sum(false_link_rate(t, r.alignment) for (_, t), r in zip(wrong, results[:len(wrong)]))
    dl = _sum(deletion_recall(t, expected, r.alignment) for (_, t), r in zip(deleted, results[len(wrong):]))
    out = {"wrong_window": vars(ww), "deletion": vars(dl), "negation": None, "fraction": fraction, "seed": seed,
           "windows": [w.key for w in windows]}
    path = ctx.path("evaluation", "perturbations.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    record_stage(ctx, "perturb")
    print(f"perturb: wrong-window false links {ww.hits}/{ww.n}; deletion recall {dl.hits}/{dl.n}")
    return out


def _sum(rates: Any) -> Rate:
    hits = n = 0
    for r in rates:
        hits, n = hits + r.hits, n + r.n
    return Rate(hits, n)


# --------------------------------------------------------------------------- components (T2)
def components_stage(ctx: RunContext, sample_fraction: float = components.DEFAULT_SAMPLE_FRACTION) -> dict[str, int]:
    """T2 on the consensus: every deviation, every sensitive unit and a sample of
    equivalent links. Descriptive only; never an input to E1-E4."""
    texts = load_texts(ctx)
    consensus = read_alignment(require(ctx.path("alignments", CONSENSUS_FILE), "build"))
    topics = load_topics(ctx, texts).topic_sets()
    seed = int((ctx.settings.prereg.get("stats") or {}).get("seed", 0))
    pairs = components.select_pairs(consensus, topics, random.Random(seed), ref_kinds=texts.ref_kinds,
                                    sample_fraction=sample_fraction)
    if not ctx.offline and not has_api_key():
        raise StageError("components needs ANTHROPIC_API_KEY, or --offline with a filled cache")
    settings = components.ComponentTaskSettings.from_config(ctx.settings.llm)
    segments = {s.id: s for s in (*texts.reference, *texts.witness)}
    codes, diagnostics = components.code_components(
        pairs, segments, make_client(ctx), settings, components.load_system(ctx.data_dir),
        CheckLexicon.load(ctx.data_dir, texts.witness), workers=int(ctx.settings.llm.get("workers", 1)))
    out = ctx.path("components")
    write_jsonl(out / "components.jsonl", (components.code_record(c) for c in codes))
    write_jsonl(out / "diagnostics.jsonl", (diagnostic_to_dict(d) for d in diagnostics))
    profile = components.rendering_profile(codes, texts.reference[0].lang, texts.witness[0].lang)
    write_csv(out / "rendering_profile.csv", [r.csv_row() for r in profile], components.PROFILE_COLUMNS)
    inventing, coded = components.invention_rate(pairs, codes)
    record_stage(ctx, "components")
    print(f"components: {len(pairs)} pairs, {len(codes)} slot codes, {len(diagnostics)} diagnostics; "
          f"equivalent sample with invented deviations {inventing}/{coded} (descriptive only)")
    return {"pairs": len(pairs), "codes": len(codes), "diagnostics": len(diagnostics)}


# --------------------------------------------------------------------------- claude-check
def claude_check(ctx: RunContext) -> dict[str, Any]:
    """One tiny live call (never cached): key, model and response path work."""
    if ctx.offline:
        raise StageError("claude-check is a live check; it cannot run --offline")
    if not has_api_key():
        raise StageError("claude-check needs ANTHROPIC_API_KEY")
    llm = ctx.settings.llm
    task = (llm.get("tasks") or {}).get("check") or {}
    request = build_check_request(model=str(llm.get("model")), effort=task.get("effort", "low"),
                                  max_tokens=int(task.get("max_tokens", 1024)))
    summary = run_check(audited(ctx, anthropic_client(ctx.settings)), request)
    ctx.path("claude_check.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return summary


def read_consensus(ctx: RunContext) -> Alignment | None:
    path = ctx.path("alignments", CONSENSUS_FILE)
    return read_alignment(path) if path.is_file() else None

