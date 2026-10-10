"""T1 collator: build the request for a window, run requests, parse answers.

Request layout (static first, so the prefix is cached across windows and replicates)

    system   the template of the configured prompt version followed by its three
             synthetic examples and, for collate.v2, the STANDING RENDERINGS block of
             verified glossary entries (``collate.prompt``); cache breakpoint
    context  the full witness text, one ``z0001<TAB>kind<TAB>text`` line per segment;
             identical for every window over the same text; cache breakpoint
    body     the reference chunk as ``r001<TAB>kind<TAB>text`` lines and the core-window
             handle ranges

The request carries no coordinates, chapter keys, topic labels, footnotes or Sanskrit
readings: it is rendered from ``Window`` alone, which holds only segments of one reference
and one witness (see ``collate.windows``), and only their kind and text are printed.

``prompt_sha`` is the sha256 of the system prompt (template, examples and glossary), so
editing any of them changes every cache key and the instrument digest. The schema is the
version's (``schema_for``), and an answer is validated against the schema that was sent.

Parsing never raises on a model answer: a refusal, a truncation, a schema-invalid answer
and an answer served by a substituted model all become an ``Unresolved`` value whose
reason makes the window's units UNALIGNED (researcher decision 4: a substituted model's
parsed proposal is kept as a reviewer hint only).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..config import ConfigError, from_mapping
from ..core.types import (
    REASON_INVALID,
    REASON_SUBSTITUTED_MODEL,
    REASON_TRUNCATED,
    Convention,
    Relation,
    WitnessOnlyKind,
    failure_reason,
)
from ..llm.client import DEFAULT_MODEL, LLMClient, LLMRequest, LLMResponse, sha256_text
from ..llm.schema import validate
from .prompt import (  # noqa: F401  (re-exported: the collator's public API predates the split)
    CONFIDENCE,
    DEFAULT_PAIR,
    LANG_NAMES,
    N_EXAMPLES,
    PROMPT_V1,
    PROMPT_V2,
    SCHEMA_V1,
    SCHEMA_V2,
    VERSIONS,
    Examples,
    PromptVersion,
    example_pairs,
    examples_file,
    glossary_file,
    load_examples,
    load_glossary,
    load_template,
    render_body,
    render_context,
    render_glossary,
    schema_for,
    system_prompt,
)
from .windows import Window

TASK = "collate"                      # the same for every prompt version (cache keys, audit)
PROMPT_ID = PROMPT_V1                 # v1 aliases, kept for callers of the single-version API
TEMPLATE_FILE = VERSIONS[PROMPT_V1].template_file
EXAMPLES_FILE = examples_file(*DEFAULT_PAIR, PROMPT_V1)
SCHEMA = SCHEMA_V1
EFFORTS = ("low", "medium", "high", "xhigh", "max")


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class TaskSettings:
    """``config/llm.yaml: tasks.collate`` plus the top-level ``model``.

    ``prompt`` selects the prompt version (``VERSIONS``); it is not an instrument
    parameter of its own, because the prompt and schema shas already capture it.
    ``max_tokens`` defaults to 128,000: only generated tokens are billed, and a higher
    cap only lowers the truncation (hence UNALIGNED) rate.
    """

    effort: str = "high"
    max_tokens: int = 128_000
    replicates: int = 3
    fallback: bool = False
    model: str = DEFAULT_MODEL
    prompt: str = PROMPT_V1

    def __post_init__(self) -> None:
        if self.prompt not in VERSIONS:
            raise ConfigError(f"tasks.collate.prompt must be one of {tuple(VERSIONS)}, got {self.prompt!r}")
        if self.effort not in EFFORTS:
            raise ConfigError(f"tasks.collate.effort must be one of {EFFORTS}, got {self.effort!r}")
        for name in ("max_tokens", "replicates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"tasks.collate.{name} must be a positive integer, got {value!r}")
        if not isinstance(self.fallback, bool):
            raise ConfigError("tasks.collate.fallback must be true or false")

    @classmethod
    def from_config(cls, llm: Mapping[str, Any]) -> "TaskSettings":
        task = dict((llm.get("tasks") or {}).get(TASK) or {})
        if "model" in task:
            raise ConfigError("llm.yaml: tasks.collate takes the model from the top-level 'model' key")
        return from_mapping(cls, {**task, "model": llm.get("model", DEFAULT_MODEL)}, "llm.yaml: tasks.collate")


def replicate_tags(k: int) -> tuple[str, ...]:
    """Cache tags of k replicates: ("r1", ..., "rk")."""
    return tuple(f"r{i}" for i in range(1, k + 1))


def source_name(model: str, replicate: str, prompt: str = PROMPT_V1) -> str:
    """``Link.source`` of one replicate, e.g. "claude-opus-5-5:collate.v1:r2"."""
    return f"{model}:{prompt}:{replicate}"


# --------------------------------------------------------------------------- requests
def build_request(window: Window, settings: TaskSettings, replicate: str, examples: Examples,
                  template: str) -> LLMRequest:
    """The T1 request for one window and one replicate tag (pure)."""
    if (window.ref_lang, window.wit_lang) != (examples.reference_lang, examples.witness_lang):
        raise ValueError(f"window {window.key} is {window.ref_lang}->{window.wit_lang} but the examples "
                         f"illustrate {examples.reference_lang}->{examples.witness_lang}")
    if settings.prompt != examples.prompt:
        raise ValueError(f"the settings select {settings.prompt} but the examples are those of {examples.prompt}")
    system = system_prompt(template, examples)
    context = render_context(
        [(h, window.segment[sid].kind, window.segment[sid].text) for h, sid in window.wit_handles.items()],
        window.wit_lang)
    body = render_body(
        [(h, window.segment[uid].kind, window.segment[uid].text) for h, uid in window.ref_handles.items()],
        window.ref_lang, window.core_ranges())
    return LLMRequest(
        task=TASK, prompt_sha=sha256_text(system), system=system, context=context, body=body,
        schema=schema_for(examples.prompt),
        effort=settings.effort, max_tokens=settings.max_tokens, replicate=replicate,
        allow_fallback=settings.fallback, model=settings.model,
    )


# --------------------------------------------------------------------------- parsing
@dataclass(frozen=True)
class RawUnit:
    """One unit record as the model returned it (handles, not ids); a v1 answer has no
    convention and parses as NONE."""

    ref: str
    wit: tuple[str, ...]
    relation: Relation
    polarity_flip: bool
    confidence: str
    ref_quote: str
    wit_quote: str
    convention: Convention = Convention.NONE


@dataclass(frozen=True)
class RawWitnessOnly:
    wit: tuple[str, ...]
    kind: WitnessOnlyKind
    wit_quote: str


@dataclass(frozen=True)
class RawCollation:
    """A schema-valid answer for one window, not yet verified."""

    window: str
    units: tuple[RawUnit, ...]
    witness_only: tuple[RawWitnessOnly, ...]


@dataclass(frozen=True)
class Unresolved:
    """A window without a usable answer; every unit of it becomes UNALIGNED(reason).

    ``hint`` is the parsed proposal of a substituted model: shown to reviewers, never
    measured.
    """

    window: str
    reason: str
    hint: RawCollation | None = None


Parsed = RawCollation | Unresolved


def parse(response: LLMResponse, window: Window, schema: Mapping[str, Any] = SCHEMA_V1) -> Parsed:
    """Turn a response into a raw collation or the reason it cannot be used. ``schema`` is
    the one the request carried (an answer valid only under another version's schema is
    invalid)."""
    if response.substituted_model:
        hint = _raw(response.data, window, schema) if response.status == "ok" else None
        return Unresolved(window.key, REASON_SUBSTITUTED_MODEL, hint if isinstance(hint, RawCollation) else None)
    if response.status == "refusal":
        return Unresolved(window.key, failure_reason(response.status, response.refusal_category))
    if response.status == "truncated":
        return Unresolved(window.key, REASON_TRUNCATED)
    if response.status != "ok":
        return Unresolved(window.key, REASON_INVALID)
    return _raw(response.data, window, schema)


def _raw(data: Mapping[str, Any] | None, window: Window, schema: Mapping[str, Any]) -> Parsed:
    if data is None or validate(data, schema):
        return Unresolved(window.key, REASON_INVALID)
    units = tuple(
        RawUnit(ref=u["ref"].strip(), wit=tuple(h.strip() for h in u["wit"]), relation=Relation(u["relation"]),
                polarity_flip=u["polarity_flip"], confidence=u["confidence"], ref_quote=u["ref_quote"],
                wit_quote=u["wit_quote"], convention=Convention(u.get("convention", Convention.NONE.value)))
        for u in data["units"])
    witness_only = tuple(
        RawWitnessOnly(wit=tuple(h.strip() for h in w["wit"]), kind=WitnessOnlyKind(w["kind"]),
                       wit_quote=w["wit_quote"])
        for w in data["witness_only"])
    return RawCollation(window.key, units, witness_only)


# --------------------------------------------------------------------------- running
def build_requests(windows: Sequence[Window], settings: TaskSettings, replicates: Sequence[str],
                   examples: Examples, template: str) -> list[tuple[str, Window, LLMRequest]]:
    """Every (replicate, window) request, replicate-major (for dry runs and cost projection)."""
    return [(tag, w, build_request(w, settings, tag, examples, template)) for tag in replicates for w in windows]


def collate(windows: Sequence[Window], client: LLMClient, settings: TaskSettings, replicates: Sequence[str],
            examples: Examples, template: str, workers: int = 1) -> dict[str, tuple[Parsed, ...]]:
    """Run every request and parse the answers.

    Returns replicate tag -> one parse result per window, in the order of ``windows``,
    whatever order the thread pool finished in. Requests are independent, so ``workers``
    of them run concurrently, except that the first request not yet in the response cache
    runs alone: every request shares the same cacheable prefix (instructions and the full
    witness text), and the others can only read the prompt cache once one has written it.
    Transport, budget and cache-miss errors propagate after the requests not yet started
    are cancelled; answers already received stay cached.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    jobs = build_requests(windows, settings, replicates, examples, template)
    is_cached = getattr(client, "is_cached", lambda request: False)
    first = next((i for i, (_, _, request) in enumerate(jobs) if not is_cached(request)), 0)
    responses: list[LLMResponse | None] = [None] * len(jobs)
    if jobs:
        responses[first] = client.complete(jobs[first][2])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {i: pool.submit(client.complete, request) for i, (_, _, request) in enumerate(jobs) if i != first}
        try:
            for i, future in futures.items():
                responses[i] = future.result()
        except BaseException:
            for future in futures.values():
                future.cancel()
            raise
    out: dict[str, list[Parsed]] = {tag: [] for tag in replicates}
    for (tag, window, request), response in zip(jobs, responses):
        out[tag].append(parse(response, window, request.schema))
    return {tag: tuple(results) for tag, results in out.items()}
