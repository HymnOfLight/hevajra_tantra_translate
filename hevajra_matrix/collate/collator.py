"""T1 collator: build the request for a window, run requests, parse answers.

Request layout (static first, so the prefix is cached across windows and replicates)

    system   the template ``prompts/collate.v1.md`` followed by the three synthetic
             examples of ``data/codebook/collate_examples.yaml``; cache breakpoint
    context  the full witness text, one ``z0001<TAB>kind<TAB>text`` line per segment;
             identical for every window over the same text; cache breakpoint
    body     the reference chunk as ``r001<TAB>kind<TAB>text`` lines and the core-window
             handle ranges

The request carries no coordinates, chapter keys, topic labels, footnotes or Sanskrit
readings: it is rendered from ``Window`` alone, which holds only segments of one reference
and one witness (see ``collate.windows``), and only their kind and text are printed.

``prompt_sha`` is the sha256 of the system prompt (template plus examples), so editing
either changes every cache key and the instrument digest.

Parsing never raises on a model answer: a refusal, a truncation, a schema-invalid answer
and an answer served by a substituted model all become an ``Unresolved`` value whose
reason makes the window's units UNALIGNED (researcher decision 4: a substituted model's
parsed proposal is kept as a reviewer hint only).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import ConfigError, from_mapping
from ..core.io import read_yaml
from ..core.types import (
    REASON_INVALID,
    REASON_REFUSED,
    REASON_SUBSTITUTED_MODEL,
    REASON_TRUNCATED,
    Relation,
    WitnessOnlyKind,
)
from ..llm.client import DEFAULT_MODEL, LLMClient, LLMRequest, LLMResponse, canonical_json, sha256_text
from ..llm.schema import strict, validate
from .windows import Window, prompt_kind

TASK = "collate"
PROMPT_ID = "collate.v1"
TEMPLATE_FILE = "collate.v1.md"
EXAMPLES_FILE = Path("codebook") / "collate_examples.yaml"
N_EXAMPLES = 3
EFFORTS = ("low", "medium", "high", "xhigh", "max")
CONFIDENCE = ("high", "medium", "low")
LANG_NAMES = {"bo": "Tibetan", "zh": "Chinese", "sa": "Sanskrit", "en": "English"}
_LAYOUT_RE = re.compile(r"[\t\r\n]+")

SCHEMA: Mapping[str, Any] = strict({
    "type": "object",
    "properties": {
        "units": {"type": "array", "items": {"type": "object", "properties": {
            "ref": {"type": "string"},
            "wit": {"type": "array", "items": {"type": "string"}},
            "relation": {"type": "string", "enum": [r.value for r in Relation]},
            "polarity_flip": {"type": "boolean"},
            "confidence": {"type": "string", "enum": list(CONFIDENCE)},
            "ref_quote": {"type": "string"},
            "wit_quote": {"type": "string"},
        }}},
        "witness_only": {"type": "array", "items": {"type": "object", "properties": {
            "wit": {"type": "array", "items": {"type": "string"}},
            "kind": {"type": "string", "enum": [k.value for k in WitnessOnlyKind]},
            "wit_quote": {"type": "string"},
        }}},
    },
})


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class TaskSettings:
    """``config/llm.yaml: tasks.collate`` plus the top-level ``model``.

    ``max_tokens`` defaults to 128,000: only generated tokens are billed, and a higher
    cap only lowers the truncation (hence UNALIGNED) rate.
    """

    effort: str = "high"
    max_tokens: int = 128_000
    replicates: int = 3
    fallback: bool = False
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
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


def source_name(model: str, replicate: str) -> str:
    """``Link.source`` of one replicate, e.g. "claude-opus-5-5:collate.v1:r2"."""
    return f"{model}:{PROMPT_ID}:{replicate}"


# --------------------------------------------------------------------------- template and examples
@dataclass(frozen=True)
class Examples:
    """The rendered few-shot block and the language pair it illustrates."""

    reference_lang: str
    witness_lang: str
    text: str


def load_template() -> str:
    """The collation template shipped with the package (``prompts/collate.v1.md``)."""
    return resources.files("hevajra_matrix").joinpath("prompts").joinpath(TEMPLATE_FILE).read_text("utf-8")


def load_examples(data_dir: Path) -> Examples:
    """Read, check and render ``data/codebook/collate_examples.yaml``.

    The file must hold exactly three examples; each answer must be valid against
    ``SCHEMA``, name only handles its example shows and record every reference line
    exactly once. (That the answers also pass the V-checks is a unit test.)
    """
    path = Path(data_dir) / EXAMPLES_FILE
    doc = read_yaml(path)
    if not isinstance(doc, Mapping) or set(doc) != {"schema_version", "reference_lang", "witness_lang", "examples"}:
        raise ValueError(f"{path}: expected keys schema_version, reference_lang, witness_lang, examples")
    items = doc["examples"]
    if doc["schema_version"] != 1 or not isinstance(items, list) or len(items) != N_EXAMPLES:
        raise ValueError(f"{path}: schema_version 1 with exactly {N_EXAMPLES} examples required")
    blocks = [_render_example(n, ex, doc["reference_lang"], doc["witness_lang"], f"{path}: example {n}")
              for n, ex in enumerate(items, start=1)]
    return Examples(doc["reference_lang"], doc["witness_lang"], "\n\n".join(blocks))


def _render_example(n: int, ex: Any, ref_lang: str, wit_lang: str, where: str) -> str:
    keys = {"title", "comment", "witness", "reference", "answer"}
    if not isinstance(ex, Mapping) or set(ex) != keys:
        raise ValueError(f"{where}: expected keys {sorted(keys)}")
    wit = [(f"z{i:04d}", line["kind"], line["text"]) for i, line in enumerate(ex["witness"], start=1)]
    ref = [(f"r{i:03d}", line["kind"], line["text"]) for i, line in enumerate(ex["reference"], start=1)]
    answer = ex["answer"]
    errors = validate(answer, SCHEMA)
    if errors:
        raise ValueError(f"{where}: answer violates the schema: {errors[:3]}")
    named = [h for u in answer["units"] for h in u["wit"]] + [h for w in answer["witness_only"] for h in w["wit"]]
    unknown = sorted({h for h in named if h not in {z for z, _, _ in wit}}
                     | {u["ref"] for u in answer["units"] if u["ref"] not in {r for r, _, _ in ref}})
    if unknown or [u["ref"] for u in answer["units"]] != [r for r, _, _ in ref]:
        raise ValueError(f"{where}: the answer must record r001.. once each, in order, and name only shown "
                         f"handles (unknown: {unknown})")
    return "\n".join([
        f"### Example {n}: {ex['title']}",
        "",
        str(ex["comment"]).strip(),
        "",
        render_context(wit, wit_lang),
        "",
        render_body(ref, ref_lang, [(wit[0][0], wit[-1][0])]),
        "",
        "ANSWER",
        canonical_json(answer),
    ])


def system_prompt(template: str, examples: Examples) -> str:
    return f"{template.rstrip()}\n\n{examples.text}\n"


# --------------------------------------------------------------------------- rendering
def _line(handle: str, kind: str, text: str) -> str:
    """One prompt line; tabs and line breaks inside the text would break the format."""
    return f"{handle}\t{prompt_kind(kind)}\t{_LAYOUT_RE.sub(' ', text).strip()}"


def render_context(lines: Sequence[tuple[str, str, str]], lang: str) -> str:
    """The witness text block: a header and one (handle, kind, text) line per segment."""
    return "\n".join([f"WITNESS TEXT ({LANG_NAMES.get(lang, lang)})", *(_line(*x) for x in lines)])


def render_body(lines: Sequence[tuple[str, str, str]], lang: str, core: Sequence[tuple[str, str]]) -> str:
    """The reference chunk block and the core-window ranges."""
    ranges = ", ".join(a if a == b else f"{a}-{b}" for a, b in core) or "(empty)"
    return "\n".join([
        f"REFERENCE CHUNK ({LANG_NAMES.get(lang, lang)})",
        *(_line(*x) for x in lines),
        "",
        f"CORE WINDOW: {ranges}",
    ])


def build_request(window: Window, settings: TaskSettings, replicate: str, examples: Examples,
                  template: str) -> LLMRequest:
    """The T1 request for one window and one replicate tag (pure)."""
    if (window.ref_lang, window.wit_lang) != (examples.reference_lang, examples.witness_lang):
        raise ValueError(f"window {window.key} is {window.ref_lang}->{window.wit_lang} but the examples "
                         f"illustrate {examples.reference_lang}->{examples.witness_lang}")
    system = system_prompt(template, examples)
    context = render_context(
        [(h, window.segment[sid].kind, window.segment[sid].text) for h, sid in window.wit_handles.items()],
        window.wit_lang)
    body = render_body(
        [(h, window.segment[uid].kind, window.segment[uid].text) for h, uid in window.ref_handles.items()],
        window.ref_lang, window.core_ranges())
    return LLMRequest(
        task=TASK, prompt_sha=sha256_text(system), system=system, context=context, body=body, schema=SCHEMA,
        effort=settings.effort, max_tokens=settings.max_tokens, replicate=replicate,
        allow_fallback=settings.fallback, model=settings.model,
    )


# --------------------------------------------------------------------------- parsing
@dataclass(frozen=True)
class RawUnit:
    """One unit record as the model returned it (handles, not ids)."""

    ref: str
    wit: tuple[str, ...]
    relation: Relation
    polarity_flip: bool
    confidence: str
    ref_quote: str
    wit_quote: str


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


def parse(response: LLMResponse, window: Window) -> Parsed:
    """Turn a response into a raw collation or the reason it cannot be used."""
    if response.substituted_model:
        hint = _raw(response.data, window) if response.status == "ok" else None
        return Unresolved(window.key, REASON_SUBSTITUTED_MODEL, hint if isinstance(hint, RawCollation) else None)
    if response.status == "refusal":
        return Unresolved(window.key, f"{REASON_REFUSED}:{response.refusal_category or 'unspecified'}")
    if response.status == "truncated":
        return Unresolved(window.key, REASON_TRUNCATED)
    if response.status != "ok":
        return Unresolved(window.key, REASON_INVALID)
    return _raw(response.data, window)


def _raw(data: Mapping[str, Any] | None, window: Window) -> Parsed:
    if data is None or validate(data, SCHEMA):
        return Unresolved(window.key, REASON_INVALID)
    units = tuple(
        RawUnit(ref=u["ref"].strip(), wit=tuple(h.strip() for h in u["wit"]), relation=Relation(u["relation"]),
                polarity_flip=u["polarity_flip"], confidence=u["confidence"], ref_quote=u["ref_quote"],
                wit_quote=u["wit_quote"])
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
    of them run concurrently. Transport, budget and cache-miss errors propagate after
    the requests not yet started are cancelled; answers already received stay cached.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    jobs = build_requests(windows, settings, replicates, examples, template)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(client.complete, request) for _, _, request in jobs]
        try:
            responses = [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    out: dict[str, list[Parsed]] = {tag: [] for tag in replicates}
    for (tag, window, _), response in zip(jobs, responses):
        out[tag].append(parse(response, window))
    return {tag: tuple(results) for tag, results in out.items()}
