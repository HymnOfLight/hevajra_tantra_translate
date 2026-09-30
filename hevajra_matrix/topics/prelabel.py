"""T3 topic pre-labeller: request building and answer verification (synthesis 3.3).

    batches  = plan_batches(reference_units, settings.units_per_call)
    request  = build_request(batch, codebook, settings)          # one call per batch
    labels   = parse_prelabels(client.complete(request), batch)  # unit id -> Prelabel

``prelabel_requests`` builds all requests at once (dry runs, cost projection, tests).

What the model sees: the prompt template ``prompts/topics.v1.md`` with the codebook's
rules and definitions (the cached ``system`` block), then a body of fixed section markers
and ``handle<TAB>kind<TAB>text`` lines: the units to label, flanked by up to
``context_units`` neighbours on each side marked as context not to be labelled. Handles
("u01", ...) stand in for unit ids, so no coordinates reach the prompt. Only segments of
one reference-language witness are accepted, which keeps the Chinese out by construction.

What code checks: every unit to label gets a ``Prelabel``. A non-neutral topic is kept only
with a cue found verbatim in that unit's own text (``core.textnorm.quote_in``), otherwise it
is dropped and flagged; ``neutral`` is kept only when it is the unit's sole topic. Refused,
truncated and invalid answers yield no topics (the units go to the top of the human queue;
there is no automatic retry). An answer from a substituted model is verified the same way
but carries reason ``substituted_model``: a reviewer hint that is never committed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from types import MappingProxyType
from typing import Any, Mapping, Sequence, get_args

from ..config import ConfigError, from_mapping
from ..core.textnorm import quote_in
from ..core.types import (
    REASON_INVALID,
    REASON_REFUSED,
    REASON_SUBSTITUTED_MODEL,
    REASON_TRUNCATED,
    REASON_UNASSESSED,
    Segment,
)
from ..llm.client import DEFAULT_MODEL, Effort, LLMRequest, LLMResponse, sha256_text
from ..llm.schema import strict
from .codebook import NEUTRAL, TOPICS, TopicCodebook

TASK = "topics"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "topics.v1.md"
# Languages a reference text may be written in (the witness, T0892, is Chinese).
REFERENCE_LANGS: Mapping[str, str] = MappingProxyType({"bo": "Tibetan", "sa": "Sanskrit (IAST transliteration)"})
BEFORE, UNITS, AFTER = "[context before: do not label]", "[units to label]", "[context after: do not label]"
CONTEXT_HANDLE = "ctx"
_LINE_BREAKS = re.compile(r"[\t\r\n]+")

FLAG_CUE_UNVERIFIED = "cue_unverified"        # written as "cue_unverified:<topic>"
FLAG_NEUTRAL_NOT_ALONE = "neutral_not_alone"
FLAG_DUPLICATE_REF = "duplicate_ref"


@dataclass(frozen=True)
class TopicTaskSettings:
    """``config/llm.yaml: tasks.topics`` plus the top-level ``model``."""

    effort: Effort = "medium"
    max_tokens: int = 16000
    replicates: int = 1
    fallback: bool = True            # on for proposal tasks (impl_decisions 4)
    units_per_call: int = 40
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        if self.effort not in get_args(Effort):
            raise ConfigError(f"tasks.topics.effort must be one of {get_args(Effort)}, not {self.effort!r}")
        if self.replicates != 1:
            raise ConfigError("tasks.topics.replicates must be 1: T3 has no replicate vote (synthesis 3.3)")
        for name in ("max_tokens", "units_per_call"):
            value = getattr(self, name)
            if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
                raise ConfigError(f"tasks.topics.{name} must be a positive integer, not {value!r}")
        if not isinstance(self.fallback, bool):
            raise ConfigError("tasks.topics.fallback must be true or false")

    @classmethod
    def from_config(cls, llm: Mapping[str, Any]) -> "TopicTaskSettings":
        """Build from the parsed ``llm.yaml``; unknown keys raise ``ConfigError``."""
        section = dict((llm.get("tasks") or {}).get(TASK) or {})
        if "model" in llm:
            section.setdefault("model", llm["model"])
        return from_mapping(cls, section, f"llm.yaml: tasks.{TASK}")


@dataclass(frozen=True)
class PrelabelBatch:
    """Units to label in one call, with neighbouring units shown as context only."""

    units: tuple[Segment, ...]
    before: tuple[Segment, ...] = ()
    after: tuple[Segment, ...] = ()

    def handles(self) -> dict[str, Segment]:
        """Prompt handle ("u01", "u02", ...) -> unit to label."""
        width = max(2, len(str(len(self.units))))
        return {f"u{i:0{width}d}": unit for i, unit in enumerate(self.units, start=1)}


@dataclass(frozen=True)
class Prelabel:
    """The verified pre-label of one unit.

    ``reason`` None means usable; otherwise ``refused[:<category>]``, ``truncated``,
    ``invalid``, ``unassessed`` (the answer skipped the unit) or ``substituted_model``
    (verified topics kept as a reviewer hint only). ``cues`` pairs each kept topic with its
    verified verbatim cue; ``flags`` records what verification dropped.
    """

    unit_id: str
    topics: frozenset[str] = frozenset()
    cues: tuple[tuple[str, str], ...] = ()
    flags: frozenset[str] = frozenset()
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.reason is None

    @property
    def committed_topics(self) -> frozenset[str]:
        """What may be written to the committed ``prelabel_topics`` column."""
        return self.topics if self.usable else frozenset()


def plan_batches(reference: Sequence[Segment], units_per_call: int = 40, context_units: int = 2) -> list[PrelabelBatch]:
    """Cut reference units, in the given (document) order, into consecutive batches.

    Every unit is labelled in exactly one batch; up to ``context_units`` neighbours on each
    side are attached as context. Raises ``ValueError`` unless all segments come from one
    witness written in a reference language: this is what keeps any witness text out.
    """
    units = tuple(reference)
    witnesses = sorted({s.witness for s in units})
    if len(witnesses) > 1:
        raise ValueError(f"pre-labelling takes the units of one reference witness, not {witnesses}")
    foreign = sorted({s.lang for s in units} - set(REFERENCE_LANGS))
    if foreign:
        raise ValueError(f"language(s) {foreign} are not reference languages {sorted(REFERENCE_LANGS)}: "
                         "the topic pre-labeller must never see a witness text")
    repeated = sorted(u for u, c in Counter(s.id for s in units).items() if c > 1)
    if repeated:
        raise ValueError(f"duplicate unit ids {repeated[:5]}")
    if units_per_call < 1 or context_units < 0:
        raise ValueError("units_per_call must be >= 1 and context_units >= 0")
    n, c = units_per_call, context_units
    return [PrelabelBatch(units[i:i + n], units[max(0, i - c):i], units[i + n:i + n + c])
            for i in range(0, len(units), n)]


def prelabel_schema(codebook: TopicCodebook) -> dict[str, Any]:
    """Strict output schema: per unit a handle and one or more (topic, cue) pairs; no free text."""
    topic = {"type": "object", "properties": {
        "topic": {"type": "string", "enum": list(codebook.names)},
        "cue": {"type": "string", "description": "verbatim excerpt of this unit's own text; empty for neutral"},
    }}
    unit = {"type": "object", "properties": {
        "ref": {"type": "string", "description": "handle of a unit under [units to label]"},
        "topics": {"type": "array", "minItems": 1, "items": topic},
    }}
    return strict({"type": "object", "properties": {"units": {"type": "array", "items": unit}}})


def render_system(codebook: TopicCodebook, lang: str) -> str:
    """The static, cached prompt: template, coding rules and topic definitions."""
    topics = "\n\n".join(
        f"### {t.name} (group: {t.group})\n\n{t.definition}\n\n"
        + "Include:\n" + "\n".join(f"- {x}" for x in t.include) + "\n\n"
        + "Exclude:\n" + "\n".join(f"- {x}" for x in t.exclude)
        for t in codebook.topics
    )
    rules = "\n".join(f"- {r}" for r in codebook.rules)
    return _template().substitute(language=REFERENCE_LANGS[lang], rules=rules, topics=topics)


@lru_cache(maxsize=1)
def _template() -> Template:
    return Template(PROMPT_PATH.read_text(encoding="utf-8"))


def render_body(batch: PrelabelBatch) -> str:
    """The variable part: the three fixed section markers and one line per unit."""
    def line(handle: str, s: Segment) -> str:
        return "\t".join((handle, s.kind, _LINE_BREAKS.sub(" ", s.text).strip()))

    return "\n".join([
        BEFORE, *(line(CONTEXT_HANDLE, s) for s in batch.before), "",
        UNITS, *(line(h, s) for h, s in batch.handles().items()), "",
        AFTER, *(line(CONTEXT_HANDLE, s) for s in batch.after),
    ]) + "\n"


def build_request(batch: PrelabelBatch, codebook: TopicCodebook, settings: TopicTaskSettings,
                  replicate: str = "r1") -> LLMRequest:
    """The T3 request for one batch. ``prompt_sha`` covers the rendered system prompt, so a
    codebook edit is an instrument change; a new ``replicate`` tag forces a fresh call."""
    if not batch.units:
        raise ValueError("a batch needs at least one unit to label")
    system = render_system(codebook, batch.units[0].lang)
    return LLMRequest(task=TASK, prompt_sha=sha256_text(system), system=system, body=render_body(batch),
                      schema=prelabel_schema(codebook), effort=settings.effort, max_tokens=settings.max_tokens,
                      replicate=replicate, allow_fallback=settings.fallback, model=settings.model)


def prelabel_requests(reference_segments: Sequence[Segment], codebook: TopicCodebook,
                      task_settings: TopicTaskSettings, units_per_call: int | None = None,
                      context_units: int = 2, replicate: str = "r1") -> list[LLMRequest]:
    """One request per batch of ``plan_batches``; ``units_per_call`` None means the configured size.

    A smaller ``units_per_call`` (e.g. to rerun refused batches) creates new cache keys.
    """
    size = task_settings.units_per_call if units_per_call is None else units_per_call
    return [build_request(b, codebook, task_settings, replicate)
            for b in plan_batches(reference_segments, size, context_units)]


def parse_prelabels(response: LLMResponse, batch: PrelabelBatch) -> dict[str, Prelabel]:
    """Verify one answer against its batch: unit id -> ``Prelabel``, for every unit to label.

    Answers for context lines or unknown handles are ignored; a repeated handle keeps its
    first answer and is flagged ``duplicate_ref``; a unit without an answer is
    ``unassessed``.
    """
    if response.status != "ok" or response.data is None:
        reason = _failure_reason(response)
        return {s.id: Prelabel(s.id, reason=reason) for s in batch.units}
    reason = REASON_SUBSTITUTED_MODEL if response.substituted_model else None
    answers: dict[str, list[Sequence[Mapping[str, str]]]] = {}
    for entry in response.data["units"]:
        answers.setdefault(entry["ref"].strip(), []).append(entry["topics"])
    out: dict[str, Prelabel] = {}
    for handle, unit in batch.handles().items():
        found = answers.get(handle)
        if not found:
            out[unit.id] = Prelabel(unit.id, reason=REASON_UNASSESSED)
            continue
        flags = {FLAG_DUPLICATE_REF} if len(found) > 1 else set()
        out[unit.id] = _verified(unit, found[0], flags, reason)
    return out


def _verified(unit: Segment, proposed: Sequence[Mapping[str, str]], flags: set[str], reason: str | None) -> Prelabel:
    named = {p["topic"] for p in proposed}
    if NEUTRAL in named and len(named) > 1:
        flags.add(FLAG_NEUTRAL_NOT_ALONE)
    cues: dict[str, str] = {}
    for p in proposed:   # a topic named twice is kept if either cue verifies (the first that does)
        if p["topic"] != NEUTRAL and p["topic"] not in cues and quote_in(p["cue"], unit.text, unit.lang):
            cues[p["topic"]] = p["cue"]
    flags |= {f"{FLAG_CUE_UNVERIFIED}:{t}" for t in named - {NEUTRAL} - set(cues)}
    topics = frozenset(cues) or (frozenset({NEUTRAL}) if named == {NEUTRAL} else frozenset())
    return Prelabel(unit.id, topics, tuple((t, cues[t]) for t in TOPICS if t in cues), frozenset(flags), reason)


def _failure_reason(response: LLMResponse) -> str:
    if response.status == "refusal":
        return f"{REASON_REFUSED}:{response.refusal_category}" if response.refusal_category else REASON_REFUSED
    return REASON_TRUNCATED if response.status == "truncated" else REASON_INVALID
