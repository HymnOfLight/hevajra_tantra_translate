"""T2 component coder, request side: settings, the answer schema, batches with full segment
texts, prompts and few-shot examples. See ``collate.components`` for the whole task."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence, get_args

from ..config import ConfigError, from_mapping
from ..core.io import read_yaml
from ..core.types import Segment
from ..llm.client import DEFAULT_MODEL, Effort, LLMRequest, canonical_json, sha256_text
from ..llm.schema import strict, validate
from .collator import LANG_NAMES

TASK = "components"
TEMPLATE_FILE = "components.v1.md"
EXAMPLES_FILE = Path("codebook") / "components_examples.yaml"

SLOTS = ("agent", "action", "patient", "instrument", "place", "quantity", "condition", "negation_modality", "result")
CODES = ("ret", "gen", "sub", "lit", "om", "add")

SCHEMA: Mapping[str, Any] = strict({"type": "object", "properties": {"pairs": {"type": "array", "items": {
    "type": "object", "properties": {
        "pair": {"type": "string"},
        "polarity_flip": {"type": "boolean"},
        "slots": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "slot": {"type": "string", "enum": list(SLOTS)},
            "code": {"type": "string", "enum": list(CODES)},
            "ref_quote": {"type": "string"},
            "wit_quote": {"type": "string"},
        }}},
    }}}}})


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class ComponentTaskSettings:
    """``config/llm.yaml: tasks.components`` plus the top-level ``model``."""

    effort: Effort = "high"
    max_tokens: int = 32000
    replicates: int = 1
    fallback: bool = True            # on for proposal tasks (impl_decisions 4)
    pairs_per_call: int = 12
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        if self.effort not in get_args(Effort):
            raise ConfigError(f"tasks.{TASK}.effort must be one of {get_args(Effort)}, not {self.effort!r}")
        for name in ("max_tokens", "replicates", "pairs_per_call"):
            value = getattr(self, name)
            if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
                raise ConfigError(f"tasks.{TASK}.{name} must be a positive integer, not {value!r}")
        if not isinstance(self.fallback, bool):
            raise ConfigError(f"tasks.{TASK}.fallback must be true or false")

    @classmethod
    def from_config(cls, llm: Mapping[str, Any]) -> "ComponentTaskSettings":
        section = dict((llm.get("tasks") or {}).get(TASK) or {})
        if "model" in section:
            raise ConfigError(f"llm.yaml: tasks.{TASK} takes the model from the top-level 'model' key")
        return from_mapping(cls, {**section, "model": llm.get("model", DEFAULT_MODEL)}, f"llm.yaml: tasks.{TASK}")


@dataclass(frozen=True)
class Pair:
    """A (reference unit, linked witness segments) pair chosen for coding.

    ``relation`` and ``polarity_flip`` are the collation's; they are recorded for analysis
    and never shown to the model. ``selected_as`` says why the pair was chosen.
    """

    ref_id: str
    wit_ids: tuple[str, ...]
    relation: str
    polarity_flip: bool
    selected_as: frozenset[str]


# --------------------------------------------------------------------------- batches and requests
@dataclass(frozen=True)
class PairText:
    """A pair with its full segment texts, as read from the segment store."""

    pair: Pair
    ref: Segment
    wit: tuple[Segment, ...]

    @property
    def wit_text(self) -> str:
        """Linked witness text in link order; the space keeps Tibetan syllables apart."""
        return " ".join(s.text for s in self.wit)


@dataclass(frozen=True)
class ComponentBatch:
    """The pairs of one call. All reference segments share one language, as do all witness segments."""

    items: tuple[PairText, ...]

    def __post_init__(self) -> None:
        if not self.items or not all(i.wit for i in self.items):
            raise ValueError("a batch needs at least one pair, each with at least one witness segment")
        for side, langs in (("reference", {i.ref.lang for i in self.items}),
                            ("witness", {s.lang for i in self.items for s in i.wit})):
            if len(langs) != 1:
                raise ValueError(f"a batch needs one {side} language, not {sorted(langs)}")

    @property
    def ref_lang(self) -> str:
        return self.items[0].ref.lang

    @property
    def wit_lang(self) -> str:
        return self.items[0].wit[0].lang

    def handles(self) -> dict[str, PairText]:
        """Prompt handle ("p01", "p02", ...) -> pair."""
        width = max(2, len(str(len(self.items))))
        return {f"p{i:0{width}d}": item for i, item in enumerate(self.items, start=1)}


def plan_batches(pairs: Sequence[Pair], segments: Mapping[str, Segment], pairs_per_call: int) -> list[ComponentBatch]:
    """Consecutive batches of ``pairs_per_call`` pairs with their full texts from ``segments``.

    Raises ``ValueError`` for a pair without witness segments or naming a segment that is
    not in the store (a stale pair list must fail loudly, not be coded on partial text).
    """
    if pairs_per_call < 1:
        raise ValueError("pairs_per_call must be >= 1")
    items = []
    for p in pairs:
        missing = [i for i in (p.ref_id, *p.wit_ids) if i not in segments]
        if missing or not p.wit_ids:
            raise ValueError(f"pair {p.ref_id}: no witness segments or not in the segment store: {missing}")
        items.append(PairText(p, segments[p.ref_id], tuple(segments[i] for i in p.wit_ids)))
    return [ComponentBatch(tuple(items[i:i + pairs_per_call])) for i in range(0, len(items), pairs_per_call)]


def load_template() -> str:
    """The coding instructions shipped with the package (``prompts/components.v1.md``)."""
    return resources.files("hevajra_matrix").joinpath("prompts").joinpath(TEMPLATE_FILE).read_text("utf-8")


def render_body(batch: ComponentBatch) -> str:
    """The variable part: the two languages, then the lines of each pair.

    Runs of whitespace (tabs and line breaks included) become one space, since they would
    break the line format; nothing else of a text is changed or cut.
    """
    def line(*fields: str) -> str:
        return "\t".join(" ".join(f.split()) for f in fields)

    blocks = [f"REFERENCE LANGUAGE: {LANG_NAMES.get(batch.ref_lang, batch.ref_lang)}\n"
              f"WITNESS LANGUAGE: {LANG_NAMES.get(batch.wit_lang, batch.wit_lang)}"]
    for handle, item in batch.handles().items():
        blocks.append("\n".join([line(handle, "ref", item.ref.kind, item.ref.text),
                                 *(line(handle, "wit", s.kind, s.text) for s in item.wit)]))
    return "\n\n".join(blocks) + "\n"


def system_prompt(template: str, examples: str) -> str:
    return f"{template.rstrip()}\n\n## Examples\n\n{examples.strip()}\n"


def load_system(data_dir: Path) -> str:
    """The cached system prompt: the template followed by the rendered synthetic examples."""
    return system_prompt(load_template(), load_examples(data_dir))


def build_request(batch: ComponentBatch, settings: ComponentTaskSettings, system: str,
                  replicate: str = "r1") -> LLMRequest:
    """The T2 request for one batch; ``prompt_sha`` covers the whole system prompt."""
    return LLMRequest(task=TASK, prompt_sha=sha256_text(system), system=system, body=render_body(batch),
                      schema=SCHEMA, effort=settings.effort, max_tokens=settings.max_tokens, replicate=replicate,
                      allow_fallback=settings.fallback, model=settings.model)


def build_requests(pairs: Sequence[Pair], segments: Mapping[str, Segment], task_settings: ComponentTaskSettings,
                   system: str, pairs_per_call: int | None = None, replicate: str = "r1") -> list[LLMRequest]:
    """One request per batch of ``plan_batches`` (same order); None means the configured size.

    ``system`` is ``load_system(data_dir)``.
    """
    size = task_settings.pairs_per_call if pairs_per_call is None else pairs_per_call
    return [build_request(b, task_settings, system, replicate) for b in plan_batches(pairs, segments, size)]


# --------------------------------------------------------------------------- few-shot examples
def load_examples(data_dir: Path) -> str:
    """Read, check and render ``data/codebook/components_examples.yaml`` (synthetic examples).

    Each answer must be schema-valid and answer the example's pairs exactly once, in order.
    That every answer also passes ``verify_answer`` is a unit test (it needs a lexicon).
    """
    path = Path(data_dir) / EXAMPLES_FILE
    doc = read_yaml(path)
    if not isinstance(doc, Mapping) or set(doc) != {"schema_version", "reference_lang", "witness_lang", "examples"}:
        raise ValueError(f"{path}: expected keys schema_version, reference_lang, witness_lang, examples")
    if doc["schema_version"] != 1 or not isinstance(doc["examples"], list) or not doc["examples"]:
        raise ValueError(f"{path}: schema_version 1 with at least one example required")
    blocks = []
    for n, ex in enumerate(doc["examples"], start=1):
        batch, answer = example_batch(ex, doc["reference_lang"], doc["witness_lang"], f"{path}: example {n}")
        blocks.append("\n".join([f"### Example {n}: {ex['title']}", "", str(ex["comment"]).strip(), "",
                                 render_body(batch).rstrip(), "", "ANSWER", canonical_json(answer)]))
    return "\n\n".join(blocks)


def example_batch(ex: Any, ref_lang: str, wit_lang: str, where: str) -> tuple[ComponentBatch, Mapping[str, Any]]:
    """The batch an example shows (synthetic segment ids) and its checked answer.

    Each example pair takes its collation polarity from the answer, so a correct example
    raises no ``polarity_disagrees`` flag.
    """
    if not isinstance(ex, Mapping) or set(ex) != {"title", "comment", "pairs", "answer"}:
        raise ValueError(f"{where}: expected keys answer, comment, pairs, title")
    answer = ex["answer"]
    errors = validate(answer, SCHEMA)
    if errors:
        raise ValueError(f"{where}: answer violates the schema: {errors[:3]}")
    if len(answer["pairs"]) != len(ex["pairs"]):
        raise ValueError(f"{where}: the answer must code every example pair once, in order")
    items = []
    for i, (p, a) in enumerate(zip(ex["pairs"], answer["pairs"]), start=1):
        ref = Segment(f"example:{i}.r", "example_ref", ref_lang, p["reference"]["text"], "", "", p["reference"]["kind"])
        wit = tuple(Segment(f"example:{i}.w{j}", "example_wit", wit_lang, w["text"], "", "", w["kind"])
                    for j, w in enumerate(p["witness"], start=1))
        pair = Pair(ref.id, tuple(s.id for s in wit), "", a["polarity_flip"], frozenset())
        items.append(PairText(pair, ref, wit))
    batch = ComponentBatch(tuple(items))
    if [a["pair"] for a in answer["pairs"]] != list(batch.handles()):
        raise ValueError(f"{where}: the answer must code {list(batch.handles())} once each, in order")
    return batch, answer
