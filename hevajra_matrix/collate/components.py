"""T2 component coder: which parts of a clause the witness keeps, generalises, replaces,
transcribes, drops or adds (synthesis 3.3; review defects #11 and #16).

    pairs    = select_pairs(consensus_alignment, topics, rng, ref_kinds=...)
    requests = build_requests(pairs, segments, settings)          # one call per batch
    codes, diagnostics = verify(client.complete(request), batch, lexicon)
    rows     = rendering_profile(codes, "bo", "zh")                # descriptive CSV

``code_components`` runs the whole loop. Everything else is a pure function.

Role: descriptive only. The codes feed ``components.jsonl`` and ``rendering_profile.csv``
and are never an input to the estimands E1-E4 (impl_decisions, "T2 components"); they
become the cell dimension ``d_comp`` only once per-code precision and recall on
double-coded dev pairs are reported.

What the model sees (defect #16)
    The full text of every segment, read from the segment store (``segments``: id ->
    ``Segment``), never the truncated text of a matrix cell. A pair is shown as one
    ``handle<TAB>ref<TAB>kind<TAB>text`` line and one ``handle<TAB>wit<TAB>kind<TAB>text``
    line per linked witness segment. Neither ids, coordinates, topic labels nor the
    collator's relation reach the prompt, so the coder is blind to why a pair was chosen
    (which is what lets the ``equivalent`` sample estimate how often T2 invents deviations).

What code checks (a pair is kept only if every rule passes; otherwise the pair gets a
``verification_failed`` diagnostic naming each broken rule and no codes)
    C1 pair set     every batch handle answered exactly once: an unknown handle is ignored,
                    a repeated one keeps its first answer, a missing one is ``unassessed``
    C2 quote shape  ret, gen, sub and lit need both quotes; ``om`` needs a reference quote
                    and an empty witness quote; ``add`` the reverse
    C3 verbatim     each non-empty quote occurs in its own side's full text
                    (``collate.verify.quote_match``; a match only after variant folding is
                    kept with flag ``quote_variant_form``)
    C4 lit          the witness quote is at least 50% transcription (V11, ``core.translit``)
    C5 polarity     ``polarity_flip`` requires a ``negation_modality`` slot coded sub, om or add
Non-fatal flags: ``corroborated`` / ``polarity_uncorroborated`` (a negator occurs in exactly
one quote of the negation_modality slot(s) of a flipped pair), ``polarity_disagrees`` (T2's
flip differs from the collation's), ``duplicate_slot`` (an identical slot entry was dropped).

A refusal, a truncation or a schema-invalid answer gives every pair of the batch a
diagnostic with that reason and no codes. Server-side fallback is on for this task
(impl_decisions 4): an answer from a substituted model is verified the same way, but its
codes carry reason ``substituted_model`` (reviewer hints; ``rendering_profile`` skips them).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, get_args

from ..config import ConfigError, from_mapping
from ..core.io import read_yaml
from ..core.textnorm import find_terms, for_quote
from ..core.translit import transliteration_density
from ..core.types import (
    COUNTED_STATUSES,
    REASON_INVALID,
    REASON_REFUSED,
    REASON_SUBSTITUTED_MODEL,
    REASON_TRUNCATED,
    REASON_UNASSESSED,
    REASON_VERIFICATION_FAILED,
    Alignment,
    Cell,
    Diagnostic,
    Relation,
    Segment,
)
from ..llm.client import DEFAULT_MODEL, Effort, LLMClient, LLMRequest, LLMResponse, canonical_json, sha256_text
from ..llm.schema import strict, validate
from ..matrix.status import deviates, outcome_class
from ..topics.codebook import topic_group
from .collator import LANG_NAMES
from .verify import FLAG_CORROBORATED, FLAG_POLARITY_UNCORROBORATED, FLAG_QUOTE_VARIANT, CheckLexicon, quote_match

TASK = "components"
TEMPLATE_FILE = "components.v1.md"
EXAMPLES_FILE = Path("codebook") / "components_examples.yaml"

SLOTS = ("agent", "action", "patient", "instrument", "place", "quantity", "condition", "negation_modality", "result")
CODES = ("ret", "gen", "sub", "lit", "om", "add")
BOTH_QUOTES = frozenset({"ret", "gen", "sub", "lit"})
FLIP_CODES = frozenset({"sub", "om", "add"})
MIN_TRANSLITERATION_DENSITY = 0.5
DEFAULT_SAMPLE_FRACTION = 0.10

SELECT_DEVIATION = "deviation"                 # consensus D_any = 1
SELECT_SENSITIVE = "sensitive"                 # topic group "sensitive"
SELECT_EQUIVALENT_SAMPLE = "equivalent_sample"  # random sample of equivalent links

FLAG_POLARITY_DISAGREES = "polarity_disagrees"
FLAG_DUPLICATE_SLOT = "duplicate_slot"

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


# --------------------------------------------------------------------------- selection
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


def select_pairs(links: Alignment | Iterable[Cell], topics: Mapping[str, Iterable[str]], rng: random.Random, *,
                 ref_kinds: Mapping[str, str], sample_fraction: float = DEFAULT_SAMPLE_FRACTION) -> list[Pair]:
    """Pairs to code, in input order: every D_any = 1 link, every link of a sensitive unit,
    and a ``sample_fraction`` random sample (``rng``) of the remaining ``equivalent`` links.

    ``links`` is the consensus ``Alignment`` or the matrix cells (countable cells only; a
    cell's relation already reflects gold and human verdicts). ``topics`` maps unit id ->
    topic set (e.g. ``{uid: label.topics}``); ``ref_kinds`` maps unit id -> segment kind,
    which decides whether ``transliterated`` deviates. Links without witness segments
    (``no_counterpart``) are skipped: there is nothing to compare component by component.
    The sample size is ``sample_fraction * n`` rounded half up.
    """
    if not 0.0 <= sample_fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in [0, 1], not {sample_fraction}")
    candidates = [c for c in _candidates(links) if c[1]]
    chosen: list[tuple[tuple[str, tuple[str, ...], str, bool], set[str]]] = []
    equivalent: list[int] = []
    for cand in candidates:
        uid, _, relation, flip = cand
        why = set()
        if deviates(outcome_class(Relation(relation), flip, ref_kinds.get(uid, ""))):
            why.add(SELECT_DEVIATION)
        if topic_group(topics.get(uid, ())) == "sensitive":
            why.add(SELECT_SENSITIVE)
        if not why and relation == Relation.EQUIVALENT.value and not flip:
            equivalent.append(len(chosen))
        chosen.append((cand, why))
    k = math.floor(sample_fraction * len(equivalent) + 0.5)
    for i in rng.sample(equivalent, k):
        chosen[i][1].add(SELECT_EQUIVALENT_SAMPLE)
    return [Pair(uid, wit, rel, flip, frozenset(why)) for (uid, wit, rel, flip), why in chosen if why]


def _candidates(links: Alignment | Iterable[Cell]) -> list[tuple[str, tuple[str, ...], str, bool]]:
    """(unit id, witness ids, relation value, polarity flip) of each reference unit, in order."""
    if isinstance(links, Alignment):
        return [(uid, link.wit_ids, str(link.relation), link.polarity_flip) for uid, link in links.by_ref().items()]
    return [(c.unit_id, c.wit_ids, c.relation, c.polarity_flip) for c in links
            if c.status in COUNTED_STATUSES and c.relation is not None]


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


# --------------------------------------------------------------------------- verification
@dataclass(frozen=True)
class SlotCode:
    """One verified slot code of one pair. ``reason`` None: usable; "substituted_model": hint only."""

    ref_id: str
    wit_ids: tuple[str, ...]
    slot: str
    code: str
    ref_quote: str
    wit_quote: str
    polarity_flip: bool
    flags: frozenset[str] = frozenset()
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.reason is None


def verify(response: LLMResponse, batch: ComponentBatch,
           lexicon: CheckLexicon) -> tuple[list[SlotCode], list[Diagnostic]]:
    """Verify one answer against its batch (rules C1-C5 of the module docstring)."""
    if response.status != "ok" or response.data is None or validate(response.data, SCHEMA):
        reason = _failure_reason(response)
        return [], [Diagnostic(reason, (i.pair.ref_id,), i.pair.wit_ids, f"{h}: {reason}")
                    for h, i in batch.handles().items()]
    reason = REASON_SUBSTITUTED_MODEL if response.substituted_model else None
    codes, diagnostics = verify_answer(response.data, batch, lexicon, reason)
    if reason:
        diagnostics.append(Diagnostic(reason, tuple(i.pair.ref_id for i in batch.items),
                                      detail=f"served by {response.served_model}: codes are reviewer hints"))
    return codes, diagnostics


def verify_answer(data: Mapping[str, Any], batch: ComponentBatch, lexicon: CheckLexicon,
                  reason: str | None = None) -> tuple[list[SlotCode], list[Diagnostic]]:
    """Verify a schema-valid answer; ``reason`` is stamped on every code (None: usable)."""
    handles = batch.handles()
    answers: dict[str, Mapping[str, Any]] = {}
    diagnostics: list[Diagnostic] = []
    for entry in data["pairs"]:                                                    # C1
        h = entry["pair"].strip()
        if h not in handles:
            diagnostics.append(Diagnostic("unknown_handle", detail=f"pair handle {h!r} is not in the batch"))
        elif h in answers:
            diagnostics.append(Diagnostic("duplicate_pair", (handles[h].pair.ref_id,), detail=f"{h}: kept the first"))
        else:
            answers[h] = entry
    codes: list[SlotCode] = []
    for h, item in handles.items():
        p = item.pair
        if h not in answers:
            diagnostics.append(Diagnostic(REASON_UNASSESSED, (p.ref_id,), p.wit_ids, f"{h}: no answer"))
            continue
        found, errors = _check_pair(answers[h], item, batch, lexicon, reason)
        if errors:
            diagnostics.append(Diagnostic(REASON_VERIFICATION_FAILED, (p.ref_id,), p.wit_ids,
                                          f"{h}: " + "; ".join(errors)))
        else:
            codes.extend(found)
    return codes, diagnostics


def _check_pair(entry: Mapping[str, Any], item: PairText, batch: ComponentBatch, lex: CheckLexicon,
                reason: str | None) -> tuple[list[SlotCode], list[str]]:
    """The pair's codes and the list of broken rules (empty when the pair is kept)."""
    flip = entry["polarity_flip"]
    errors: list[str] = []
    slots: list[tuple[str, str, str, str]] = []
    variant = False
    for s in entry["slots"]:
        key = (s["slot"], s["code"], s["ref_quote"].strip(), s["wit_quote"].strip())
        if key in slots:
            continue
        slots.append(key)
        slot, code, ref_q, wit_q = key
        where = f"{slot}/{code}"
        if bool(ref_q) != (code != "add") or bool(wit_q) != (code != "om"):              # C2
            need = {"om": "a ref_quote and an empty wit_quote", "add": "a wit_quote and an empty ref_quote"}
            errors.append(f"C2 {where} needs {need.get(code, 'both quotes')}")
            continue
        for side, quote, text, lang in (("ref", ref_q, item.ref.text, batch.ref_lang),
                                        ("wit", wit_q, item.wit_text, batch.wit_lang)):
            if quote:                                                                   # C3
                match = quote_match(quote, text, lang, lex.variants.for_lang(lang))
                if match is None:
                    errors.append(f"C3 {where} {side}_quote is not verbatim in the {side} text")
                variant = variant or match == "variant"
        if code == "lit" and transliteration_density(                                  # C4
                wit_q, batch.wit_lang, lex.translit_charset) < MIN_TRANSLITERATION_DENSITY:
            errors.append(f"C4 {where} wit_quote is not mostly transcription")
    negation = [(r, w) for slot, code, r, w in slots if slot == "negation_modality" and code in FLIP_CODES]
    if flip and not negation:                                                           # C5
        errors.append("C5 polarity_flip needs a negation_modality slot coded sub, om or add")
    flags = set()
    if len(slots) < len(entry["slots"]):
        flags.add(FLAG_DUPLICATE_SLOT)
    if variant:
        flags.add(FLAG_QUOTE_VARIANT)
    if flip != item.pair.polarity_flip:
        flags.add(FLAG_POLARITY_DISAGREES)
    if flip:
        corroborated = any(_negated(r, batch.ref_lang, lex) != _negated(w, batch.wit_lang, lex) for r, w in negation)
        flags.add(FLAG_CORROBORATED if corroborated else FLAG_POLARITY_UNCORROBORATED)
    p = item.pair
    return [SlotCode(p.ref_id, p.wit_ids, slot, code, r, w, flip, frozenset(flags), reason)
            for slot, code, r, w in slots], errors


def _negated(text: str, lang: str, lex: CheckLexicon) -> bool:
    negators = lex.negators
    return bool(text) and bool(find_terms(text, lang, negators.for_lang(lang), negators.exclusions_for(lang)))


def _failure_reason(response: LLMResponse) -> str:
    if response.status == "refusal":
        return f"{REASON_REFUSED}:{response.refusal_category}" if response.refusal_category else REASON_REFUSED
    return REASON_TRUNCATED if response.status == "truncated" else REASON_INVALID


def code_components(pairs: Sequence[Pair], segments: Mapping[str, Segment], client: LLMClient,
                    settings: ComponentTaskSettings, system: str, lexicon: CheckLexicon, workers: int = 1,
                    replicate: str = "r1") -> tuple[list[SlotCode], list[Diagnostic]]:
    """Plan, request (``workers`` at a time) and verify every batch; results in pair order."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    batches = plan_batches(pairs, segments, settings.pairs_per_call)
    requests = [build_request(b, settings, system, replicate) for b in batches]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(client.complete, requests))
    codes: list[SlotCode] = []
    diagnostics: list[Diagnostic] = []
    for batch, response in zip(batches, responses):
        found, diags = verify(response, batch, lexicon)
        codes.extend(found)
        diagnostics.extend(diags)
    return codes, diagnostics


# --------------------------------------------------------------------------- outputs
CODE_COLUMNS = ("ref_id", "wit_ids", "slot", "code", "ref_quote", "wit_quote", "polarity_flip", "flags", "reason")
PROFILE_COLUMNS = ("source_quote", "rendering_quote", "count", "n_units", "units", "codes", "slots")


def code_record(code: SlotCode) -> dict[str, Any]:
    """One ``components.jsonl`` record (JSON-ready; lists sorted)."""
    return {"ref_id": code.ref_id, "wit_ids": list(code.wit_ids), "slot": code.slot, "code": code.code,
            "ref_quote": code.ref_quote, "wit_quote": code.wit_quote, "polarity_flip": code.polarity_flip,
            "flags": sorted(code.flags), "reason": code.reason}


@dataclass(frozen=True)
class ProfileRow:
    """How often one source-term quote was rendered by one witness quote (descriptive).

    An empty ``rendering_quote`` is an omission (``om``); an empty ``source_quote`` an
    addition (``add``). ``codes`` and ``slots`` count the codes and slots of the rows' codes.
    """

    source_quote: str
    rendering_quote: str
    count: int
    units: tuple[str, ...]
    codes: tuple[tuple[str, int], ...]
    slots: tuple[tuple[str, int], ...]

    def csv_row(self) -> dict[str, Any]:
        return {"source_quote": self.source_quote, "rendering_quote": self.rendering_quote, "count": self.count,
                "n_units": len(self.units), "units": ";".join(self.units),
                "codes": ";".join(f"{c}:{n}" for c, n in self.codes),
                "slots": ";".join(f"{s}:{n}" for s, n in self.slots)}


def rendering_profile(slot_codes: Iterable[SlotCode], ref_lang: str, wit_lang: str) -> list[ProfileRow]:
    """Group usable codes by (source quote, rendering quote) after ``for_quote`` normalisation.

    Codes with a reason (substituted-model hints) are skipped. The first spelling seen is
    shown. Rows are ordered by count (descending), then by the normalised quotes.
    """
    groups: dict[tuple[str, str], list[SlotCode]] = {}
    for c in slot_codes:
        if c.usable:
            groups.setdefault((for_quote(c.ref_quote, ref_lang), for_quote(c.wit_quote, wit_lang)), []).append(c)
    rows = [(key, ProfileRow(members[0].ref_quote, members[0].wit_quote, len(members),
                             tuple(dict.fromkeys(m.ref_id for m in members)),
                             tuple(sorted(Counter(m.code for m in members).items(), key=lambda x: CODES.index(x[0]))),
                             tuple(sorted(Counter(m.slot for m in members).items(), key=lambda x: SLOTS.index(x[0])))))
            for key, members in groups.items()]
    return [row for _, row in sorted(rows, key=lambda kr: (-kr[1].count, kr[0]))]


def invention_rate(pairs: Iterable[Pair], slot_codes: Iterable[SlotCode]) -> tuple[int, int]:
    """(coded pairs of the equivalent sample with any code other than ``ret``, coded sample pairs).

    Only pairs selected solely for the sample count, and only usable codes.
    """
    sample = {p.ref_id for p in pairs if p.selected_as == {SELECT_EQUIVALENT_SAMPLE}}
    coded: dict[str, set[str]] = {}
    for c in slot_codes:
        if c.usable and c.ref_id in sample:
            coded.setdefault(c.ref_id, set()).add(c.code)
    return sum(1 for codes in coded.values() if codes != {"ret"}), len(coded)


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
