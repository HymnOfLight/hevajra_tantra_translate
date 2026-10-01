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
from typing import Any, Iterable, Mapping, Sequence

from ..core.textnorm import for_quote
from ..core.types import COUNTED_STATUSES, Alignment, Cell, Diagnostic, Relation, Segment
from ..llm.client import LLMClient
from ..matrix.status import deviates, outcome_class
from ..topics.codebook import topic_group
from .components_request import (  # noqa: F401  (the public API of the task is re-exported here)
    CODES,
    EXAMPLES_FILE,
    SCHEMA,
    SLOTS,
    TASK,
    TEMPLATE_FILE,
    ComponentBatch,
    ComponentTaskSettings,
    Pair,
    PairText,
    build_request,
    build_requests,
    example_batch,
    load_examples,
    load_system,
    load_template,
    plan_batches,
    render_body,
    system_prompt,
)
from .components_verify import (  # noqa: F401
    BOTH_QUOTES,
    FLAG_DUPLICATE_SLOT,
    FLAG_POLARITY_DISAGREES,
    FLIP_CODES,
    MIN_TRANSLITERATION_DENSITY,
    SlotCode,
    verify,
    verify_answer,
)
from .verify import CheckLexicon

DEFAULT_SAMPLE_FRACTION = 0.10

SELECT_DEVIATION = "deviation"                 # consensus D_any = 1
SELECT_SENSITIVE = "sensitive"                 # topic group "sensitive"
SELECT_EQUIVALENT_SAMPLE = "equivalent_sample"  # random sample of equivalent links


# --------------------------------------------------------------------------- selection
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


# --------------------------------------------------------------------------- run
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
