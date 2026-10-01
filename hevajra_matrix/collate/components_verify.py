"""T2 component coder, verification side: rules C1-C5 on an answer (see ``collate.components``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core.textnorm import find_terms
from ..core.translit import transliteration_density
from ..core.types import (
    REASON_SUBSTITUTED_MODEL,
    REASON_UNASSESSED,
    REASON_VERIFICATION_FAILED,
    Diagnostic,
    failure_reason,
)
from ..llm.client import LLMResponse
from ..llm.schema import validate
from .components_request import SCHEMA, ComponentBatch, PairText
from .verify import FLAG_CORROBORATED, FLAG_POLARITY_UNCORROBORATED, FLAG_QUOTE_VARIANT, CheckLexicon, quote_match

BOTH_QUOTES = frozenset({"ret", "gen", "sub", "lit"})
FLIP_CODES = frozenset({"sub", "om", "add"})
MIN_TRANSLITERATION_DENSITY = 0.5

FLAG_POLARITY_DISAGREES = "polarity_disagrees"
FLAG_DUPLICATE_SLOT = "duplicate_slot"


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
        reason = failure_reason(response.status, response.refusal_category)
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
