"""Code checks V1-V11 on T1 answers: the model proposes, this module decides what stands.

Every function is pure. A failed check never turns into a silent PRESENT or ABSENT: a
fatal failure makes the unit UNALIGNED with a reason (``Collation.unresolved``), a
non-fatal one keeps the link with a flag (flags other than ``corroborated`` make the
consensus grade C), and window-level findings become ``Diagnostic`` records.

    V1  coverage         every reference handle exactly once. A duplicate keeps the first
                         record (flag ``duplicate_record``); a missing unit is
                         UNALIGNED(unassessed); more than 2% missing makes the whole
                         window UNALIGNED(invalid)
    V2  existence        unknown handles are dropped (flag ``unknown_handle``); a link left
                         with no witness segment is UNALIGNED(verification_failed)
    V3  consistency      ``no_counterpart`` exactly when ``wit`` is empty, else verification_failed
    V4  verbatim quotes  ``ref_quote`` in the unit, ``wit_quote`` in the linked segments
                         (``core.textnorm.quote_in``); a match only after variant folding
                         keeps the link with flag ``quote_variant_form``; no match is
                         verification_failed
    V5  required quotes  ``ref_quote`` for every relation except equivalent, paraphrase and
                         expanded; ``wit_quote`` whenever ``wit`` is non-empty
    V6  polarity         ``reversal`` exactly when ``polarity_flip`` (else flag
                         ``polarity_mismatch``); for a reversal or flip, a negator in exactly
                         one quote gives ``corroborated``, otherwise ``polarity_uncorroborated``
    V7  locality         a link outside the core window: flag and ``relocation`` diagnostic
    V8  order            units off the longest order-preserving chain of links: flag
                         ``crossing``; the number of crossing pairs goes in a diagnostic
    V9  exhaustiveness   see ``collate.merge`` (needs all chunks of a chapter)
    V10 overlap          see ``collate.merge`` (needs two adjacent chunks)
    V11 transliteration  ``transliterated`` needs a witness quote at least 50% transcription
                         (``core.translit``), else flag ``transliteration_unsupported``; a
                         non-mantra reference unit linked to a mantra segment gets the
                         candidate flag ``instruction_as_mantra``

Witness-only records are checked for V2, V4 and V5 too; a failing one is dropped (its
segments then show up under V9). A segment claimed both by a unit link and by a
witness-only record stays linked and leaves the record (flag ``witness_only_trimmed``).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from ..core.lexicon import Negators, VariantForms, load_negators, load_variant_forms
from ..core.textnorm import find_terms, fold_variants, for_quote, quote_in
from ..core.translit import transliteration_charset, transliteration_density
from ..core.types import (
    REASON_INVALID,
    REASON_UNASSESSED,
    REASON_VERIFICATION_FAILED,
    Alignment,
    Diagnostic,
    Link,
    Quote,
    Relation,
    Segment,
    WitnessOnlyKind,
)
from .collator import Parsed, RawCollation, RawUnit, Unresolved
from .windows import Window

MAX_MISSING_FRACTION = 0.02
MIN_TRANSLITERATION_DENSITY = 0.5
QUOTE_OPTIONAL = frozenset({Relation.EQUIVALENT, Relation.PARAPHRASE, Relation.EXPANDED})

FLAG_CORROBORATED = "corroborated"
FLAG_POLARITY_UNCORROBORATED = "polarity_uncorroborated"
FLAG_POLARITY_MISMATCH = "polarity_mismatch"
FLAG_QUOTE_VARIANT = "quote_variant_form"
FLAG_RELOCATION = "relocation"
FLAG_CROSSING = "crossing"
FLAG_OVERLAP_DISAGREEMENT = "overlap_disagreement"
FLAG_TRANSLITERATION_UNSUPPORTED = "transliteration_unsupported"
FLAG_INSTRUCTION_AS_MANTRA = "instruction_as_mantra"
FLAG_DUPLICATE = "duplicate_record"
FLAG_UNKNOWN_HANDLE = "unknown_handle"
FLAG_WITNESS_ONLY_TRIMMED = "witness_only_trimmed"
FLAG_HINT = "substituted_model_hint"


@dataclass(frozen=True)
class CheckLexicon:
    """Word lists the checks need. Build it with ``load`` from the whole witness text, so
    that the transliteration character set covers all of the witness's mantras."""

    negators: Negators
    variants: VariantForms
    translit_charset: frozenset[str] = frozenset()

    @classmethod
    def load(cls, data_dir: Path, witness_segments: Iterable[Segment] = ()) -> "CheckLexicon":
        return cls(load_negators(data_dir), load_variant_forms(data_dir), transliteration_charset(witness_segments))


@dataclass(frozen=True)
class Collation:
    """Verified output of one replicate over a window, a chapter or the whole text.

    ``alignment``    links that passed every fatal check: one per resolved reference
                     unit (in reference order), then the witness-only records
    ``unresolved``   reference unit id -> UNALIGNED reason (never also in ``alignment``)
    ``diagnostics``  findings for the operator and the review queue
    ``hints``        links parsed from a substituted model's answer (flag
                     ``substituted_model_hint``); reviewer hints, never measured
    """

    alignment: Alignment
    unresolved: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    diagnostics: tuple[Diagnostic, ...] = ()
    hints: tuple[Link, ...] = ()

    def unit_flags(self) -> dict[str, frozenset[str]]:
        """Reference unit id -> flags of its link (resolved units only)."""
        return {uid: link.flags for uid, link in self.alignment.by_ref().items()}


# --------------------------------------------------------------------------- one window
def verify(parsed: Parsed, window: Window, lexicon: CheckLexicon, source: str) -> Collation:
    """Apply V1-V8 and V11 to one window's parsed answer."""
    empty = Alignment(source, window.reference, window.witness)
    if isinstance(parsed, Unresolved):
        hints: tuple[Link, ...] = ()
        if parsed.hint is not None:
            checked = _verify_raw(parsed.hint, window, lexicon, f"{source}:hint")
            hints = tuple(replace(link, flags=link.flags | {FLAG_HINT}) for link in checked.alignment.links)
        return Collation(empty, frozen_mapping({u.id: parsed.reason for u in window.units}),
                         (Diagnostic("window_unresolved", tuple(u.id for u in window.units),
                                     detail=f"{window.key}: {parsed.reason}"),), hints)
    if parsed.window != window.key:
        raise ValueError(f"answer for window {parsed.window} verified against window {window.key}")
    return _verify_raw(parsed, window, lexicon, source)


def _verify_raw(raw: RawCollation, window: Window, lexicon: CheckLexicon, source: str) -> Collation:
    diagnostics: list[Diagnostic] = []
    records: dict[str, RawUnit] = {}
    duplicates: set[str] = set()
    for rec in raw.units:                                                        # V1, V2 (reference side)
        uid = window.ref_handles.get(rec.ref)
        if uid is None:
            diagnostics.append(Diagnostic("unknown_handle", detail=f"{window.key}: reference handle {rec.ref!r}"))
        elif uid in records:
            duplicates.add(uid)
        else:
            records[uid] = rec
    missing = tuple(u.id for u in window.units if u.id not in records)
    if len(missing) > MAX_MISSING_FRACTION * len(window.units):
        diagnostics.append(Diagnostic("window_invalid", missing,
                                      detail=f"{window.key}: {len(missing)} of {len(window.units)} units missing"))
        return Collation(Alignment(source, window.reference, window.witness),
                         frozen_mapping({u.id: REASON_INVALID for u in window.units}), tuple(diagnostics))
    if missing:
        diagnostics.append(Diagnostic("unassessed", missing, detail=f"{window.key}: units missing from the answer"))
    unresolved = {uid: REASON_UNASSESSED for uid in missing}
    unit_links: list[Link] = []
    for unit in window.units:
        if unit.id not in records:
            continue
        outcome = _check_unit(unit, records[unit.id], unit.id in duplicates, window, lexicon, source)
        if isinstance(outcome, Diagnostic):
            unresolved[unit.id] = REASON_VERIFICATION_FAILED
            diagnostics.append(outcome)
            continue
        link, found = outcome
        unit_links.append(link)
        diagnostics.extend(found)
    unit_links, crossing = _order_check(unit_links, window)                      # V8
    diagnostics.extend(crossing)
    witness_only: list[Link] = []
    for rec in raw.witness_only:
        outcome = _check_witness_only(rec.wit, rec.kind, rec.wit_quote, window, lexicon, source)
        if isinstance(outcome, Diagnostic):
            diagnostics.append(outcome)
        else:
            witness_only.append(outcome)
    links = unit_links + reconcile_witness_only(unit_links, witness_only)
    return Collation(Alignment(source, window.reference, window.witness, tuple(links)),
                     frozen_mapping(unresolved), tuple(diagnostics))


def _check_unit(unit: Segment, rec: RawUnit, duplicate: bool, window: Window, lex: CheckLexicon,
                source: str) -> tuple[Link, list[Diagnostic]] | Diagnostic:
    """The link of one unit and its non-fatal findings, or the diagnostic of a fatal failure."""
    def failed(rule: str) -> Diagnostic:
        return Diagnostic(REASON_VERIFICATION_FAILED, (unit.id,), detail=f"{window.key} {rec.ref}: {rule}")

    relation = rec.relation
    flags: set[str] = {FLAG_DUPLICATE} if duplicate else set()
    wit_ids, unknown = _resolve(rec.wit, window)
    if unknown:
        flags.add(FLAG_UNKNOWN_HANDLE)
    if (relation is Relation.NO_COUNTERPART) != (not rec.wit):                   # V3
        return failed("V3 no_counterpart requires an empty wit list, every other relation a non-empty one")
    if rec.wit and not wit_ids:                                                  # V2
        return failed(f"V2 none of the linked handles exists: {list(rec.wit)}")
    if relation not in QUOTE_OPTIONAL and not rec.ref_quote.strip():             # V5
        return failed(f"V5 ref_quote is required for {relation.value}")
    if wit_ids and not rec.wit_quote.strip():
        return failed("V5 wit_quote is required when wit is non-empty")
    quotes = []
    for side, quote, text, lang in (("ref", rec.ref_quote, unit.text, window.ref_lang),
                                    ("wit", rec.wit_quote, _joined(wit_ids, window), window.wit_lang)):
        if not quote.strip():
            continue
        match = quote_match(quote, text, lang, lex.variants.for_lang(lang))       # V4
        if match is None:
            return failed(f"V4 {side}_quote is not verbatim in the {'unit' if side == 'ref' else 'linked segments'}")
        if match == "variant":
            flags.add(FLAG_QUOTE_VARIANT)
        quotes.append(Quote(side, quote, (unit.id,) if side == "ref" else wit_ids))
    flags |= _polarity_flags(relation, rec.polarity_flip, rec.ref_quote, rec.wit_quote, window, lex.negators)
    found: list[Diagnostic] = []
    outside = tuple(i for i in wit_ids if i not in window.core)                  # V7
    if outside:
        flags.add(FLAG_RELOCATION)
        where = sorted({window.segment[i].local_chapter or "none" for i in outside})
        found.append(Diagnostic(FLAG_RELOCATION, (unit.id,), outside,
                                detail=f"{window.key}: linked outside the core window {sorted(window.core_locals)} "
                                       f"to witness chapter(s) {where}"))
    if relation is Relation.TRANSLITERATED and transliteration_density(      # V11
            rec.wit_quote, window.wit_lang, lex.translit_charset) < MIN_TRANSLITERATION_DENSITY:
        flags.add(FLAG_TRANSLITERATION_UNSUPPORTED)
    if unit.kind != "mantra" and any(window.segment[i].kind == "mantra" for i in wit_ids):
        flags.add(FLAG_INSTRUCTION_AS_MANTRA)
    link = Link(ref_id=unit.id, wit_ids=wit_ids, relation=relation, polarity_flip=rec.polarity_flip,
                confidence=rec.confidence, quotes=tuple(quotes), flags=frozenset(flags), source=source)
    return link, found


def _polarity_flags(relation: Relation, flip: bool, ref_quote: str, wit_quote: str, window: Window,
                    negators: Negators) -> set[str]:
    """V6: consistency of relation and flag, and corroboration by a negator asymmetry."""
    flags = {FLAG_POLARITY_MISMATCH} if (relation is Relation.REVERSAL) != flip else set()
    if relation is Relation.REVERSAL or flip:
        ref_neg = _has_negator(ref_quote, window.ref_lang, negators)
        wit_neg = _has_negator(wit_quote, window.wit_lang, negators)
        flags.add(FLAG_CORROBORATED if ref_neg != wit_neg else FLAG_POLARITY_UNCORROBORATED)
    return flags


def _has_negator(text: str, lang: str, negators: Negators) -> bool:
    return bool(find_terms(text, lang, negators.for_lang(lang), negators.exclusions_for(lang)))


def _check_witness_only(handles: Sequence[str], kind: WitnessOnlyKind, quote: str, window: Window,
                        lex: CheckLexicon, source: str) -> Link | Diagnostic:
    ids, unknown = _resolve(handles, window)
    if not ids:
        return Diagnostic("witness_only_dropped", detail=f"{window.key}: no existing handle in {list(handles)}")
    if not quote.strip():
        return Diagnostic(REASON_VERIFICATION_FAILED, wit_ids=ids, detail=f"{window.key}: V5 witness-only "
                                                                          "record without wit_quote")
    match = quote_match(quote, _joined(ids, window), window.wit_lang, lex.variants.for_lang(window.wit_lang))
    if match is None:
        return Diagnostic(REASON_VERIFICATION_FAILED, wit_ids=ids,
                          detail=f"{window.key}: V4 witness-only wit_quote is not verbatim")
    flags = {FLAG_UNKNOWN_HANDLE} if unknown else set()
    if match == "variant":
        flags.add(FLAG_QUOTE_VARIANT)
    return Link(None, ids, kind, quotes=(Quote("wit", quote, ids),), flags=frozenset(flags), source=source)


def quote_match(quote: str, text: str, lang: str, variants: Mapping[str, str]) -> str | None:
    """"exact", "variant" (only after folding variant forms) or None (not verbatim)."""
    if quote_in(quote, text, lang):
        return "exact"
    if variants and quote_in(fold_variants(for_quote(quote, lang), variants),
                             fold_variants(for_quote(text, lang), variants), lang):
        return "variant"
    return None


def _resolve(handles: Sequence[str], window: Window) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(known segment ids in text order without repeats, unknown handles)."""
    known = {window.wit_handles[h] for h in handles if h in window.wit_handles}
    unknown = tuple(h for h in handles if h not in window.wit_handles)
    return tuple(sorted(known, key=window.position.__getitem__)), unknown


def _joined(ids: Sequence[str], window: Window) -> str:
    """Linked segments' text in text order; the space keeps Tibetan syllables apart."""
    return " ".join(window.segment[i].text for i in ids)


def _order_check(links: list[Link], window: Window) -> tuple[list[Link], list[Diagnostic]]:
    """V8: flag the units off a longest chain whose first linked segments never go back."""
    placed = [(i, window.position[link.wit_ids[0]]) for i, link in enumerate(links) if link.wit_ids]
    positions = [p for _, p in placed]
    pairs = sum(1 for a in range(len(positions)) for b in range(a + 1, len(positions))
                if positions[a] > positions[b])
    if not pairs:
        return links, []
    keep = {placed[j][0] for j in _longest_non_decreasing(positions)}
    off = {i for i, _ in placed} - keep
    out = [replace(link, flags=link.flags | {FLAG_CROSSING}) if i in off else link for i, link in enumerate(links)]
    ids = tuple(links[i].ref_id or "" for i in sorted(off))
    return out, [Diagnostic(FLAG_CROSSING, ids, detail=f"{window.key}: {pairs} crossing pair(s) of links")]


def _longest_non_decreasing(values: Sequence[int]) -> list[int]:
    """Indices of one longest non-decreasing subsequence (patience sorting, O(n log n))."""
    tail_values: list[int] = []    # tail_values[k]: smallest last value of a chain of length k + 1
    tail_index: list[int] = []     # the index holding that value
    back = [-1] * len(values)
    for i, v in enumerate(values):
        k = bisect_right(tail_values, v)
        back[i] = tail_index[k - 1] if k else -1
        if k == len(tail_values):
            tail_values.append(v)
            tail_index.append(i)
        else:
            tail_values[k], tail_index[k] = v, i
    chain, i = [], tail_index[-1] if tail_index else -1
    while i != -1:
        chain.append(i)
        i = back[i]
    return chain[::-1]


def reconcile_witness_only(unit_links: Sequence[Link], witness_only: Sequence[Link]) -> list[Link]:
    """Witness-only records with every segment claimed at most once.

    A segment linked by a unit is never witness-only. Among witness-only records a
    segment goes to the first record that claims it, where ``belongs_elsewhere`` records
    come after all others (a positive statement about a segment outranks "see
    elsewhere"). Trimmed records get the flag ``witness_only_trimmed``; emptied ones go.
    """
    claimed = {i for link in unit_links for i in link.wit_ids}
    ordered = sorted(witness_only, key=lambda link: link.relation is WitnessOnlyKind.BELONGS_ELSEWHERE)
    out = []
    for link in ordered:
        ids = tuple(i for i in link.wit_ids if i not in claimed)
        claimed.update(ids)
        if ids:
            trimmed = len(ids) < len(link.wit_ids)
            out.append(replace(link, wit_ids=ids, flags=link.flags | {FLAG_WITNESS_ONLY_TRIMMED})
                       if trimmed else link)
    return out


def frozen_mapping(mapping: Mapping[str, str]) -> Mapping[str, str]:
    """A read-only copy (``Collation.unresolved`` must not change after construction)."""
    return MappingProxyType(dict(mapping))
