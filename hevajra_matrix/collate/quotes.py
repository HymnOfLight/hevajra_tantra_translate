"""Verbatim quotes in the code checks: V4 and C3 (``quote_match``), V12's glossary forms
(``term_in``) and the quote the negator checks V6 and C5 read (``completed_quote``).

A Tibetan quote starts at a syllable and ends at one, or where the letters of a particle
fused to its last syllable begin: -s, -r, -'i, -'o, -'ang, -'am (the unit's -por quoted as
-po). The test is on letters alone, so a final -s or -r is dropped whatever it is: la for
las, na for nas and rnam for rnams pass, because spelling cannot tell a fused particle from
a final letter. Any other letters left out of a syllable fail. After a final a-chung the
particle's own a-chung is the quote's (nam mkha' in nam mkha'i); where -r or -s replaces the
a-chung (sems dpar) nothing is restored: the quote stops where the text's letters stop
(sems dp), never at the citation form (sems dpa'), and dka' never matches dkar. The only
consumer that reads meaning from a Tibetan quote, the negator check, reads the quote
completed to whole syllables of its text (its own punctuation kept, so a two-syllable
negator such as ma yin is never read across a shad of the quote).
"""

from __future__ import annotations

import re
from typing import Mapping

from ..core.textnorm import TSHEG, chunks, fold_variants, for_quote, quote_in

_BO_FUSED_PARTICLES = ("\u0f66", "\u0f62", "\u0f60\u0f72", "\u0f60\u0f7c", "\u0f60\u0f44", "\u0f60\u0f58")
_A_CHUNG = "\u0f60"
_SHAD = "\u0f0d"
_BO_FUSED_SUFFIX = "(" + "|".join(_BO_FUSED_PARTICLES) + ")"
_BO_FUSED_AFTER_A_CHUNG = "(" + "|".join(            # the particle's a-chung is the term's own
    _BO_FUSED_PARTICLES + tuple(p[1:] for p in _BO_FUSED_PARTICLES if p.startswith(_A_CHUNG))) + ")"


def quote_match(quote: str, text: str, lang: str, variants: Mapping[str, str]) -> str | None:
    """"exact", "variant" (only after folding variant forms) or None (not verbatim); a
    Tibetan quote may stop where a fused particle begins (module docstring)."""
    q, t = for_quote(quote, lang), for_quote(text, lang)
    if _verbatim(q, t, lang):
        return "exact"
    if variants and _verbatim(fold_variants(q, variants), fold_variants(t, variants), lang):
        return "variant"
    return None


def completed_quote(quote: str, text: str, lang: str, variants: Mapping[str, str]) -> str:
    """``quote`` as the negator checks read it (V6, C5): a Tibetan quote that ``quote_match``
    let stop where a fused particle begins, its last syllable completed with that particle's
    letters (mi cut from mi'am reads mi'am, which negates nothing) and its shads kept, so no
    term is read across one; any other quote unchanged."""
    if lang != "bo":
        return quote
    q, t = for_quote(quote, lang), for_quote(text, lang)
    tail = _fused_tail(q, t)
    if tail is None and variants:
        tail = _fused_tail(fold_variants(q, variants), fold_variants(t, variants))
    if not tail:
        return quote
    parts = chunks(quote, lang)
    parts[-1][-1] += tail
    return _SHAD.join(TSHEG.join(tokens) for tokens in parts)


def term_in(term: str, key: str, lang: str) -> bool:
    """A Tibetan term (a quote key) starts at a syllable of ``key`` and ends at one, or where
    a fused particle begins on its last syllable (the Derge text writes a name with -'i, -s
    and -r); elsewhere a substring."""
    if lang != "bo":
        return bool(term) and term in key
    return _fused_tail(term, key) is not None


def _verbatim(quote_key: str, text_key: str, lang: str) -> bool:
    return quote_in(quote_key, text_key, lang) or (lang == "bo" and term_in(quote_key, text_key, lang))


def _fused_tail(term: str, key: str) -> str | None:
    """What ``key`` adds after a Tibetan ``term`` that starts at a syllable (both quote
    keys): "" where the term ends at a syllable boundary, else the letters of the fused
    particle where it ends as one begins (the first such place); None when neither occurs."""
    if not term:
        return None
    if f"{TSHEG}{term}{TSHEG}" in f"{TSHEG}{key}{TSHEG}":
        return ""
    suffix = _BO_FUSED_AFTER_A_CHUNG if term.endswith(_A_CHUNG) else _BO_FUSED_SUFFIX
    found = re.search(re.escape(f"{TSHEG}{term}") + suffix + TSHEG, f"{TSHEG}{key}{TSHEG}")
    return found.group(1) if found else None
