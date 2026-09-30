"""Transliteration density: how much of a stretch of text is phonetic transcription.

Two consumers use the same measure, so they cannot drift apart:

    collate.verify (V11)   a ``transliterated`` link needs a witness quote that is mostly
                           transcription; otherwise it is flagged for review
    matrix (d_lit)         the descriptive transliteration density of a cell

Chinese transcription has no script of its own: it reuses ordinary characters chosen for
their sound. The character set is therefore *derived from the witness itself*: the
characters of its mantra (dharani) segments, which are transcription by construction.
Tibetan transcription of Sanskrit is recognisable by letters that native words do not
use, so Tibetan needs no character set (``core.textnorm.bo_sanskrit_syllable_ratio``).

Why the Taisho glosses are NOT part of the set
    The design first proposed adding the Chinese lemmas of the Taisho glosses. On T0892
    (checked 2026-09-30) most of those lemmas are translations, not transcriptions (e.g.
    the terms for vajra, compassion, blood vessel), and adding their characters raised
    the share of prose/verse segments with density >= 0.5 from 134 to about 1,150 of 2,533,
    which would make the V11 check meaningless. Mantra segments alone give a mean prose
    density of 0.10.
"""

from __future__ import annotations

from typing import Iterable

from .textnorm import bo_sanskrit_syllable_ratio, for_quote
from .types import Segment

MANTRA_KIND = "mantra"


def transliteration_charset(witness_segments: Iterable[Segment]) -> frozenset[str]:
    """Characters of the Chinese mantra segments (punctuation and spaces excluded).

    Segments of other kinds or languages contribute nothing, so passing a whole witness
    text is the intended use.
    """
    return frozenset(
        ch
        for seg in witness_segments
        if seg.kind == MANTRA_KIND and seg.lang == "zh"
        for ch in for_quote(seg.text, "zh")
    )


def transliteration_density(text: str, lang: str, charset: frozenset[str] = frozenset()) -> float:
    """Share of ``text`` that is transcription, in [0, 1]; 0.0 for empty text.

    zh     share of characters (punctuation and whitespace ignored) in ``charset``
    bo     share of syllables containing a Sanskrit-only letter
    other  0.0 (Sanskrit is the source language, not a transcription of it)
    """
    if lang == "zh":
        key = for_quote(text, "zh")
        return sum(ch in charset for ch in key) / len(key) if key else 0.0
    if lang == "bo":
        return bo_sanskrit_syllable_ratio(text)
    return 0.0
