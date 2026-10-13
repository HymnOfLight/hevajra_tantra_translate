"""Transliteration density: how much of a stretch of text is phonetic transcription.

V11, V12 and C4 share the density measure and the threshold ``MIN_TRANSLITERATION_DENSITY``
(V11 and V12 add the ``MIN_TRANSCRIBED_RUN`` alternative); d_lit reports the density itself:

    collate.verify (V11)   a ``transliterated`` link needs a witness quote that contains a
                           transcription (``contains_transcription``); otherwise it is
                           flagged for review
    collate.verify (V12)   a ``transcribed_name`` convention needs the same of its quote
    components (C4)        a ``lit`` slot code needs a quote that is mostly transcription
                           (the whole-quote density, as in ``looks_transcribed``)
    matrix (d_lit)         the descriptive transliteration density of a cell

V11 and V12 ask whether a quote *contains* a transcription because one witness quote also
shows the rest of the unit: an ordinal before a channel name, the gloss after a seed
syllable, "how is ... called" around a deity name. On the I.1 run of collate.v2
(2026-10-13) the whole-quote density put 12 of 34 claimed transcribed names below one half,
each quoted together with such words (10 links flagged ``convention_unsupported``; the
other 2 were on units V4 had rejected).

C4 knowingly keeps the whole-quote density: its quote is the coded word itself. The test
also rejects a bare, fully transcribed word whose characters are missing from the mantra
charset, a known C4 false failure (fatal for the pair). Bare quotes in the collate.v2
answers on I.1: samayasattva at density 0.40 (no T0892 mantra holds the three characters
of samaya), yogini at 0.33 and samadhi at 0.00. Measure it on double-coded dev pairs before
``d_comp`` is used. The run rule would rescue only samayasattva (through the two characters
of sattva) and would accept under a fifth of six-character prose stretches as well.

For a witness language other than Chinese or Tibetan the density is always 0, so V11 and
V12 always flag there.

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
MIN_TRANSLITERATION_DENSITY = 0.5   # share of a quote that must be transcription (V11, V12, C4)
MIN_TRANSCRIBED_RUN = 2             # adjacent Chinese transcription characters that make a transcription (V11, V12)


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


def looks_transcribed(text: str, lang: str, charset: frozenset[str] = frozenset()) -> bool:
    """Whether ``text`` is mostly phonetic transcription (density >= the shared threshold)."""
    return transliteration_density(text, lang, charset) >= MIN_TRANSLITERATION_DENSITY


def contains_transcription(text: str, lang: str, charset: frozenset[str] = frozenset()) -> bool:
    """Whether ``text`` contains a transcription: it looks transcribed as a whole or, in
    Chinese, ``MIN_TRANSCRIBED_RUN`` characters of ``charset`` stand together.

    A single transcription character among other words does not count: about one prose
    character in ten is in the charset by chance (mean prose density 0.10), while two in a
    row occur in under a fifth of six-character prose stretches of T0892. A two-character
    quote with one transcription character still passes, on density (exactly one half).
    """
    if looks_transcribed(text, lang, charset):
        return True
    if lang != "zh":
        return False
    run = 0
    for ch in for_quote(text, "zh"):
        run = run + 1 if ch in charset else 0
        if run >= MIN_TRANSCRIBED_RUN:
            return True
    return False
