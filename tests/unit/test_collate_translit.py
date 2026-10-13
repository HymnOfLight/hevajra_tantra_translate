"""core.translit (owned by the collate workstream; V11 and d_lit share it)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hevajra_matrix.core.textnorm import for_quote
from hevajra_matrix.core.translit import (
    MIN_TRANSCRIBED_RUN,
    contains_transcription,
    looks_transcribed,
    transliteration_charset,
    transliteration_density,
)

from test_collate_support import reference, seg, witness

Z_MANTRA, Z_TREES, U_MANTRA, U_SAID = "ZT:0001a05.1", "ZT:0001a03.2", "BT:1a.2.2", "BT:1a.1.1"


def test_charset_comes_from_chinese_mantra_segments_only() -> None:
    charset = transliteration_charset(witness())
    assert charset == frozenset(seg(Z_MANTRA).text)
    # a mantra segment of another language contributes nothing
    assert transliteration_charset([replace(seg(Z_MANTRA), lang="bo")]) == frozenset()
    assert transliteration_charset(reference()) == frozenset()


def test_chinese_density() -> None:
    charset = transliteration_charset(witness())
    assert transliteration_density(seg(Z_MANTRA).text, "zh", charset) == 1.0
    assert transliteration_density(seg(Z_TREES).text, "zh", charset) == 0.0
    mixed = seg(Z_MANTRA).text + seg(Z_TREES).text          # punctuation is ignored
    n_mantra, n_other = len(for_quote(seg(Z_MANTRA).text, "zh")), len(for_quote(seg(Z_TREES).text, "zh"))
    assert transliteration_density(mixed, "zh", charset) == pytest.approx(n_mantra / (n_mantra + n_other))
    assert transliteration_density("", "zh", charset) == 0.0


def test_tibetan_density_uses_sanskrit_only_letters() -> None:
    assert transliteration_density(seg(U_MANTRA).text, "bo") == 1.0
    assert transliteration_density(seg(U_SAID).text, "bo") == 0.0


def test_other_languages_have_no_transliteration_density() -> None:
    assert transliteration_density("om ah hum", "sa") == 0.0


def test_contains_transcription_finds_a_transcribed_stretch_among_other_words() -> None:
    charset = transliteration_charset(witness())
    prose, mantra = seg(Z_TREES).text, for_quote(seg(Z_MANTRA).text, "zh")
    assert not contains_transcription(prose, "zh", charset)
    padded = prose + mantra[:MIN_TRANSCRIBED_RUN] + prose
    assert not looks_transcribed(padded, "zh", charset)
    assert contains_transcription(padded, "zh", charset)
    # transcription characters apart from each other do not make a transcription
    apart = mantra[0] + prose + mantra[1]
    assert not contains_transcription(apart, "zh", charset)
    assert contains_transcription(mantra, "zh", charset)                   # mostly transcription
    assert contains_transcription(mantra[0], "zh", charset)


def test_contains_transcription_elsewhere_is_looks_transcribed() -> None:
    for text in (seg(U_MANTRA).text, seg(U_SAID).text, seg(U_SAID).text + seg(U_MANTRA).text):
        assert contains_transcription(text, "bo") is looks_transcribed(text, "bo")
    assert contains_transcription(seg(U_MANTRA).text, "bo")
    assert not contains_transcription("om ah hum", "sa")
