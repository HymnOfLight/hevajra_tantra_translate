"""core.translit (owned by the collate workstream; V11 and d_lit share it)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hevajra_matrix.core.textnorm import for_quote
from hevajra_matrix.core.translit import transliteration_charset, transliteration_density

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
