"""core.textnorm: quote keys, fingerprints, lengths, numerals, term matching, variant folding.

Multilingual inputs live in data/fixtures/core_textnorm.yaml and core_numerals.yaml; the
numeral, negator and variant tables are the real files in data/lexicon/.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.core import textnorm
from hevajra_matrix.core.lexicon import load_negators, load_numerals, load_variant_forms
from hevajra_matrix.core.textnorm import (
    bo_sanskrit_syllable_ratio,
    bo_syllables,
    chunks,
    find_terms,
    fingerprint,
    fold_variants,
    for_quote,
    length,
    numerals,
    quote_in,
    zh_parse_numeral,
)

DATA = Path(__file__).resolve().parents[2] / "data"
CASES = yaml.safe_load((DATA / "fixtures" / "core_textnorm.yaml").read_text(encoding="utf-8"))
NUMERAL_CASES = yaml.safe_load((DATA / "fixtures" / "core_numerals.yaml").read_text(encoding="utf-8"))["cases"]
NUMERALS = load_numerals(DATA)
VARIANTS = load_variant_forms(DATA)


def _id(case: dict) -> str:
    return f"{case.get('lang', 'bo')}-{(case.get('text') or case.get('quote') or case.get('a') or '')[:12]}"


# --------------------------------------------------------------------------- for_quote
@pytest.mark.parametrize("case", CASES["for_quote"], ids=_id)
def test_for_quote(case):
    assert for_quote(case["text"], case["lang"]) == case["expected"]


@pytest.mark.parametrize("case", CASES["for_quote"], ids=_id)
def test_for_quote_is_idempotent_and_ignores_normalisation_form(case):
    key = for_quote(case["text"], case["lang"])
    assert for_quote(key, case["lang"]) == key
    assert for_quote(unicodedata.normalize("NFD", case["text"]), case["lang"]) == key


@pytest.mark.parametrize("case", CASES["quote_contained"], ids=_id)
def test_quote_in(case):
    assert quote_in(case["quote"], case["text"], case["lang"]) is case["contained"]


def test_quote_in_after_variant_folding():
    table = VARIANTS.for_lang("zh")
    quote, text = "\u6301\u9ad1\u9ac5\u5668", "\u6301\u9ad1\u9acf\u5668"   # skull cup, simplified vs traditional
    assert not quote_in(quote, text, "zh")
    assert quote_in(fold_variants(for_quote(quote, "zh"), table), fold_variants(for_quote(text, "zh"), table), "zh")


# --------------------------------------------------------------------------- fingerprint
@pytest.mark.parametrize("case", CASES["fingerprint_same"], ids=_id)
def test_fingerprint_ignores_punctuation_and_whitespace(case):
    assert fingerprint(case["a"], case["lang"]) == fingerprint(case["b"], case["lang"])


@pytest.mark.parametrize("case", CASES["fingerprint_differs"], ids=_id)
def test_fingerprint_changes_with_the_text(case):
    assert fingerprint(case["a"], case["lang"]) != fingerprint(case["b"], case["lang"])


def test_fingerprint_definition():
    text = "evam maya srutam"
    fp = fingerprint(text, "sa")
    assert len(fp) == 12 and int(fp, 16) >= 0
    assert fp == hashlib.sha1(for_quote(text, "sa").encode("utf-8")).hexdigest()[:12]
    assert fingerprint("Evam maya, srutam.", "sa") == fp
    assert fingerprint("evam maya srutam iti", "sa") != fp


def test_fingerprint_depends_on_language():
    # Tibetan keeps a tsheg between syllables, other languages join tokens directly.
    text = "\u0f62\u0fa1\u0f7c\u0f0b\u0f62\u0f97\u0f7a"   # rdo rje
    assert fingerprint(text, "bo") != fingerprint(text, "zh")


# --------------------------------------------------------------------------- lengths
@pytest.mark.parametrize("case", CASES["length"], ids=_id)
def test_length(case):
    assert length(case["text"], case["lang"]) == case["expected"]


@pytest.mark.parametrize("case", CASES["bo_syllables"], ids=_id)
def test_bo_syllables(case):
    assert bo_syllables(case["text"]) == case["expected"]


@pytest.mark.parametrize("case", CASES["bo_sanskrit_ratio"], ids=_id)
def test_bo_sanskrit_syllable_ratio(case):
    assert case["low"] <= bo_sanskrit_syllable_ratio(case["text"]) <= case["high"]


def test_chunks_split_at_punctuation_and_drop_format_characters():
    assert chunks("rather, than", "en") == [["rather"], ["than"]]
    assert chunks("soft\u00adhyphen zero\u200bwidth", "en") == [["softhyphen", "zerowidth"]]
    assert chunks("", "zh") == []
    # Tibetan: tsheg separates syllables inside a chunk, shad ends the chunk.
    tsheg, shad, ka, kha = "\u0f0b", "\u0f0d", "\u0f40", "\u0f41"
    assert chunks(f"{ka}{tsheg}{kha}{shad} {ka}", "bo") == [[ka, kha], [ka]]


# --------------------------------------------------------------------------- numerals
@pytest.mark.parametrize("case", NUMERAL_CASES, ids=_id)
def test_numerals(case):
    assert numerals(case["text"], case["lang"], NUMERALS) == set(case["expected"])


def test_numerals_never_report_one_or_zero():
    for case in NUMERAL_CASES:
        assert all(v >= textnorm.MIN_NUMERAL for v in numerals(case["text"], case["lang"], NUMERALS))


def test_zh_parse_numeral_rejects_non_numerals():
    assert zh_parse_numeral("", NUMERALS.zh) is None
    assert zh_parse_numeral("abc", NUMERALS.zh) is None


# --------------------------------------------------------------------------- terms
@pytest.mark.parametrize("case", CASES["find_terms"], ids=_id)
def test_find_terms(case):
    found = find_terms(case["text"], case["lang"], case["terms"], case["exclusions"])
    assert found == set(case["expected"])


def test_find_terms_rejects_terms_with_punctuation():
    with pytest.raises(ValueError):
        find_terms("rather than", "en", ["rather, than"])
    with pytest.raises(ValueError):
        find_terms("text", "en", ["..."])


def test_real_negator_list_on_the_review_denial_cases():
    """Repro #1: a denial of censorship must be recognised as negated."""
    neg = load_negators(DATA)
    assert find_terms("This is not censorship.", "en", neg.for_lang("en"), neg.exclusions_for("en")) == {"not"}
    assert find_terms("No doubt it was censored.", "en", neg.for_lang("en"), neg.exclusions_for("en")) == set()


# --------------------------------------------------------------------------- variant folding
@pytest.mark.parametrize("case", CASES["fold_variants"], ids=_id)
def test_fold_variants(case):
    table = VARIANTS.for_lang(case["lang"])
    folded = fold_variants(case["text"], table)
    assert folded == case["expected"]
    assert fold_variants(folded, table) == folded


def test_fold_variants_with_an_empty_table_is_identity():
    assert fold_variants("abc", {}) == "abc"


def test_fold_variants_prefers_the_longest_form():
    assert fold_variants("ab-a", {"a": "x", "ab": "y"}) == "y-x"
