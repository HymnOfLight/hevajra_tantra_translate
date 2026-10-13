"""V4 on Tibetan quotes: a quote may stop where the letters of a particle fused to its
last syllable begin (a test on letters alone: a final -s or -r is dropped whatever it is).

The cases are in data/fixtures/collate_quotes.yaml (short quotes from the Derge text).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate.quotes import completed_quote, quote_match
from hevajra_matrix.core.lexicon import load_negators
from hevajra_matrix.core.textnorm import find_terms, for_quote, quote_in

DATA = Path(__file__).resolve().parents[2] / "data"
FIXTURE = yaml.safe_load((DATA / "fixtures" / "collate_quotes.yaml").read_text(encoding="utf-8"))
CASES = FIXTURE["quote_match_bo"]


@pytest.mark.parametrize("case", CASES, ids=[c["why"] for c in CASES])
def test_v4_tibetan_quote_boundaries(case: dict) -> None:
    assert quote_match(case["quote"], case["text"], case["lang"], {}) == case["match"]


def test_the_fused_particle_rule_is_v4s_not_quote_ins() -> None:
    dropped = [c for c in CASES if c["match"] == "exact" and not quote_in(c["quote"], c["text"], c["lang"])]
    assert len(dropped) == 10, "core.textnorm.quote_in stays strict about whole syllables"


def test_variant_folding_keeps_the_particle_rule() -> None:
    case = CASES[0]
    first = case["quote"][0]
    table = {first: "\u0f00"}
    folded_text = case["text"].replace(first, "\u0f00")
    assert quote_match(case["quote"], folded_text, "bo", table) == "variant"


def test_a_completed_quote_keeps_its_shads() -> None:
    """The negator check reads no term across a shad of the quote (ma | yin is not ma yin)."""
    case = FIXTURE["negator_across_shad"]
    negators = load_negators(DATA)
    whole = completed_quote(case["quote"], case["text"], "bo", {})
    assert quote_match(case["quote"], case["text"], "bo", {}) == "exact"
    assert for_quote(whole, "bo") == for_quote(case["quote"], "bo") + "\u0f62"
    assert not find_terms(whole, "bo", negators.for_lang("bo"), negators.exclusions_for("bo"))
