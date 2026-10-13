"""V4 on Tibetan quotes: a quote may leave out a particle fused to its last syllable.

The cases are in data/fixtures/collate_quotes.yaml (short quotes from chapter I.1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate.verify import quote_match
from hevajra_matrix.core.textnorm import quote_in

CASES = yaml.safe_load((Path(__file__).resolve().parents[2] / "data" / "fixtures" / "collate_quotes.yaml")
                       .read_text(encoding="utf-8"))["quote_match_bo"]


@pytest.mark.parametrize("case", CASES, ids=[c["why"] for c in CASES])
def test_v4_tibetan_quote_boundaries(case: dict) -> None:
    assert quote_match(case["quote"], case["text"], case["lang"], {}) == case["match"]


def test_the_fused_particle_rule_is_v4s_not_quote_ins() -> None:
    dropped = [c for c in CASES if c["match"] == "exact" and not quote_in(c["quote"], c["text"], c["lang"])]
    assert len(dropped) == 5, "core.textnorm.quote_in stays strict about whole syllables"


def test_variant_folding_keeps_the_particle_rule() -> None:
    case = CASES[0]
    first = case["quote"][0]
    table = {first: "\u0f00"}
    folded_text = case["text"].replace(first, "\u0f00")
    assert quote_match(case["quote"], folded_text, "bo", table) == "variant"
