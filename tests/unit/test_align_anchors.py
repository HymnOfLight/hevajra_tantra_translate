"""align.anchors: lexicon loading and validation, and every extraction rule.

Multilingual inputs are in data/fixtures/align_anchors.yaml; the real lexicon is
data/lexicon/anchors.yaml (with numerals.yaml).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hevajra_matrix.align.anchors import (
    LANGS,
    NUMERAL_PREFIX,
    TYPES,
    AnchorLexicon,
    extract,
    load_anchor_lexicon,
    parse_anchors,
)
from hevajra_matrix.core.lexicon import LexiconError, load_numerals

DATA = Path(__file__).resolve().parents[2] / "data"
CASES = yaml.safe_load((DATA / "fixtures" / "align_anchors.yaml").read_text(encoding="utf-8"))
LEXICON = load_anchor_lexicon(DATA)
RULES = AnchorLexicon(anchors=parse_anchors(CASES["rules"]["lexicon"]), numerals=load_numerals(DATA))


def _id(case: dict) -> str:
    return f"{case['lang']}-{case['text'][:10]}"


def _check(case: dict, found: frozenset[str]) -> None:
    if "exact" in case:
        assert found == set(case["exact"]), case.get("why", "")
    for key in case.get("present", ()):
        assert key in found
    for key in case.get("absent", ()):
        assert key not in found


# --------------------------------------------------------------------------- real lexicon
@pytest.mark.parametrize("case", CASES["real_lexicon"], ids=_id)
def test_real_lexicon_cases(case):
    _check(case, extract(case["text"], case["lang"], LEXICON))


def test_absent_keys_in_fixtures_exist_in_the_lexicon():
    """An 'absent' assertion about a key the lexicon does not define would be vacuous."""
    keys = LEXICON.keys()
    for case in CASES["real_lexicon"]:
        for key in [*case.get("absent", ()), *case.get("exact", ()), *case.get("present", ())]:
            assert key.startswith(f"{NUMERAL_PREFIX}:") or key in keys, key


def test_real_lexicon_is_well_formed():
    assert len(LEXICON.anchors) >= 50
    for anchor in LEXICON.anchors:
        kind, _, name = anchor.key.partition(":")
        assert kind in TYPES and name
        assert set(anchor.forms) == set(LANGS)
        assert all(len(form) >= 2 for form in anchor.forms["zh"])


def test_real_lexicon_forms_find_their_own_anchor():
    """Every form, read on its own, yields its anchor: no form is shadowed by another
    anchor's exclusion context or by a longer form of a different anchor."""
    for anchor in LEXICON.anchors:
        for lang, forms in anchor.forms.items():
            for form in forms:
                assert anchor.key in extract(form, lang, LEXICON), (anchor.key, lang, form)


def test_removed_one_sided_anchors_stay_removed():
    """v0.2 keys that fired in one witness only (repro #14) are not in the lexicon."""
    for key in ("term:māṃsa", "term:vidyā", "term:madya", "term:samādhi"):
        assert key not in LEXICON.keys()


# --------------------------------------------------------------------------- matching rules
@pytest.mark.parametrize("case", CASES["rules"]["cases"], ids=lambda c: c["why"])
def test_matching_rules(case):
    _check(case, extract(case["text"], case["lang"], RULES))


def test_extract_returns_a_frozenset_and_is_deterministic():
    text = CASES["real_lexicon"][0]["text"]
    first = extract(text, "bo", LEXICON)
    assert isinstance(first, frozenset)
    assert extract(text, "bo", LEXICON) == first


def test_empty_text_has_no_anchors():
    assert extract("", "bo", LEXICON) == frozenset()
    assert extract("   ", "zh", LEXICON) == frozenset()


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", CASES["invalid"], ids=lambda b: b["error"])
def test_loader_rejects_malformed_records(bad):
    with pytest.raises(LexiconError, match=bad["error"]):
        parse_anchors(bad["records"])


def test_loader_rejects_a_non_list():
    with pytest.raises(LexiconError, match="must be a list"):
        parse_anchors({"key": "vajra"})


def test_loader_rejects_unexpected_top_level_keys(tmp_path):
    (tmp_path / "lexicon").mkdir()
    for name in ("numerals.yaml",):
        (tmp_path / "lexicon" / name).write_bytes((DATA / "lexicon" / name).read_bytes())
    (tmp_path / "lexicon" / "anchors.yaml").write_text("anchors: []\nterms: []\n", encoding="utf-8")
    with pytest.raises(LexiconError, match="top-level"):
        load_anchor_lexicon(tmp_path)


def test_lexicon_equality_ignores_the_derived_index():
    again = AnchorLexicon(anchors=LEXICON.anchors, numerals=LEXICON.numerals)
    assert again == LEXICON
