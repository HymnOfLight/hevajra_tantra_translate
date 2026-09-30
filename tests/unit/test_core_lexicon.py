"""core.lexicon: the real lexicon files load and validate; malformed files are rejected."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from hevajra_matrix.core.lexicon import (
    LexiconError,
    lexicon_path,
    load_negators,
    load_numerals,
    load_variant_forms,
    read_lexicon,
)
from hevajra_matrix.core.textnorm import chunks, fold_variants

DATA = Path(__file__).resolve().parents[2] / "data"


def _write(data_dir: Path, name: str, doc: object) -> Path:
    path = data_dir / "lexicon" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return data_dir


# --------------------------------------------------------------------------- real files
def test_real_numeral_table_loads_and_every_form_is_one_token_sequence():
    table = load_numerals(DATA)
    assert table.zh.digits and table.zh.units and table.bo.words and table.sa.stems
    assert table.bo.exclusions and table.zh.exclusions and table.sa.exclusions
    forms = [
        *((f, "zh") for f in [*table.zh.digits, *table.zh.units, *table.zh.exclusions]),
        *((f, "bo") for f in [*table.bo.words, *table.bo.teen_prefixes, *table.bo.connectors, *table.bo.exclusions]),
        *((f, "sa") for f in [*table.sa.stems, *table.sa.exclusions]),
    ]
    for form, lang in forms:
        assert len(chunks(form, lang)) == 1, (form, lang)
    for form in table.sa.stems:
        assert len(chunks(form, "sa")[0]) == 1, form   # stems are single words


def test_real_numeral_values_are_sane():
    table = load_numerals(DATA)
    assert sorted(set(table.zh.digits.values())) == list(range(10))
    assert all(v >= 10 for v in table.zh.units.values())
    assert {v for v in table.bo.words.values() if v < 10} == set(range(1, 10))


def test_real_negators_load():
    neg = load_negators(DATA)
    for lang in ("zh", "bo", "sa", "en"):
        assert neg.for_lang(lang), lang
        assert all(isinstance(f, str) for f in neg.for_lang(lang))
    assert neg.for_lang("xx") == () and neg.exclusions_for("xx") == ()


def test_real_variant_forms_fold_idempotently():
    variants = load_variant_forms(DATA)
    assert variants.for_lang("zh")
    for lang, table in variants.forms.items():
        assert not set(table) & set(table.values()), lang
        for form in table:
            once = fold_variants(form, table)
            assert fold_variants(once, table) == once
    assert dict(variants.for_lang("xx")) == {}


# --------------------------------------------------------------------------- read_lexicon
def test_read_lexicon_is_deeply_read_only_and_nfc(tmp_path):
    decomposed = unicodedata.normalize("NFD", "\u1e63a\u1e6d")   # IAST sat, decomposed
    _write(tmp_path, "demo", {"items": [{"form": decomposed}], "nested": {"k": [1, 2]}})
    doc = read_lexicon(tmp_path, "demo")
    assert isinstance(doc, MappingProxyType)
    assert doc["items"][0]["form"] == "\u1e63a\u1e6d"
    assert doc["nested"]["k"] == (1, 2)
    with pytest.raises(TypeError):
        doc["new"] = 1
    with pytest.raises(TypeError):
        doc["nested"]["k"] = ()


@pytest.mark.parametrize("name", ["../numerals", "Numerals", "num-erals", "", "a/b"])
def test_lexicon_names_are_plain(name):
    with pytest.raises(ValueError):
        lexicon_path(DATA, name)


def test_missing_file_and_non_mapping(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_lexicon(tmp_path, "absent")
    _write(tmp_path, "listdoc", [1, 2])
    with pytest.raises(LexiconError):
        read_lexicon(tmp_path, "listdoc")


# --------------------------------------------------------------------------- numerals validation
def _numerals_doc() -> dict:
    return {
        "zh": {"digits": [{"form": "a", "value": 2}], "units": [{"form": "b", "value": 10}], "exclusions": []},
        "bo": {"words": [{"form": "c", "value": 3}], "teen_prefixes": ["d"], "connectors": ["e"], "exclusions": []},
        "sa": {"stems": [{"form": "Dvi", "value": 2}], "exclusions": ["Lak"]},
    }


def test_minimal_numeral_table(tmp_path):
    table = load_numerals(_write(tmp_path, "numerals", _numerals_doc()))
    assert dict(table.sa.stems) == {"dvi": 2}           # Latin-script forms are lower-cased
    assert table.sa.exclusions == ("lak",)
    assert table.bo.teen_prefixes == frozenset({"d"})


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(xx={}),                                        # unknown language
    lambda d: d["zh"].update(extra=[]),                               # unknown key
    lambda d: d.pop("bo"),                                            # missing section
    lambda d: d["zh"]["digits"].append({"form": "ab", "value": 3}),   # zh digits are single characters
    lambda d: d["zh"]["digits"].append({"form": "b", "value": 3}),    # both digit and unit
    lambda d: d["zh"]["digits"].append({"form": "a", "value": 4}),    # duplicate form
    lambda d: d["bo"]["words"].append({"form": "x", "value": True}),  # YAML boolean, not a number
    lambda d: d["bo"]["words"].append({"form": "x", "value": -1}),
    lambda d: d["bo"]["words"].append({"form": "x"}),                 # missing field
    lambda d: d["bo"]["words"].append({"form": "", "value": 5}),
    lambda d: d["sa"].update(exclusions=["lak", "lak"]),              # duplicate
    lambda d: d["sa"].update(exclusions=[False]),                     # e.g. an unquoted YAML "no"
    lambda d: d["sa"].update(stems={"form": "dvi", "value": 2}),      # mapping instead of list
    lambda d: d["sa"]["stems"].append({"form": "two words", "value": 2}),  # stems are single words
    lambda d: d["bo"].update(exclusions=["x, y"]),                    # punctuation inside a context
    lambda d: d["bo"].update(connectors=["x y"]),                     # connectors are one syllable
])
def test_malformed_numeral_tables_are_rejected(tmp_path, mutate):
    doc = _numerals_doc()
    mutate(doc)
    with pytest.raises(LexiconError):
        load_numerals(_write(tmp_path, "numerals", doc))


# --------------------------------------------------------------------------- negators and variants
def test_negators_validation(tmp_path):
    neg = load_negators(_write(tmp_path, "negators", {"forms": {"en": ["Not"]}, "exclusions": {}}))
    assert neg.for_lang("en") == ("not",)
    with pytest.raises(LexiconError):
        load_negators(_write(tmp_path, "negators", {"forms": ["not"]}))
    with pytest.raises(LexiconError):
        load_negators(_write(tmp_path, "negators", {"forms": {"en": ["not"]}, "other": {}}))
    with pytest.raises(LexiconError):
        load_negators(_write(tmp_path, "negators", {"forms": {"en": ["n't"]}}))   # apostrophe is punctuation


@pytest.mark.parametrize("records", [
    [{"form": "a", "canonical": "b"}, {"form": "b", "canonical": "c"}],   # chain: not idempotent
    [{"form": "a", "canonical": "a"}],                                   # identity
    [{"form": "a", "canonical": "b"}, {"form": "a", "canonical": "c"}],   # two representatives
    [{"form": "a"}],                                                      # missing field
    {"form": "a", "canonical": "b"},                                      # mapping instead of list
])
def test_malformed_variant_tables_are_rejected(tmp_path, records):
    with pytest.raises(LexiconError):
        load_variant_forms(_write(tmp_path, "variant_forms", {"zh": records}))


def test_variant_forms_many_to_one(tmp_path):
    variants = load_variant_forms(_write(tmp_path, "variant_forms", {
        "zh": [{"form": "a", "canonical": "c"}, {"form": "b", "canonical": "c"}], "bo": [],
    }))
    assert dict(variants.for_lang("zh")) == {"a": "c", "b": "c"}
    assert dict(variants.for_lang("bo")) == {}
