"""Unit tests of hevajra_matrix.ingest.notes and the rules in data/lexicon/notes_zh.yaml.

The note texts are in data/fixtures/ingest_notes_zh.yaml: every distinct inline note of
T0892 with its expected class, and near misses that must stay unclassified.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hevajra_matrix.core.lexicon import LexiconError
from hevajra_matrix.ingest import notes

DATA = Path(__file__).resolve().parents[2] / "data"
CASES = yaml.safe_load((DATA / "fixtures" / "ingest_notes_zh.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules() -> notes.NoteRules:
    return notes.load_rules(DATA)


def test_rule_file_covers_every_class_and_has_a_marker(rules):
    assert {r.note_class for r in rules.rules} == set(notes.NOTE_CLASSES)
    assert rules.tokyo335_marker


@pytest.mark.parametrize("case", CASES["classified"], ids=lambda c: c["class"])
def test_each_real_note_text_has_exactly_one_matching_rule(rules, case):
    key = notes.normalise_note(case["text"])
    matching = [r.note_class for r in rules.rules if r.pattern.fullmatch(key)]
    assert matching == [case["class"]]
    assert notes.classify(case["text"], rules) == case["class"]


def test_fixture_covers_the_48_distinct_notes_of_t0892():
    assert len(CASES["classified"]) == 48
    assert {c["class"] for c in CASES["classified"]} == set(notes.NOTE_CLASSES)


@pytest.mark.parametrize("text", CASES["unclassified"])
def test_near_misses_stay_unclassified(rules, text):
    assert notes.classify(text, rules) == notes.UNCLASSIFIED


def test_whitespace_inside_a_note_is_ignored(rules):
    assert notes.normalise_note(" a\nb\u3000c ") == "abc"      # ideographic space counts as whitespace


def test_footnote_source_depends_only_on_the_marker(rules):
    assert notes.footnote_source("viyoga. " + rules.tokyo335_marker, rules) == notes.FOOTNOTE_TOKYO335
    assert notes.footnote_source("viyoga.", rules) == notes.FOOTNOTE_TAISHO


def test_first_matching_rule_wins(tmp_path):
    write_rules(tmp_path, 'rules:\n  - {class: title, pattern: "a.*"}\n  - {class: fanqie, pattern: ".*b"}\n'
                          'tokyo335_marker: "M"\n')
    rules = notes.load_rules(tmp_path)
    assert notes.classify("ab", rules) == "title"
    assert notes.classify("cb", rules) == "fanqie"
    assert notes.classify("abc", rules) == "title"
    assert notes.classify("c", rules) == notes.UNCLASSIFIED


@pytest.mark.parametrize("doc, message", [
    ('rules:\n  - {class: gloss, pattern: "a"}\ntokyo335_marker: "M"\n', "class"),
    ('rules:\n  - {class: title, pattern: "("}\ntokyo335_marker: "M"\n', "bad pattern"),
    ('rules:\n  - {class: title, pattern: "a", note: x}\ntokyo335_marker: "M"\n', "class"),
    ('rules:\n  - {class: title, pattern: "a"}\n', "tokyo335_marker"),
    ('rules: []\ntokyo335_marker: "M"\n', "rules"),
    ('rules:\n  - {class: title, pattern: "a"}\ntokyo335_marker: "M"\nextra: 1\n', "unknown key"),
])
def test_malformed_rule_files_are_rejected(tmp_path, doc, message):
    write_rules(tmp_path, doc)
    with pytest.raises(LexiconError, match=message):
        notes.load_rules(tmp_path)


def write_rules(data_dir: Path, text: str) -> None:
    (data_dir / "lexicon").mkdir(exist_ok=True)
    (data_dir / "lexicon" / "notes_zh.yaml").write_text(text, encoding="utf-8")
