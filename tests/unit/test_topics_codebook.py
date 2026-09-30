"""topics.codebook: the fixed vocabulary, topic groups and the codebook file (review defect #20)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.topics import (
    GROUP_OF,
    GROUP_PRECEDENCE,
    SENSITIVE_TOPICS,
    TOPICS,
    CodebookError,
    TopicError,
    load_codebook,
    topic_group,
    topic_set_error,
)

ROOT = Path(__file__).resolve().parents[2]
CODEBOOK = ROOT / "data" / "codebook" / "topics.yaml"


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(CODEBOOK.read_text(encoding="utf-8"))


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "topics.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- vocabulary
def test_vocabulary_is_tokushiges_eight_plus_three_controls():
    assert SENSITIVE_TOPICS == ("sexual", "female_agent", "flesh_food", "harm", "theft",
                                "impure_substance", "bone_corpse", "ritual")
    assert TOPICS == SENSITIVE_TOPICS + ("neutral", "frame", "mantra_control")
    assert {GROUP_OF[t] for t in SENSITIVE_TOPICS} == {"sensitive"}
    assert (GROUP_OF["neutral"], GROUP_OF["frame"], GROUP_OF["mantra_control"]) == ("neutral", "frame", "mantra_control")


@pytest.mark.parametrize("topic", SENSITIVE_TOPICS)
def test_each_of_the_eight_categories_is_sensitive(topic):
    assert topic_group({topic}) == "sensitive"


@pytest.mark.parametrize("topics, group", [
    (set(), "unlabelled"),
    ({"neutral"}, "neutral"),
    ({"frame"}, "frame"),
    ({"mantra_control"}, "mantra_control"),
    ({"frame", "harm"}, "sensitive"),                 # any of the eight makes a unit sensitive
    ({"mantra_control", "ritual"}, "sensitive"),
    ({"mantra_control", "frame"}, "mantra_control"),  # a framed mantra is still a mantra
    ({"sexual", "female_agent"}, "sensitive"),
])
def test_topic_group_follows_the_precedence(topics, group):
    assert topic_group(topics) == group


def test_precedence_puts_sensitive_first():
    assert GROUP_PRECEDENCE == ("sensitive", "mantra_control", "frame", "neutral")


@pytest.mark.parametrize("topics, message", [
    ({"neutral", "harm"}, "neutral must appear alone"),
    ({"neutral", "frame"}, "neutral must appear alone"),
    ({"violence"}, "unknown topic"),
    ({"sexual,harm"}, "separate topics with ';'"),
])
def test_invalid_topic_sets_are_rejected(topics, message):
    assert message in topic_set_error(frozenset(topics))
    with pytest.raises(TopicError, match=message):
        topic_group(topics)


def test_valid_sets_have_no_error():
    assert topic_set_error(frozenset()) is None
    assert topic_set_error(frozenset({"harm", "ritual", "frame"})) is None


# --------------------------------------------------------------------------- codebook file
def test_the_committed_codebook_loads_and_matches_the_vocabulary():
    cb = load_codebook(CODEBOOK)
    assert set(cb.names) == set(TOPICS) and len(cb.names) == len(TOPICS)
    assert cb.status in {"draft", "verified"}
    assert cb.rules and all(r.strip() for r in cb.rules)
    for t in cb.topics:
        assert t.group == GROUP_OF[t.name]
        assert t.definition and t.include and t.exclude
    counts = {t.name: t.tokushige_2026_verses for t in cb.topics}
    assert counts == {"sexual": 55, "female_agent": 54, "flesh_food": 19, "harm": 33, "theft": 5,
                      "impure_substance": 45, "bone_corpse": 8, "ritual": 52,
                      "neutral": None, "frame": None, "mantra_control": None}
    assert sum(v for v in counts.values() if v) == 271   # verses may fall in several categories


def test_codebook_definitions_never_mention_the_witness(raw):
    """The codebook is rendered into a prompt that must stay blind to the Chinese."""
    rendered = yaml.safe_dump({k: raw[k] for k in ("rules", "topics")}, allow_unicode=True).lower()
    assert "chinese" not in rendered and "t0892" not in rendered


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["topics"].pop(), "missing ['mantra_control']"),
    (lambda d: d["topics"].append({**d["topics"][0]}), "duplicated ['sexual']"),
    (lambda d: d["topics"].append({**d["topics"][0], "name": "violence"}), "extra ['violence']"),
    (lambda d: d["topics"][0].update(group="neutral"), "group must be 'sensitive'"),
    (lambda d: d["topics"][8].update(group="sensitive"), "group must be 'neutral'"),
    (lambda d: d.update(group_precedence=["sensitive", "frame", "mantra_control", "neutral"]), "group_precedence"),
    (lambda d: d["topics"][0].update(definition=" "), "definition must be a non-empty string"),
    (lambda d: d["topics"][0].update(include=[]), "include must be a non-empty list"),
    (lambda d: d["topics"][0].update(exclude=[{"a": "b"}]), "exclude must be a non-empty list"),
    (lambda d: d["topics"][0].update(colour="red"), "unknown key 'colour'"),
    (lambda d: d["topics"][0].update(tokushige_2026_verses=-1), "tokushige_2026_verses"),
    (lambda d: d.update(extra=1), "unknown key 'extra'"),
    (lambda d: d.pop("rules"), "missing key 'rules'"),
    (lambda d: d.update(status="final"), "status must be draft or verified"),
    (lambda d: d.update(status="verified", verified_by=None), "needs verified_by"),
    (lambda d: d.update(schema_version=2), "schema_version must be 1"),
    (lambda d: d.update(topics={"sexual": {}}), "topics must be a list"),
])
def test_codebook_that_disagrees_with_the_code_is_refused(tmp_path, raw, mutate, message):
    doc = copy.deepcopy(raw)
    mutate(doc)
    with pytest.raises(CodebookError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_codebook(_write(tmp_path, doc))


def test_every_codebook_problem_is_reported_at_once(tmp_path, raw):
    doc = copy.deepcopy(raw)
    doc["topics"][0]["definition"] = ""
    doc["topics"][1]["group"] = "frame"
    doc["extra"] = True
    with pytest.raises(CodebookError) as err:
        load_codebook(_write(tmp_path, doc))
    text = str(err.value)
    assert "definition" in text and "group must be 'sensitive'" in text and "unknown key 'extra'" in text


def test_a_verified_codebook_with_a_verifier_loads(tmp_path, raw):
    doc = copy.deepcopy(raw)
    doc.update(status="verified", verified_by="researcher")
    cb = load_codebook(_write(tmp_path, doc))
    assert (cb.status, cb.verified_by) == ("verified", "researcher")


def test_a_non_mapping_codebook_is_refused(tmp_path):
    path = tmp_path / "topics.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(CodebookError, match="must be a mapping"):
        load_codebook(path)
