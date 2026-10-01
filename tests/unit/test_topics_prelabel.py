"""topics.prelabel: T3 requests (reference text only) and verification of the answers.

Multilingual inputs come from data/fixtures/topics_units.yaml and the mini CBETA / Derge
fixtures; every answer goes through ``llm.fake.FakeClient``.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.config import ConfigError, load_settings
from hevajra_matrix.core.types import Segment
from hevajra_matrix.ingest import CONTENT_KINDS, cbeta, derge
from hevajra_matrix.llm.client import sha256_text
from hevajra_matrix.llm.fake import (
    FakeClient,
    invalid_response,
    refusal_response,
    substituted_response,
    truncated_response,
)
from hevajra_matrix.topics import (
    FLAG_DUPLICATE_REF,
    FLAG_NEUTRAL_NOT_ALONE,
    TOPICS,
    TopicTaskSettings,
    build_request,
    load_codebook,
    parse_prelabels,
    plan_batches,
    prelabel_requests,
    render_body,
    render_system,
)
from hevajra_matrix.topics.prelabel import AFTER, BEFORE, UNITS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURE = yaml.safe_load((DATA / "fixtures" / "topics_units.yaml").read_text(encoding="utf-8"))
CJK = re.compile("[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]")   # Chinese script: the witness


def _segments(records, witness: str, lang: str) -> list[Segment]:
    return [Segment(r["id"], witness, lang, r["text"], r["id"].split(":")[1], r["id"].split(":")[1], r["kind"])
            for r in records]


REF = _segments(FIXTURE["reference_units"], FIXTURE["reference_witness"], "bo")
WIT = _segments(FIXTURE["witness_units"], FIXTURE["witness"], "zh")
EXPECTED = {r["id"]: (r["topic"], r["cue"]) for r in FIXTURE["reference_units"]}
SKULL, CHARNEL, KILL = (FIXTURE["cues"][k] for k in ("skull", "charnel_ground", "kill"))  # cues for REF[1]


@pytest.fixture(scope="module")
def cb():
    return load_codebook(DATA / "codebook" / "topics.yaml")


@pytest.fixture(scope="module")
def settings():
    return TopicTaskSettings.from_config(load_settings(ROOT).llm)


def _units_in(body: str) -> list[tuple[str, str]]:
    """(handle, text) of the lines under [units to label]."""
    section = body.split(UNITS + "\n", 1)[1].split("\n\n", 1)[0]
    return [(line.split("\t")[0], line.split("\t")[2]) for line in section.splitlines()]


def _echo(lookup):
    """A scripted pre-labeller answering ``lookup[text] -> [(topic, cue), ...]`` per unit."""
    def script(request):
        return {"units": [{"ref": h, "topics": [{"topic": t, "cue": c} for t, c in lookup(text)]}
                          for h, text in _units_in(request.body)]}
    return script


def _one(topics, unit_index: int = 1):
    """A batch holding one fixture unit, and an answer giving it ``topics``."""
    batch = plan_batches([REF[unit_index]], 40, 0)[0]
    return batch, {"units": [{"ref": "u01", "topics": [{"topic": t, "cue": c} for t, c in topics]}]}


def _parse(cb, settings, batch, answer):
    request = build_request(batch, cb, settings)
    return parse_prelabels(FakeClient(lambda req: answer).complete(request), batch)


# --------------------------------------------------------------------------- settings
def test_settings_come_from_llm_yaml(settings):
    assert settings == TopicTaskSettings(effort="medium", max_tokens=16000, replicates=1, fallback=True,
                                         units_per_call=40, model="claude-opus-5-5")


@pytest.mark.parametrize("section, message", [
    ({"temperature": 0}, "unknown key"),
    ({"replicates": 3}, "replicates must be 1"),
    ({"effort": "extreme"}, "effort must be one of"),
    ({"units_per_call": 0}, "units_per_call must be a positive integer"),
    ({"max_tokens": True}, "max_tokens must be a positive integer"),
    ({"fallback": "yes"}, "fallback must be true or false"),
])
def test_bad_settings_fail_loudly(section, message):
    with pytest.raises(ConfigError, match=message):
        TopicTaskSettings.from_config({"model": "claude-opus-5-5", "tasks": {"topics": section}})


# --------------------------------------------------------------------------- batches
def test_batches_label_every_unit_once_with_neighbours_as_context():
    batches = plan_batches(REF, units_per_call=3, context_units=2)
    assert [b.units for b in batches] == [tuple(REF[0:3]), tuple(REF[3:6]), tuple(REF[6:7])]
    assert [b.before for b in batches] == [(), tuple(REF[1:3]), tuple(REF[4:6])]
    assert [b.after for b in batches] == [tuple(REF[3:5]), tuple(REF[6:7]), ()]
    assert list(batches[0].handles()) == ["u01", "u02", "u03"]


def test_context_can_be_switched_off_and_handles_widen_for_large_batches():
    batch = plan_batches(REF, units_per_call=100, context_units=0)[0]
    assert batch.before == batch.after == ()
    many = [dataclasses.replace(REF[0], id=f"D418:9a.1.{i}") for i in range(1, 101)]
    assert list(plan_batches(many, 100, 0)[0].handles())[-1] == "u100"
    assert list(plan_batches(many, 100, 0)[0].handles())[0] == "u001"
    assert plan_batches([], 40, 2) == []


@pytest.mark.parametrize("segments, message", [
    (WIT, "not reference languages"),
    (REF[:2] + WIT[:1], "one reference witness"),
    (REF[:2] + REF[:1], "duplicate unit ids"),
])
def test_witness_segments_are_refused_by_construction(segments, message):
    with pytest.raises(ValueError, match=message):
        plan_batches(segments)


def test_a_witness_disguised_under_the_reference_id_is_still_refused(cb, settings):
    disguised = [dataclasses.replace(s, witness=FIXTURE["reference_witness"]) for s in WIT]
    with pytest.raises(ValueError, match="not reference languages"):
        prelabel_requests(disguised, cb, settings)


@pytest.mark.parametrize("n, c", [(0, 2), (40, -1)])
def test_batch_sizes_are_checked(n, c):
    with pytest.raises(ValueError):
        plan_batches(REF, n, c)


# --------------------------------------------------------------------------- requests
def test_request_fields_follow_the_settings(cb, settings):
    request = build_request(plan_batches(REF, 40, 2)[0], cb, settings)
    assert (request.task, request.effort, request.max_tokens, request.model) == (
        "topics", "medium", 16000, "claude-opus-5-5")
    assert request.allow_fallback is True and request.replicate == "r1" and request.context is None
    assert request.prompt_sha == sha256_text(request.system)
    item = request.schema["properties"]["units"]["items"]["properties"]["topics"]["items"]
    assert item["properties"]["topic"]["enum"] == list(cb.names)
    assert request.schema["additionalProperties"] is False


def test_prelabel_requests_use_the_configured_size_unless_overridden(cb, settings):
    assert len(prelabel_requests(REF, cb, settings)) == 1
    small = prelabel_requests(REF, cb, settings, units_per_call=2, replicate="r2")
    assert len(small) == 4 and {r.replicate for r in small} == {"r2"}
    assert small[0].key() != prelabel_requests(REF, cb, settings, units_per_call=2)[0].key()


def test_body_holds_markers_handles_kinds_and_reference_text_only():
    batch = plan_batches(REF, units_per_call=3, context_units=1)[1]
    lines = render_body(batch).splitlines()
    assert lines == [
        BEFORE, f"ctx\t{REF[2].kind}\t{REF[2].text}", "",
        UNITS, *(f"u0{i + 1}\t{s.kind}\t{s.text}" for i, s in enumerate(REF[3:6])), "",
        AFTER, f"ctx\t{REF[6].kind}\t{REF[6].text}",
    ]


def test_tabs_and_line_breaks_in_a_text_cannot_break_the_line_format():
    unit = dataclasses.replace(REF[0], text="a\tb\nc\r\nd ")
    assert render_body(plan_batches([unit], 40, 0)[0]).splitlines() == [
        BEFORE, "", UNITS, f"u01\t{unit.kind}\ta b c d", "", AFTER]


def test_system_prompt_renders_the_whole_codebook(cb):
    system = render_system(cb, "bo")
    assert "Tibetan text of the Hevajratantra" in system and "$" not in system
    for topic in cb.topics:
        assert f"### {topic.name} (group: {topic.group})" in system
        assert topic.definition in system
        assert all(x in system for x in topic.include + topic.exclude)
    assert all(rule in system for rule in cb.rules)
    assert "Sanskrit (IAST transliteration) text" in render_system(cb, "sa")


def test_system_prompt_is_blind_and_never_asks_for_reasoning(cb):
    """No witness in the prompt; no request to show or explain reasoning (Opus 5.5 classifier)."""
    system = render_system(cb, "bo").lower()
    assert "chinese" not in system
    for word in ("reasoning", "explain", "explanation", "think", "step by step", "justif", "rationale"):
        assert word not in system, word


def test_editing_a_definition_changes_the_instrument(cb, settings):
    batch = plan_batches(REF, 40, 2)[0]
    edited = dataclasses.replace(cb, topics=(dataclasses.replace(cb.topics[0], definition="Edited."),)
                                 + cb.topics[1:])
    a, b = build_request(batch, cb, settings), build_request(batch, edited, settings)
    assert a.prompt_sha != b.prompt_sha and a.key() != b.key()


def test_schema_rejects_unknown_topics_and_empty_topic_lists(cb, settings):
    request = build_request(plan_batches(REF[:1], 40, 0)[0], cb, settings)
    for bad in ([{"topic": "violence", "cue": "x"}], []):
        with pytest.raises(ValueError, match="violates its schema"):
            FakeClient(lambda req: {"units": [{"ref": "u01", "topics": bad}]}).complete(request)


# --------------------------------------------------------------------------- blindness
def test_no_witness_text_can_reach_a_prelabel_request(cb, settings):
    reference = [s for s in derge.parse(DATA / "fixtures" / "mini_derge.txt", ["D417", "D418"],
                                        "bo_derge_D417_418", DATA).segments if s.kind in CONTENT_KINDS]
    witness = [*cbeta.parse(DATA / "fixtures" / "mini_cbeta.xml", "zh_T0892_song", DATA).segments, *WIT]
    requests = prelabel_requests(reference, cb, settings, units_per_call=4)
    assert len(requests) == -(-len(reference) // 4)
    reference_texts = {s.text for s in reference}
    for request in requests:
        visible = "\n".join((request.system, request.body, request.context or ""))
        assert not CJK.search(visible)
        assert not [s.id for s in witness if len(s.text.strip()) > 1 and s.text.strip() in visible]
        for line in request.body.splitlines():
            if "\t" in line:
                assert line.split("\t", 2)[2] in reference_texts
    with pytest.raises(ValueError):
        prelabel_requests(witness, cb, settings)


# --------------------------------------------------------------------------- answers
def test_correct_answers_are_kept_with_their_cues(cb, settings):
    by_text = {s.text: EXPECTED[s.id] for s in REF}
    client = FakeClient(_echo(lambda text: [by_text[text]]))
    prelabels = {}
    for batch in plan_batches(REF, 3, 2):
        prelabels.update(parse_prelabels(client.complete(build_request(batch, cb, settings)), batch))
    assert set(prelabels) == {s.id for s in REF} and len(client.requests) == 3
    for uid, (topic, cue) in EXPECTED.items():
        p = prelabels[uid]
        assert p.usable and p.topics == {topic} and not p.flags
        assert p.cues == (() if topic == "neutral" else ((topic, cue),))
        assert p.committed_topics == {topic}


@pytest.mark.parametrize("case", FIXTURE["cue_cases"], ids=lambda c: repr(c["cue"]))
def test_cue_must_be_verbatim_in_the_units_own_text(cb, settings, case):
    batch, answer = _one([("bone_corpse", case["cue"])])
    p = _parse(cb, settings, batch, answer)[REF[1].id]
    if case["verified"]:
        assert p.topics == {"bone_corpse"} and not p.flags
    else:
        assert p.topics == frozenset() and p.flags == {"cue_unverified:bone_corpse"}
    assert p.usable


def test_only_the_unverified_topic_is_dropped(cb, settings):
    batch, answer = _one([("bone_corpse", SKULL), ("harm", KILL), ("ritual", "skull cup")])
    p = _parse(cb, settings, batch, answer)[REF[1].id]
    assert p.topics == {"bone_corpse"} and p.flags == {"cue_unverified:harm", "cue_unverified:ritual"}


def test_a_topic_named_twice_is_kept_if_any_cue_verifies(cb, settings):
    batch, answer = _one([("bone_corpse", "skull"), ("bone_corpse", CHARNEL)])
    p = _parse(cb, settings, batch, answer)[REF[1].id]
    assert p.topics == {"bone_corpse"} and p.cues == (("bone_corpse", CHARNEL),) and not p.flags


def test_neutral_alone_is_kept_and_its_cue_ignored(cb, settings):
    batch, answer = _one([("neutral", "anything")], unit_index=4)
    p = _parse(cb, settings, batch, answer)[REF[4].id]
    assert p.topics == {"neutral"} and p.cues == () and not p.flags


def test_neutral_with_another_topic_is_dropped_and_flagged(cb, settings):
    batch, answer = _one([("neutral", ""), ("bone_corpse", SKULL)])
    p = _parse(cb, settings, batch, answer)[REF[1].id]
    assert p.topics == {"bone_corpse"} and p.flags == {FLAG_NEUTRAL_NOT_ALONE}
    batch, answer = _one([("neutral", ""), ("harm", "not a quote")])
    p = _parse(cb, settings, batch, answer)[REF[1].id]
    assert p.topics == frozenset() and p.flags == {FLAG_NEUTRAL_NOT_ALONE, "cue_unverified:harm"}


def test_missing_duplicate_context_and_unknown_handles(cb, settings):
    batch = plan_batches(REF, 3, 2)[1]          # units REF[3:6]
    answer = {"units": [
        {"ref": " u01 ", "topics": [{"topic": "female_agent", "cue": EXPECTED[REF[3].id][1]}]},
        {"ref": "u01", "topics": [{"topic": "neutral", "cue": ""}]},
        {"ref": "ctx", "topics": [{"topic": "harm", "cue": KILL}]},
        {"ref": "u99", "topics": [{"topic": "neutral", "cue": ""}]},
        {"ref": "u03", "topics": [{"topic": "mantra_control", "cue": EXPECTED[REF[5].id][1]}]},
    ]}
    prelabels = _parse(cb, settings, batch, answer)
    assert set(prelabels) == {s.id for s in REF[3:6]}
    first, second, third = (prelabels[s.id] for s in REF[3:6])
    assert first.topics == {"female_agent"} and first.flags == {FLAG_DUPLICATE_REF}
    assert second.reason == "unassessed" and second.topics == frozenset() and not second.usable
    assert third.topics == {"mantra_control"} and third.usable


@pytest.mark.parametrize("make, reason", [
    (lambda req: refusal_response(req, category="cyber", explanation="x"), "refused:cyber"),
    (lambda req: refusal_response(req), "refused:unspecified"),     # one reason string for every task
    (lambda req: truncated_response(req), "truncated"),
    (lambda req: invalid_response(req), "invalid"),
])
def test_failed_calls_give_no_topics(cb, settings, make, reason):
    batch = plan_batches(REF, 3, 2)[0]
    prelabels = parse_prelabels(FakeClient(make).complete(build_request(batch, cb, settings)), batch)
    assert set(prelabels) == {s.id for s in REF[:3]}
    for p in prelabels.values():
        assert p.reason == reason and p.topics == frozenset() and p.committed_topics == frozenset()


def test_substituted_model_answers_are_hints_never_committed(cb, settings):
    batch, answer = _one([("bone_corpse", SKULL), ("harm", "not a quote")])
    request = build_request(batch, cb, settings)
    response = FakeClient(lambda req: substituted_response(req, answer)).complete(request)
    p = parse_prelabels(response, batch)[REF[1].id]
    assert p.reason == "substituted_model" and not p.usable
    assert p.topics == {"bone_corpse"} and p.flags == {"cue_unverified:harm"}   # verified the same way
    assert p.committed_topics == frozenset()



def test_a_sanskrit_reference_is_accepted_and_its_cues_checked_as_latin_text(cb, settings):
    unit = Segment("sa:I.7.3", "sa_snellgrove1959", "sa", "kapāle bhakṣayed māṃsam", "I.7.3", "I.7.3",
                   "verse_line")
    batch = plan_batches([unit], 40, 0)[0]
    request = build_request(batch, cb, settings)
    assert "Sanskrit (IAST transliteration) text" in request.system
    answer = {"units": [{"ref": "u01", "topics": [{"topic": "bone_corpse", "cue": "kapāle"},
                                                  {"topic": "flesh_food", "cue": "bhakṣayed māṃsam"},
                                                  {"topic": "harm", "cue": "māraya"}]}]}
    p = parse_prelabels(FakeClient(lambda req: answer).complete(request), batch)[unit.id]
    assert p.topics == {"bone_corpse", "flesh_food"} and p.flags == {"cue_unverified:harm"}
