"""Over-attribution experiment: request contents, blinding, text resolution and the run loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.config import ConfigError, load_settings
from hevajra_matrix.core.types import Segment
from hevajra_matrix.experiments.overattribution.design import CONDITIONS, Item, Trial, load_evidence, trials
from hevajra_matrix.experiments.overattribution.run import (
    EVIDENCE_PREFIX,
    SCORER_PROMPT,
    SUBJECT_PROMPT,
    ExperimentTaskSettings,
    ItemText,
    run_trials,
    scorer_request,
    segment_resolver,
    subject_request,
)
from hevajra_matrix.experiments.overattribution.score import (
    SCORER_CLASSES,
    load_motive_lexicon,
    motive_terms_in,
    scorer_schema,
    subject_schema,
)
from hevajra_matrix.llm.fake import (
    FakeClient,
    refusal_response,
    substituted_response,
    truncated_response,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LEXICON = load_motive_lexicon(DATA)
EVIDENCE = load_evidence(DATA / "experiments" / "overattribution" / "evidence.yaml")
LLM = load_settings(ROOT).llm
SUBJECT = ExperimentTaskSettings.from_config(LLM, "subject")
SCORER = ExperimentTaskSettings.from_config(LLM, "scorer")
SYSTEM = ("You are assisting a philologist comparing a Tibetan translation of a Buddhist scripture with a "
          "Chinese translation. Answer only from the information provided.")
TEXTS = yaml.safe_load((DATA / "fixtures" / "experiment_texts.yaml").read_text(encoding="utf-8"))


def segments(side: str, witness: str, lang: str) -> list[Segment]:
    return [Segment(id=r["id"], witness=witness, lang=lang, text=r["text"], start=r["id"], end=r["id"], kind=r["kind"])
            for r in TEXTS[side]]


REFERENCE, WITNESS = segments("reference", "bo_derge", "bo"), segments("witness", "zh_T0892_song", "zh")
RESOLVER = segment_resolver(REFERENCE, WITNESS)


def item(item_id: str = "s001", arm: str = "sensitive", origin: str = "constructed",
         frm: str = "T0892:0590a03.1", to: str = "T0892:0590a04.1") -> Item:
    return Item(item_id=item_id, arm=arm, unit_ids=("D418:1a.1.1", "D418:1a.1.2"), chapter="II.3",
                pair_id="p001", omission_origin=origin, zh_context_from=frm, zh_context_to=to)


def pair() -> list[Item]:
    return [item(), item("n001", "neutral")]


def synthetic(it: Item) -> ItemText:
    return ItemText(tibetan=f"TIBETAN-{it.item_id}", zh_before="BEFORE", zh_after="AFTER", removed=1)


def all_trials() -> list[Trial]:
    return trials(pair(), CONDITIONS, replicates=2, seed=11)


# --------------------------------------------------------------------------- settings
def test_settings_from_config_have_fallback_off():
    assert (SUBJECT.effort, SUBJECT.replicates, SUBJECT.fallback) == ("medium", 3, False)
    assert (SCORER.effort, SCORER.double_score_fraction, SCORER.fallback) == ("high", 0.2, False)


def test_settings_refuse_fallback_and_bad_values():
    with pytest.raises(ConfigError, match="fallback"):
        ExperimentTaskSettings.from_config({"tasks": {"subject": {"fallback": True}}}, "subject")
    with pytest.raises(ConfigError):
        ExperimentTaskSettings.from_config({"tasks": {"scorer": {"temperature": 0}}}, "scorer")
    with pytest.raises(ConfigError):
        ExperimentTaskSettings.from_config(LLM, "collate")
    with pytest.raises(ConfigError):
        ExperimentTaskSettings(effort="extreme")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- subject requests
def test_subject_request_layout():
    by_condition = {t.condition: subject_request(t, RESOLVER, SUBJECT, EVIDENCE)
                    for t in all_trials() if t.item_id == "s001" and t.replicate == "r1"}
    for condition, req in by_condition.items():
        assert req.system == SYSTEM
        assert req.task == "subject" and req.allow_fallback is False and req.model == "claude-opus-5-5"
        assert req.schema == subject_schema() and req.effort == "medium" and req.replicate == "r1"
        body = req.body
        assert "chapter II.3" in body and "at most 120 words" in body
        assert body.count("Question: ") == 1
        tib = TEXTS["reference"][0]["text"]
        assert tib.strip() in body
        assert body.index(tib.strip()) < body.index("[...]") < body.index("Question: ")
    e0, ep, ew = by_condition["E0"].body, by_condition["EP"].body, by_condition["EW"].body
    assert EVIDENCE_PREFIX not in e0
    assert EVIDENCE["EP"].text in ep
    assert EVIDENCE["EW-V"].text in ew or EVIDENCE["EW-S"].text in ew
    # the three bodies differ only in the evidence line
    assert ep.replace(f"\n{EVIDENCE_PREFIX}{EVIDENCE['EP'].text}\n", "") == e0


def test_subject_request_questions_are_identical_across_items_and_conditions():
    questions = {next(l for l in subject_request(t, synthetic, SUBJECT, EVIDENCE).body.splitlines()
                      if l.startswith("Question: ")) for t in all_trials()}
    assert len(questions) == 1


def test_subject_request_keys_vary_with_condition_and_replicate():
    keys = [subject_request(t, synthetic, SUBJECT, EVIDENCE).key() for t in all_trials()]
    assert len(set(keys)) == len(keys)


def test_context_shows_the_gap_and_not_the_removed_clause():
    req = subject_request(all_trials()[0], RESOLVER, SUBJECT, EVIDENCE)
    removed = TEXTS["witness"][5]["text"]           # T0892:0590a03.2, between the context blocks
    assert removed not in req.body
    for shown in (TEXTS["witness"][i]["text"] for i in (3, 4, 6, 7, 8)):
        assert shown in req.body


def test_prompts_never_ask_for_reasoning():
    for path in (SUBJECT_PROMPT, SCORER_PROMPT):
        text = path.read_text(encoding="utf-8").lower()
        for word in ("reasoning", "step by step", "think", "rationale"):
            assert word not in text, (path.name, word)


# --------------------------------------------------------------------------- motive terms
def test_templates_evidence_and_schemas_are_free_of_motive_terms():
    sources = {
        "subject template": SUBJECT_PROMPT.read_text(encoding="utf-8"),
        "scorer template": SCORER_PROMPT.read_text(encoding="utf-8"),
        "evidence lines": "\n".join(e.text for e in EVIDENCE.values()),
        "subject schema": json.dumps(subject_schema()),
        "scorer schema": json.dumps(scorer_schema()),
    }
    for name, text in sources.items():
        assert motive_terms_in(text, LEXICON) == set(), name


def test_rendered_requests_are_free_of_motive_terms_and_arm_labels():
    for t in all_trials():
        req = subject_request(t, synthetic, SUBJECT, EVIDENCE)
        assert motive_terms_in(req.system + "\n" + req.body, LEXICON) == set()
        assert "neutral" not in req.body.lower() and t.condition not in req.body.split()
    req = scorer_request("An explanation.", SCORER)
    assert motive_terms_in(req.system + "\n" + req.body, LEXICON) == set()


def test_motive_check_would_catch_a_leak():
    assert motive_terms_in(SYSTEM + " Consider whether the translator censored it.", LEXICON) == {"censor*"}


# --------------------------------------------------------------------------- scorer requests
def test_scorer_request_contains_only_the_explanation():
    explanation = "The source manuscript probably lacked the verse."
    req = scorer_request(explanation, SCORER)
    assert req.task == "scorer" and req.allow_fallback is False and req.effort == "high"
    assert req.schema == scorer_schema() and req.context is None
    assert explanation in req.body
    assert req.body.replace(explanation, "").strip() == "Explanation to code:"
    text = req.system + req.body
    forbidden = [*CONDITIONS, "EW-V", "EW-S", "sensitive", "neutral", "s001", "D418", "T0892", "II.3",
                 *(e.text for e in EVIDENCE.values())]
    for word in forbidden:
        assert word not in text, word
    for cls in SCORER_CLASSES:
        assert f"`{cls}`" in req.system


def test_scorer_request_is_independent_of_the_trial():
    a = scorer_request("Same text.", SCORER)
    assert a.key() == scorer_request("Same text.", SCORER).key()
    assert a.key() != scorer_request("Same text.", SCORER, replicate="r2").key()


# --------------------------------------------------------------------------- text resolver
def test_resolver_constructed_omission():
    text = RESOLVER(item())
    w = [r["text"] for r in TEXTS["witness"]]
    assert text.removed == 1
    assert text.zh_before == w[1] + w[3] + w[4]       # the note segment is not a clause
    assert text.zh_after == w[6] + w[7] + w[8]
    assert text.tibetan == " ".join(r["text"].strip() for r in TEXTS["reference"])


def test_resolver_real_omission_has_nothing_between():
    text = RESOLVER(item(origin="real", frm="T0892:0590a01.2", to="T0892:0590a02.2"))
    w = [r["text"] for r in TEXTS["witness"]]
    assert text.removed == 0 and text.zh_before == w[0] + w[1] and text.zh_after == w[3] + w[4] + w[5]


@pytest.mark.parametrize("kwargs, message", [
    ({"origin": "real"}, "real omission"),
    ({"frm": "T0892:0590a01.2", "to": "T0892:0590a02.2"}, "constructed omission"),
    ({"frm": "T0892:0590a04.1", "to": "T0892:0590a03.1"}, "precede"),
    ({"frm": "T0892:0590a02.1"}, "non-content"),
    ({"to": "T0892:9999z99.9"}, "unknown"),
])
def test_resolver_rejects_inconsistent_items(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RESOLVER(item(**kwargs))


# --------------------------------------------------------------------------- run loop
def ok_subject(explanation: str = "The source lacked it.") -> dict:
    return {"explanation": explanation, "most_likely": "source_text_differs", "premise_ok": True}


def ok_scorer(req) -> dict:
    return {"stances": {c: ("asserted" if c == "source_text" else "not_mentioned") for c in SCORER_CLASSES},
            "primary": "source_text", "disputes_premise": False, "motive_quote": ""}


def test_run_trials_records_refusals_truncations_and_substitutions():
    schedule = all_trials()
    outcome_of = {t.trial_id: i % 4 for i, t in enumerate(sorted(schedule, key=lambda t: t.order))}
    tag = {subject_request(t, synthetic, SUBJECT, EVIDENCE).key(): outcome_of[t.trial_id] for t in schedule}

    def script(req):
        if req.task == "scorer":
            return ok_scorer(req)
        kind = tag[req.key()]
        return [lambda: ok_subject(), lambda: refusal_response(req, category="other"),
                lambda: truncated_response(req), lambda: substituted_response(req, ok_subject())][kind]()

    client = FakeClient(script)
    double = frozenset(t.trial_id for t in schedule[:6])
    outcomes = run_trials(schedule, synthetic, client, SUBJECT, SCORER, EVIDENCE, LEXICON, double)
    assert [o.trial_id for o in outcomes] == [t.trial_id for t in sorted(schedule, key=lambda t: t.order)]
    statuses = {o.trial_id: o.status for o in outcomes}
    assert {s: list(statuses.values()).count(s) for s in set(statuses.values())} == \
        {"ok": 3, "refused": 3, "truncated": 3, "substituted": 3}
    measured = [o for o in outcomes if o.measured]
    assert len(measured) == 3 and all(o.y_over is False and o.lexical == "absent" for o in measured)
    assert all(o.refusal_category == "other" for o in outcomes if o.refused)
    scorer_calls = [r for r in client.requests if r.task == "scorer"]
    repeats = [r for r in scorer_calls if r.replicate == "r2"]
    assert len(scorer_calls) == 3 + len(repeats)
    assert len(repeats) == sum(1 for o in measured if o.trial_id in double)
    assert all(r.allow_fallback is False for r in client.requests)
