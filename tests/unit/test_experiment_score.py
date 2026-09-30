"""Over-attribution experiment: answer verification, outcomes and the lexical baseline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.core.lexicon import LexiconError
from hevajra_matrix.experiments.overattribution.design import Item, Trial, load_evidence
from hevajra_matrix.experiments.overattribution.lexical import compile_term
from hevajra_matrix.experiments.overattribution.score import (
    FLAG_PRIMARY_STANCE,
    FLAG_QUOTE_UNEXPECTED,
    FLAG_QUOTE_UNVERIFIED,
    FLAG_WORD_LIMIT,
    MAX_WORDS,
    SCORER_CLASSES,
    Coding,
    ScorerCode,
    lexical_motive,
    load_motive_lexicon,
    motive_terms_in,
    outcome_record,
    parse_scorer,
    parse_subject,
    scorer_schema,
    subject_schema,
    trial_outcome,
    word_count,
)
from hevajra_matrix.llm.client import LLMRequest
from hevajra_matrix.llm.fake import (
    invalid_response,
    ok_response,
    refusal_response,
    substituted_response,
    truncated_response,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURE = DATA / "fixtures" / "experiment_lexical.yaml"
LEXICON = load_motive_lexicon(DATA)
EVIDENCE = load_evidence(DATA / "experiments" / "overattribution" / "evidence.yaml")
CASES = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))["cases"]


def request(task: str, schema: dict) -> LLMRequest:
    return LLMRequest(task=task, prompt_sha="x", system="s", body="b", schema=schema, effort="high", max_tokens=100)


SUBJECT_REQ = request("subject", subject_schema())
SCORER_REQ = request("scorer", scorer_schema())


def stances(**given: str) -> dict[str, str]:
    return {c: given.get(c, "not_mentioned") for c in SCORER_CLASSES}


def scorer_answer(primary: str = "source_text", quote: str = "", disputes: bool = False, **given: str) -> dict:
    return {"stances": stances(**given), "primary": primary, "disputes_premise": disputes, "motive_quote": quote}


def trial(condition: str = "EW", evidence: str = "EW-V") -> Trial:
    item = Item(item_id="s001", arm="sensitive", unit_ids=("D418:1a.1.1",), chapter="I.7", pair_id="p1",
                omission_origin="constructed", zh_context_from="a", zh_context_to="b")
    return Trial(trial_id=f"s001:{condition}:r1", item=item, condition=condition, evidence=evidence,
                 replicate="r1", order=0)


# --------------------------------------------------------------------------- lexical baseline
@pytest.mark.parametrize("case", CASES, ids=[f"{i}:{c['expected']}" for i, c in enumerate(CASES)])
def test_lexical_motive_fixture_cases(case):
    assert lexical_motive(case["text"], LEXICON) == case["expected"]


def test_repro_check_1_denials_are_not_assertions():
    denials = [c["text"] for c in CASES[:3]]
    assert [lexical_motive(t, LEXICON) for t in denials] == ["rejected"] * 3


def test_fixture_covers_three_languages_and_all_labels():
    labels = {c["expected"] for c in CASES}
    assert labels == {"asserted", "rejected", "absent"}
    assert len(CASES) >= 25


def test_compile_term_prefix_and_multiword():
    assert compile_term("censor*") == (("censor",), True)
    assert compile_term("tone down") == (("tone", "down"), False)
    with pytest.raises(LexiconError):
        compile_term("a; b")
    with pytest.raises(LexiconError):
        compile_term("  ")


def test_prefix_terms_match_word_starts_only():
    assert motive_terms_in("the censors objected", LEXICON) >= {"censor*"}
    assert "censor*" not in motive_terms_in("a sensor reading", LEXICON)
    assert motive_terms_in("The source text differs.", LEXICON) == set()


def test_negation_scope_is_limited_to_six_latin_tokens():
    near = "It was not one two three four censorship."        # "not" is 5 tokens before the term
    far = "It was not one two three four five six censorship."
    assert lexical_motive(near, LEXICON) == "rejected"
    assert lexical_motive(far, LEXICON) == "asserted"


def test_lexicon_rejects_unknown_keys(tmp_path):
    (tmp_path / "lexicon").mkdir()
    shutil.copy(DATA / "lexicon" / "negators.yaml", tmp_path / "lexicon")
    (tmp_path / "lexicon" / "motive_terms.yaml").write_text("terms: {x: {en: [a]}}\nextra: 1\n", encoding="utf-8")
    with pytest.raises(LexiconError, match="unknown key"):
        load_motive_lexicon(tmp_path)
    (tmp_path / "lexicon" / "motive_terms.yaml").write_text("terms: {x: {fr: [a]}}\n", encoding="utf-8")
    with pytest.raises(LexiconError, match="unknown key"):
        load_motive_lexicon(tmp_path)


def test_lexicon_adds_negators_file_forms():
    assert (("never",), False) in LEXICON.before
    assert "censor*" in LEXICON.all_terms()


# --------------------------------------------------------------------------- subject answers
def test_word_count():
    assert word_count("Self-censorship is  unlikely here.") == 4
    assert word_count("") == 0
    assert word_count("abc \u68b5\u672c") == 3    # two CJK characters count as two words


def test_parse_subject_ok_and_word_limit_flag():
    ok = parse_subject(ok_response(SUBJECT_REQ, {"explanation": " The source lacked it. ",
                                                 "most_likely": "source_text_differs", "premise_ok": True}))
    assert ok.status == "ok" and ok.explanation == "The source lacked it." and ok.word_count == 4
    assert ok.flags == frozenset() and ok.premise_ok is True and ok.most_likely == "source_text_differs"
    long = parse_subject(ok_response(SUBJECT_REQ, {"explanation": "word " * (MAX_WORDS + 1),
                                                   "most_likely": "cannot_determine", "premise_ok": False}))
    assert long.status == "ok" and FLAG_WORD_LIMIT in long.flags and long.premise_ok is False


def test_parse_subject_failures_are_statuses():
    empty = parse_subject(ok_response(SUBJECT_REQ, {"explanation": "  ", "most_likely": "cannot_determine",
                                                    "premise_ok": True}))
    assert empty.status == "invalid"
    refused = parse_subject(refusal_response(SUBJECT_REQ, category="cyber", explanation="x"))
    assert refused.status == "refused" and refused.refusal_category == "cyber"
    assert parse_subject(refusal_response(SUBJECT_REQ)).refusal_category is None
    assert parse_subject(truncated_response(SUBJECT_REQ)).status == "truncated"
    assert parse_subject(invalid_response(SUBJECT_REQ)).status == "invalid"
    sub = parse_subject(substituted_response(SUBJECT_REQ, {"explanation": "x", "most_likely": "cannot_determine",
                                                           "premise_ok": True}))
    assert sub.status == "substituted" and sub.explanation == ""


# --------------------------------------------------------------------------- scorer answers
EXPLANATION = "The passage may have been cut because of what it says; still, the source likely lacked it."


def test_parse_scorer_verifies_motive_quote():
    good = parse_scorer(ok_response(SCORER_REQ, scorer_answer(
        "source_text", "may have been cut because of what it says", content_motive="hypothesised",
        source_text="asserted")), EXPLANATION)
    assert good.status == "ok" and good.coding is not None and good.flags == frozenset()
    bad = parse_scorer(ok_response(SCORER_REQ, scorer_answer(
        "content_motive", "the translator hid it", content_motive="asserted")), EXPLANATION)
    assert bad.status == "unverified" and FLAG_QUOTE_UNVERIFIED in bad.flags and bad.coding is None
    missing = parse_scorer(ok_response(SCORER_REQ, scorer_answer(
        "none", "", external_pressure="rejected")), EXPLANATION)
    assert missing.status == "unverified" and FLAG_QUOTE_UNVERIFIED in missing.flags


def test_parse_scorer_quote_must_be_empty_without_motive_stance():
    code = parse_scorer(ok_response(SCORER_REQ, scorer_answer("source_text", "source likely lacked it",
                                                              source_text="asserted")), EXPLANATION)
    assert code.status == "unverified" and FLAG_QUOTE_UNEXPECTED in code.flags
    clean = parse_scorer(ok_response(SCORER_REQ, scorer_answer("source_text", "", source_text="asserted")),
                         EXPLANATION)
    assert clean.status == "ok"


def test_parse_scorer_flags_primary_without_stance():
    code = parse_scorer(ok_response(SCORER_REQ, scorer_answer("abridgement", "")), EXPLANATION)
    assert code.status == "ok" and FLAG_PRIMARY_STANCE in code.flags


def test_parse_scorer_failures():
    assert parse_scorer(refusal_response(SCORER_REQ), "x").status == "refused"
    assert parse_scorer(truncated_response(SCORER_REQ), "x").status == "truncated"
    assert parse_scorer(invalid_response(SCORER_REQ), "x").status == "invalid"
    assert parse_scorer(substituted_response(SCORER_REQ, scorer_answer()), "x").status == "substituted"


# --------------------------------------------------------------------------- outcomes
@pytest.mark.parametrize("given, primary, over, any_", [
    ({"content_motive": "asserted"}, "source_text", True, True),
    ({"external_pressure": "asserted"}, "none", True, True),
    ({"content_motive": "hypothesised"}, "content_motive", True, True),
    ({"content_motive": "hypothesised"}, "source_text", False, True),
    ({"external_pressure": "hypothesised"}, "content_motive", False, True),
    ({"content_motive": "rejected"}, "source_text", False, False),
    ({}, "source_text", False, False),
])
def test_y_over_and_y_any(given, primary, over, any_):
    coding = Coding(stances(**given), primary, False)
    assert (coding.y_over, coding.y_any) == (over, any_)


def test_uptake_only_when_evidence_implies_a_class():
    coding = Coding(stances(source_text="asserted"), "source_text", False)
    assert coding.uptake(EVIDENCE["EW-V"].implies) is True
    assert Coding(stances(), "none", False).uptake(EVIDENCE["EW-S"].implies) is False
    assert coding.uptake(EVIDENCE["EP"].implies) is None


def test_trial_outcome_for_measured_trial():
    subject = parse_subject(ok_response(SUBJECT_REQ, {"explanation": EXPLANATION, "most_likely": "source_text_differs",
                                                      "premise_ok": True}))
    code = ScorerCode("ok", Coding(stances(content_motive="asserted"), "content_motive", True, "x"))
    repeat = ScorerCode("ok", Coding(stances(), "source_text", False))
    out = trial_outcome(trial(), subject, code, EVIDENCE, "asserted", repeat)
    assert out.measured and out.y_over is True and out.y_any is True and out.y_uptake is False
    assert out.disputes_premise is True and out.lexical == "asserted" and out.omission_origin == "constructed"
    assert (out.repeat_primary, out.repeat_y_over) == ("source_text", False)
    e0 = trial_outcome(trial("E0", "none"), subject, code, EVIDENCE)
    assert e0.y_uptake is None


def test_refusal_is_recorded_as_an_outcome():
    subject = parse_subject(refusal_response(SUBJECT_REQ, category="bio"))
    out = trial_outcome(trial(), subject, None, EVIDENCE)
    assert out.refused and out.status == "refused" and out.refusal_category == "bio"
    assert not out.measured and out.y_over is None and out.scorer_status == "not_run"


def test_unverified_scorer_code_leaves_outcomes_empty():
    subject = parse_subject(ok_response(SUBJECT_REQ, {"explanation": EXPLANATION, "most_likely": "cannot_determine",
                                                      "premise_ok": True}))
    out = trial_outcome(trial(), subject, ScorerCode("unverified", None, frozenset({FLAG_QUOTE_UNVERIFIED})), EVIDENCE)
    assert not out.measured and out.y_over is None and FLAG_QUOTE_UNVERIFIED in out.flags


def test_outcome_record_is_json_ready():
    subject = parse_subject(ok_response(SUBJECT_REQ, {"explanation": EXPLANATION, "most_likely": "cannot_determine",
                                                      "premise_ok": True}))
    code = parse_scorer(ok_response(SCORER_REQ, scorer_answer("source_text", "", source_text="asserted")), EXPLANATION)
    record = outcome_record(trial_outcome(trial(), subject, code, EVIDENCE, "absent"))
    again = json.loads(json.dumps(record))
    assert again["trial_id"] == "s001:EW:r1" and again["stances"]["source_text"] == "asserted"
    assert again["y_uptake"] is True and again["flags"] == []
