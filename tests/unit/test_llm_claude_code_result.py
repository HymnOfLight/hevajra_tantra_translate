"""llm.claude_code_result: one ``claude -p`` JSON result -> one LLMResponse (pure).

The result shapes follow Claude Code 2.1.296: ``is_error`` is set on some answers too, a
capped answer carries ``api_error`` ``max_output_tokens`` and a synthetic ``stop_reason``
``stop_sequence``, and ``stop_reason`` can be stale on a run that failed.
"""

from __future__ import annotations

import pytest

from hevajra_matrix.llm.claude_code_result import (
    ModelUnavailable,
    UsageLimitReached,
    same_model,
    served_model,
    to_response,
)
from hevajra_matrix.llm.client import LLMRequest, LLMTransportError
from hevajra_matrix.llm.schema import strict

MODEL = "claude-opus-5-5"
SCHEMA = strict({"type": "object", "properties": {"ok": {"type": "boolean"}}})


def request(max_tokens: int = 2048) -> LLMRequest:
    return LLMRequest(task="check", prompt_sha="p", system="SYSTEM", body="BODY", schema=SCHEMA, effort="high",
                      max_tokens=max_tokens, context="CONTEXT")


def result(**overrides) -> dict:
    base = {"type": "result", "subtype": "success", "is_error": False, "stop_reason": "end_turn",
            "session_id": "sess-1", "structured_output": {"ok": True}, "result": "",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
                      "cache_creation_input_tokens": 7},
            "modelUsage": {MODEL: {"inputTokens": 10, "outputTokens": 5}}}
    base.update(overrides)
    return base


def respond(**overrides):
    return to_response(result(**overrides), request(), "k", "now")


# --------------------------------------------------------------------------- answers
def test_success_with_a_valid_structured_answer_is_ok() -> None:
    r = to_response(result(num_turns=2, terminal_reason="completed"), request(), "k", "now",
                    provenance={"cli_version": "2.1.296", "auth": "oauth_token (firstParty)", "seconds": 3.5})
    assert r.status == "ok" and r.data == {"ok": True} and r.usable
    assert r.served_model == MODEL and not r.fallback_used and r.request_id == "sess-1"
    assert r.usage.cache_read_input_tokens == 100 and r.usage.output_tokens == 5
    assert r.client_info == {"cli_version": "2.1.296", "auth": "oauth_token (firstParty)", "seconds": 3.5,
                             "subtype": "success", "num_turns": 2, "terminal_reason": "completed",
                             "cli_stop_reason": "end_turn"}


def test_live_check_shape_with_stop_reason_tool_use_is_ok() -> None:
    assert respond(stop_reason="tool_use").status == "ok"


@pytest.mark.parametrize("overrides", [
    {"structured_output": {"ok": "yes"}},
    {"structured_output": None, "result": "plain text"},
    {"subtype": "error_max_structured_output_retries", "is_error": True, "structured_output": None,
     "terminal_reason": "structured_output_retry_exhausted"},
    {"subtype": "error_max_turns", "is_error": True, "structured_output": None, "num_turns": 3},
])
def test_answers_without_a_valid_structured_output_are_invalid(overrides) -> None:
    r = respond(**overrides)
    assert r.status == "invalid" and r.data is None and not r.usable


@pytest.mark.parametrize("overrides", [{"stop_reason": "refusal"}, {"safety_stops": 1},
                                       {"stop_reason": "refusal", "is_error": True, "structured_output": None}])
def test_refusals_are_values(overrides) -> None:
    r = respond(**overrides)
    assert r.status == "refusal" and r.data is None


@pytest.mark.parametrize("overrides", [
    {"stop_reason": "max_tokens"},
    {"is_error": True, "api_error": "max_output_tokens", "stop_reason": "stop_sequence", "structured_output": None},
    {"is_error": True, "structured_output": None, "stop_reason": "stop_sequence",
     "result": "API Error: Claude's response exceeded the 128000 output token maximum."},
])
def test_truncations_are_values_reported_as_max_tokens(overrides) -> None:
    r = respond(**overrides)
    assert r.status == "truncated" and r.stop_reason == "max_tokens"
    assert r.client_info["cli_stop_reason"] == overrides["stop_reason"]


def test_the_truncation_text_alone_does_not_truncate_a_normal_answer() -> None:
    r = respond(result='{"note": "exceeds the output token maximum"}')
    assert r.status == "ok"


def test_output_beyond_max_tokens_is_flagged() -> None:
    r = to_response(result(usage={"output_tokens": 5000}), request(max_tokens=2048), "k", "now")
    assert r.status == "ok" and r.client_info["output_over_max_tokens"] is True


# --------------------------------------------------------------------------- models
def test_output_from_another_model_is_a_substitution() -> None:
    r = respond(modelUsage={MODEL: {"outputTokens": 3}, "claude-opus-4-8": {"outputTokens": 40}})
    assert r.served_model == "claude-opus-4-8" and r.fallback_used and not r.usable


def test_served_model_reading() -> None:
    assert served_model({f"{MODEL}[1m]": {"outputTokens": 9}}, MODEL) == (MODEL, False)
    assert served_model({f"{MODEL}-20260701": {"outputTokens": 9}}, MODEL) == (MODEL, False)
    assert served_model({}, MODEL) == ("unknown", True)
    assert served_model({MODEL: {"outputTokens": 9}, "claude-haiku-4-5": {"outputTokens": 0}}, MODEL) == (MODEL, False)


def test_a_longer_model_id_is_not_the_requested_model() -> None:
    assert not same_model("claude-opus-5-5", "claude-opus-5")
    assert served_model({"claude-opus-5-5": {"outputTokens": 9}}, "claude-opus-5") == ("claude-opus-5-5", True)


# --------------------------------------------------------------------------- failures
LIMIT_TEXTS = ["You've hit your session limit \u00b7 resets 7pm", "You've hit your weekly limit \u00b7 resets Mon",
               "You\u2019ve hit your Opus limit", "You're out of usage credits",
               "You've hit your org's monthly spend limit", "You've reached your Fable limit. Switch to another model",
               "Opus requires usage credits. Switch to another model", "You're out of extra usage"]


@pytest.mark.parametrize("text", LIMIT_TEXTS)
def test_plan_limit_messages_stop_the_stage(text) -> None:
    with pytest.raises(UsageLimitReached):
        respond(is_error=True, structured_output=None, result=text)


@pytest.mark.parametrize("api_error", ["usage_limit_reached", "model_requires_usage_credits"])
def test_typed_usage_limit_fields_stop_the_stage(api_error) -> None:
    with pytest.raises(UsageLimitReached):
        respond(is_error=True, structured_output=None, result="x", api_error=api_error, api_error_status=429)


@pytest.mark.parametrize("api_error", ["model_substitution_disabled", "no_allowed_fallback"])
def test_an_unavailable_model_is_its_own_failure(api_error) -> None:
    with pytest.raises(ModelUnavailable):
        respond(is_error=True, structured_output=None, result="x", api_error=api_error)


@pytest.mark.parametrize("overrides", [
    {"is_error": True, "result": "API Error: 500"},
    {"subtype": "error_during_execution", "errors": ["boom"]},
    {"subtype": "error_during_execution", "is_error": True, "stop_reason": "max_tokens", "errors": ["boom"]},
    {"subtype": "error_during_execution", "is_error": True, "stop_reason": "refusal"},
    {"subtype": "error_max_budget_usd", "is_error": True},
    {"is_error": True, "result": "API Error: Server is temporarily limiting requests (not your usage limit)",
     "api_error_status": 429},
    {"is_error": True, "result": "API Error: 529 overloaded", "api_error_status": 529},
])
def test_run_failures_are_raised_not_cached(overrides) -> None:
    with pytest.raises(LLMTransportError) as caught:
        respond(**{"structured_output": None, **overrides})
    assert not isinstance(caught.value, (UsageLimitReached, ModelUnavailable)), "transient failures stay retryable"
