"""llm.fake: the scripted client and the outcome builders other workstreams test with."""

from __future__ import annotations

import pytest

from hevajra_matrix.llm.client import LLMRequest, Usage
from hevajra_matrix.llm.fake import (
    FALLBACK_MODEL,
    FakeClient,
    invalid_response,
    ok_response,
    refusal_response,
    substituted_response,
    truncated_response,
)
from hevajra_matrix.llm.schema import strict

SCHEMA = strict({"type": "object", "properties": {"units": {"type": "array", "items": {"type": "string"}}}})


def _request(body: str = "BODY", **overrides) -> LLMRequest:
    return LLMRequest(
        task="topics",
        prompt_sha="p",
        system="S",
        body=body,
        schema=SCHEMA,
        effort="medium",
        max_tokens=100,
        **overrides,
    )


def test_dict_answers_become_ok_responses_and_requests_are_recorded():
    client = FakeClient(lambda req: {"units": [req.body]})
    first, second = _request("a"), _request("b")
    response = client.complete(first)
    client.complete(second)
    assert client.requests == [first, second]
    assert response.status == "ok" and response.usable and response.data == {"units": ["a"]}
    assert response.key == first.key() and response.served_model == response.requested_model == first.model
    assert response.stop_reason == "end_turn" and response.from_cache is False


def test_schema_invalid_dict_is_a_test_bug_not_a_silent_ok():
    client = FakeClient(lambda req: {"units": "not a list"})
    with pytest.raises(ValueError, match="invalid_response"):
        client.complete(_request())


def test_script_may_return_any_outcome():
    outcomes = {
        "refusal": lambda r: refusal_response(r, "cyber", "why"),
        "truncated": truncated_response,
        "invalid": invalid_response,
        "ok": lambda r: substituted_response(r, {"units": []}),
    }
    for status, build in outcomes.items():
        response = FakeClient(build).complete(_request())
        assert response.status == status and not response.usable


def test_outcome_builders():
    req = _request()
    refusal = refusal_response(req)
    assert (refusal.data, refusal.raw_text, refusal.stop_reason) == (None, "", "refusal")
    assert refusal.refusal_category is None and refusal.refusal_explanation is None
    assert refusal_response(req, "bio").refusal_category == "bio"
    truncated = truncated_response(req)
    assert truncated.stop_reason == "max_tokens" and truncated.data is None and truncated.raw_text
    assert invalid_response(req, "{").raw_text == "{"
    substituted = substituted_response(req, {"units": ["x"]})
    assert substituted.status == "ok" and substituted.served_model == FALLBACK_MODEL
    assert substituted.fallback_used and substituted.substituted_model and not substituted.usable
    ok = ok_response(req, {"units": []}, usage=Usage(1, 2, 3, 4))
    assert ok.usage == Usage(1, 2, 3, 4) and ok.raw_text == '{"units":[]}'
    assert {r.key for r in (refusal, truncated, substituted, ok)} == {req.key()}
    with pytest.raises(ValueError):
        substituted_response(req, {"units": [1]})


def test_response_for_another_request_is_rejected():
    other = _request("other")
    with pytest.raises(ValueError, match="another request"):
        FakeClient(lambda req: ok_response(other, {"units": []})).complete(_request())


def test_script_must_return_dict_or_response():
    with pytest.raises(TypeError):
        FakeClient(lambda req: "text").complete(_request())
