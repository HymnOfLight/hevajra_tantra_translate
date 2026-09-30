"""llm.check: the claude-check request and its summary."""

from __future__ import annotations

import dataclasses

from hevajra_matrix.llm.anthropic_client import build_kwargs
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.check import build_check_request, run_check
from hevajra_matrix.llm.client import Usage
from hevajra_matrix.llm.fake import (
    FakeClient,
    invalid_response,
    ok_response,
    refusal_response,
    substituted_response,
)


def test_check_request_is_tiny_strict_and_never_falls_back():
    request = build_check_request()
    assert request.task == "check" and request.effort == "low" and request.max_tokens == 1024
    assert request.allow_fallback is False and request.model == "claude-opus-5-5" and request.context is None
    assert request.schema == {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
        "required": ["ok"],
    }
    assert "betas" not in build_kwargs(request)
    assert build_check_request().key() == request.key()  # deterministic
    assert "reason" not in (request.system + request.body).lower()  # no reasoning-extraction wording


def test_passing_check_summary_has_no_text():
    usage = Usage(20, 5, 0, 0)
    fake = FakeClient(
        lambda r: dataclasses.replace(ok_response(r, {"ok": True}, usage=usage), request_id="req_9")
    )
    summary = run_check(fake)
    assert summary == {
        "passed": True,
        "status": "ok",
        "requested_model": "claude-opus-5-5",
        "served_model": "claude-opus-5-5",
        "fallback_used": False,
        "stop_reason": "end_turn",
        "refusal_category": None,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "request_id": "req_9",
        "from_cache": False,
    }


def test_check_fails_on_anything_but_a_true_answer_from_the_requested_model():
    for script in (
        lambda r: {"ok": False},
        lambda r: refusal_response(r, "cyber"),
        invalid_response,
        lambda r: substituted_response(r, {"ok": True}),
    ):
        assert run_check(FakeClient(script))["passed"] is False


def test_cached_replay_is_visible(tmp_path):
    cached = CachedClient(FakeClient(lambda r: {"ok": True}), tmp_path)
    run_check(cached)
    assert run_check(cached)["from_cache"] is True


def test_custom_request_is_used():
    fake = FakeClient(lambda r: {"ok": True})
    run_check(fake, build_check_request(effort="medium", replicate="2026-09-30"))
    assert fake.requests[0].effort == "medium" and fake.requests[0].replicate == "2026-09-30"
