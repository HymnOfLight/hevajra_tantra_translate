"""Deterministic stand-ins for the Claude client, for the tests of every workstream.

    FakeClient(script)                      script(request) -> dict or LLMResponse
    ok_response(request, data)              the outcomes the real client can return,
    refusal_response(request, ...)          built exactly as ``AnthropicClient`` would
    truncated_response(request, ...)        build them (same key, models, stop reasons),
    invalid_response(request, ...)          so code under test can exercise every path
    substituted_response(request, data)     without the SDK or a network

A dict returned by a script becomes an ``ok`` response and must therefore be valid
against ``request.schema``, as the real API guarantees for strict schemas; otherwise
``ValueError`` names the violations. A test that wants a malformed answer returns
``invalid_response(...)`` explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .client import LLMRequest, LLMResponse, Usage, canonical_json
from .schema import validate

FAKE_TIMESTAMP = "2026-01-01T00:00:00+00:00"
FALLBACK_MODEL = "claude-opus-4-8"  # a model the server-side fallback may substitute

Script = Callable[[LLMRequest], "Mapping[str, Any] | LLMResponse"]


class FakeClient:
    """``LLMClient`` that answers from ``script`` and records every request it saw."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        result = self.script(request)
        if isinstance(result, LLMResponse):
            if result.key != request.key():
                raise ValueError(f"script returned a response for another request (task {request.task!r})")
            return result
        if isinstance(result, Mapping):
            return ok_response(request, result)
        raise TypeError(f"script must return a dict or an LLMResponse, not {type(result).__name__}")


def ok_response(request: LLMRequest, data: Mapping[str, Any], usage: Usage | None = None) -> LLMResponse:
    """A schema-valid answer from the requested model."""
    return LLMResponse(
        status="ok",
        data=_valid(request, data),
        raw_text=canonical_json(data),
        stop_reason="end_turn",
        **_common(request, usage),
    )


def substituted_response(
    request: LLMRequest,
    data: Mapping[str, Any],
    served_model: str = FALLBACK_MODEL,
    usage: Usage | None = None,
) -> LLMResponse:
    """A schema-valid answer produced by a fallback model: a reviewer hint, never a measurement."""
    common = {**_common(request, usage), "served_model": served_model}
    return LLMResponse(
        status="ok",
        data=_valid(request, data),
        raw_text=canonical_json(data),
        stop_reason="end_turn",
        fallback_used=True,
        **common,
    )


def refusal_response(
    request: LLMRequest,
    category: str | None = None,
    explanation: str | None = None,
    usage: Usage | None = None,
) -> LLMResponse:
    """A declined request; ``category`` None mirrors a refusal without ``stop_details``."""
    return LLMResponse(
        status="refusal",
        data=None,
        raw_text="",
        stop_reason="refusal",
        refusal_category=category,
        refusal_explanation=explanation,
        **_common(request, usage),
    )


def truncated_response(
    request: LLMRequest, raw_text: str = '{"units": [', usage: Usage | None = None
) -> LLMResponse:
    """Output cut off at ``max_tokens``."""
    return LLMResponse(
        status="truncated", data=None, raw_text=raw_text, stop_reason="max_tokens", **_common(request, usage)
    )


def invalid_response(
    request: LLMRequest, raw_text: str = "not json", usage: Usage | None = None
) -> LLMResponse:
    """A finished answer that is not schema-valid JSON."""
    return LLMResponse(
        status="invalid", data=None, raw_text=raw_text, stop_reason="end_turn", **_common(request, usage)
    )


def _common(request: LLMRequest, usage: Usage | None) -> dict[str, Any]:
    return {
        "key": request.key(),
        "requested_model": request.model,
        "served_model": request.model,
        "usage": usage or Usage(),
        "created_at": FAKE_TIMESTAMP,
    }


def _valid(request: LLMRequest, data: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate(data, request.schema)
    if errors:
        shown = "; ".join(errors[:5])
        raise ValueError(
            f"fake answer for task {request.task!r} violates its schema: {shown}. "
            "Use invalid_response() to simulate an invalid answer."
        )
    return dict(data)
