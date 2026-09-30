"""llm.anthropic_client against the real Anthropic SDK, without any network.

Skipped when ``anthropic`` is not installed. The SDK talks to an in-process
``httpx2.MockTransport`` that serves canned server-sent events, so these tests check
the real parameter names, the streaming helper, the beta fallback path, the
request-id header and the exception mapping end to end. (The SDK's message models
are fed to ``to_response`` in ``test_llm_to_response.py``.)
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from typing import Any, Callable

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")

from hevajra_matrix.llm.anthropic_client import AnthropicClient, build_kwargs, map_sdk_error  # noqa: E402
from hevajra_matrix.llm.check import build_check_request  # noqa: E402
from hevajra_matrix.llm.client import ConfigurationError, LLMRequest, LLMTransportError  # noqa: E402
from hevajra_matrix.llm.schema import strict  # noqa: E402

CHECK = build_check_request()


def _request(**overrides: Any) -> LLMRequest:
    base = dict(
        task="collate",
        prompt_sha="p" * 64,
        system="SYSTEM-TEXT",
        context="CONTEXT-TEXT",
        body="BODY-TEXT",
        schema=strict({"type": "object", "properties": {"ok": {"type": "boolean"}}}),
        effort="high",
        max_tokens=128000,
    )
    return LLMRequest(**{**base, **overrides})


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _events(model: str, blocks: list[dict[str, Any]], stop_reason: str, usage: dict[str, Any]) -> list[dict]:
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 1},
            },
        }
    ]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            events.append(
                {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
            )
            events.append(
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": block["text"]},
                }
            )
        else:
            events.append({"type": "content_block_start", "index": i, "content_block": block})
        events.append({"type": "content_block_stop", "index": i})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        }
    )
    events.append({"type": "message_stop"})
    return events


class MockAPI:
    """In-process API: records each HTTP request and answers with ``respond``."""

    def __init__(self, respond: Callable[[Any], Any]) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self._respond = respond

    def handler(self, request: Any) -> Any:
        self.requests.append((request.url.path, dict(request.headers), json.loads(request.content)))
        return self._respond(request)

    def client(self) -> AnthropicClient:
        sdk = anthropic.Anthropic(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )
        return AnthropicClient(sdk=sdk)


def _stream_response(events: list[dict[str, Any]]) -> Any:
    return httpx.Response(
        200, headers={"content-type": "text/event-stream", "request-id": "req_mock"}, content=_sse(events)
    )


def test_kwargs_bind_to_the_real_sdk_signatures():
    sdk = anthropic.Anthropic(api_key="test-key")
    inspect.signature(sdk.messages.stream).bind(**build_kwargs(_request()))
    inspect.signature(sdk.beta.messages.stream).bind(**build_kwargs(_request(allow_fallback=True)))
    with pytest.raises(TypeError):  # the fallback parameters exist only on the beta endpoint
        inspect.signature(sdk.messages.stream).bind(**build_kwargs(_request(allow_fallback=True)))


def test_plain_stream_end_to_end():
    api = MockAPI(
        lambda _: _stream_response(
            _events(
                "claude-opus-5-5",
                [{"type": "text", "text": '{"ok": true}'}],
                "end_turn",
                {"output_tokens": 7},
            )
        )
    )
    request = _request()
    response = api.client().complete(request)
    assert response.status == "ok" and response.usable and response.data == {"ok": True}
    assert response.request_id == "req_mock" and response.usage.output_tokens == 7
    ((path, headers, body),) = api.requests
    assert path == "/v1/messages" and "anthropic-beta" not in headers
    assert body["stream"] is True and body["model"] == "claude-opus-5-5"
    assert not {"thinking", "temperature", "top_p", "top_k", "tools", "tool_choice", "fallbacks"} & set(body)
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["messages"][0]["content"][1]
    assert body["output_config"]["effort"] == "high"


def test_fallback_stream_end_to_end_is_marked_substituted():
    iterations = [
        {
            "type": "message",
            "model": "claude-opus-5-5",
            "input_tokens": 11,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        {
            "type": "fallback_message",
            "model": "claude-opus-4-8",
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    ]
    blocks = [
        {
            "type": "fallback",
            "from": {"model": "claude-opus-5-5"},
            "to": {"model": "claude-opus-4-8"},
            "trigger": {"type": "refusal", "category": "cyber"},
        },
        {"type": "text", "text": '{"ok": true}'},
    ]
    api = MockAPI(
        lambda _: _stream_response(
            _events("claude-opus-4-8", blocks, "end_turn", {"output_tokens": 7, "iterations": iterations})
        )
    )
    response = api.client().complete(_request(allow_fallback=True))
    assert response.status == "ok" and response.data == {"ok": True}
    assert response.fallback_used and response.substituted_model and not response.usable
    assert response.served_model == "claude-opus-4-8"
    assert response.usage.input_tokens == 22  # both attempts, conservatively
    ((_, headers, body),) = api.requests
    assert headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert body["fallbacks"] == "default"


def test_refusal_stream_end_to_end():
    api = MockAPI(lambda _: _stream_response(_events("claude-opus-5-5", [], "refusal", {"output_tokens": 0})))
    response = api.client().complete(_request())
    assert response.status == "refusal" and response.data is None and response.raw_text == ""


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ConfigurationError),
        (403, ConfigurationError),
        (404, ConfigurationError),
        (400, LLMTransportError),
        (429, LLMTransportError),
        (500, LLMTransportError),
    ],
)
def test_http_errors_are_translated(status, expected):
    error = {"type": "error", "error": {"type": "some_error", "message": "synthetic"}}
    api = MockAPI(lambda _: httpx.Response(status, json=error))
    with pytest.raises(expected):
        api.client().complete(_request())


def test_connection_errors_are_translated():
    def refuse(request: Any) -> Any:
        raise httpx.ConnectError("synthetic connection failure", request=request)

    with pytest.raises(LLMTransportError, match="could not reach"):
        MockAPI(refuse).client().complete(_request())


def test_count_tokens_end_to_end():
    api = MockAPI(lambda _: httpx.Response(200, json={"input_tokens": 4242}))
    assert api.client().count_tokens(_request(allow_fallback=True)) == 4242
    ((path, headers, body),) = api.requests
    assert path == "/v1/messages/count_tokens" and "anthropic-beta" not in headers
    assert set(body) == {"model", "system", "messages", "output_config"}


def test_map_sdk_error_with_real_exception_classes():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    auth = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    timeout = anthropic.APITimeoutError(request=request)
    assert isinstance(map_sdk_error(auth, anthropic), ConfigurationError)
    assert isinstance(map_sdk_error(timeout, anthropic), LLMTransportError)
    assert map_sdk_error(RuntimeError("x"), anthropic) is None


def test_check_request_is_accepted_by_the_sdk_path():
    api = MockAPI(
        lambda _: _stream_response(
            _events(
                "claude-opus-5-5",
                [{"type": "text", "text": '{"ok": true}'}],
                "end_turn",
                {"output_tokens": 3},
            )
        )
    )
    response = api.client().complete(dataclasses.replace(CHECK))
    assert response.usable
    ((_, _, body),) = api.requests
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": CHECK.schema},
    }


def test_production_stack_replays_from_cache_and_audits_both_calls(tmp_path):
    from hevajra_matrix.llm.audit import AuditedClient
    from hevajra_matrix.llm.cache import CachedClient

    api = MockAPI(
        lambda _: _stream_response(
            _events(
                "claude-opus-5-5",
                [{"type": "text", "text": '{"ok": true}'}],
                "end_turn",
                {"output_tokens": 7},
            )
        )
    )
    pricing = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}
    stack = AuditedClient(
        CachedClient(api.client(), tmp_path / "cache"), tmp_path / "audit.jsonl", pricing, 1.0, "run-sdk"
    )
    first, second = stack.complete(_request()), stack.complete(_request())
    assert len(api.requests) == 1
    assert (first.from_cache, second.from_cache) == (False, True) and second.data == first.data
    lines = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()]
    assert [x["from_cache"] for x in lines] == [False, True]
    assert lines[0]["request_id"] == "req_mock" and lines[0]["usage"] == {
        "in": 11,
        "out": 7,
        "cache_read": 0,
        "cache_write": 0,
    }
