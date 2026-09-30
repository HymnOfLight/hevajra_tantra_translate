"""llm.anthropic_client.to_response: one SDK message -> one LLMResponse.

The canned messages in ``data/fixtures/llm/messages.json`` (synthetic) are fed in three
shapes: attribute objects (how the SDK models are read), plain mappings, and, when
``anthropic`` is installed, the SDK's own ``BetaMessage`` model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hevajra_matrix.llm.anthropic_client import parse_answer, to_response
from hevajra_matrix.llm.check import build_check_request
from hevajra_matrix.llm.client import Usage

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "llm" / "messages.json"
CASES = {
    name: case for name, case in json.loads(FIXTURE.read_text("utf-8")).items() if not name.startswith("_")
}
CHECK = build_check_request()  # schema {"ok": boolean}, model claude-opus-5-5
NOW = "2026-09-30T12:00:00+00:00"
KEY = "k" * 64
HAVE_SDK = importlib.util.find_spec("anthropic") is not None


def as_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: as_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [as_namespace(v) for v in obj]
    return obj


def as_sdk_model(obj: dict[str, Any]) -> Any:
    import anthropic

    return anthropic.types.beta.BetaMessage.model_validate(obj)


SHAPES = {"attributes": as_namespace, "mapping": lambda obj: obj, "sdk_model": as_sdk_model}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_fixture_messages(name, shape):
    if shape == "sdk_model" and not HAVE_SDK:
        pytest.skip("anthropic SDK not installed")
    case, expect = CASES[name], CASES[name]["expect"]
    response = to_response(SHAPES[shape](case["message"]), CHECK, KEY, NOW)
    assert response.key == KEY and response.created_at == NOW and response.from_cache is False
    assert response.status == expect["status"]
    assert response.data == expect["data"]
    assert response.raw_text == expect["raw_text"]
    assert response.fallback_used is expect["fallback_used"]
    assert response.requested_model == "claude-opus-5-5"
    assert response.served_model == expect.get("served_model", "claude-opus-5-5")
    for field in ("refusal_category", "refusal_explanation"):
        if field in expect:
            assert getattr(response, field) == expect[field]
    usage = response.usage
    counts = [
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
    ]
    assert counts == expect["usage"]
    assert response.usable is (expect["status"] == "ok" and not expect["fallback_used"])


def test_the_fixture_covers_every_status_and_fallback_signal():
    statuses = {case["expect"]["status"] for case in CASES.values()}
    assert statuses == {"ok", "refusal", "truncated", "invalid"}
    assert {
        "fallback_signalled_only_by_iterations",
        "sticky_fallback_without_block",
        "fallback_mid_stream_continued",
        "refusal_without_details",
    } <= set(CASES)


def _message(
    text: str = '{"ok": true}', stop_reason: str | None = "end_turn", **extra: Any
) -> SimpleNamespace:
    fields = dict(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        stop_details=None,
        content=[SimpleNamespace(type="text", text=text)],
        usage=None,
    )
    return SimpleNamespace(**{**fields, **extra})


def test_request_id_from_the_message_or_the_stream():
    message = _message()
    assert to_response(message, CHECK, KEY, NOW).request_id is None
    message._request_id = "req_from_message"
    assert to_response(message, CHECK, KEY, NOW).request_id == "req_from_message"
    assert to_response(message, CHECK, KEY, NOW, request_id="req_from_stream").request_id == "req_from_stream"


def test_context_window_stop_is_a_truncation():
    response = to_response(_message('{"ok": tr', "model_context_window_exceeded"), CHECK, KEY, NOW)
    assert response.status == "truncated" and response.data is None and response.raw_text == '{"ok": tr'


@pytest.mark.parametrize("stop_reason", ["tool_use", "pause_turn", "stop_sequence", None])
def test_unexpected_stop_reason_is_invalid_even_with_valid_json(stop_reason):
    response = to_response(_message(stop_reason=stop_reason), CHECK, KEY, NOW)
    assert response.status == "invalid" and response.data is None
    assert response.stop_reason == (stop_reason or "")


def test_missing_usage_and_null_counts_become_zero():
    assert to_response(_message(), CHECK, KEY, NOW).usage == Usage()
    partial = SimpleNamespace(input_tokens=5, output_tokens=None)  # other fields absent
    assert to_response(_message(usage=partial), CHECK, KEY, NOW).usage == Usage(5, 0, 0, 0)
    assert to_response(
        _message(usage=SimpleNamespace(input_tokens=5, iterations=None)), CHECK, KEY, NOW
    ).usage == Usage(5, 0, 0, 0)


def test_missing_served_model_counts_as_substituted():
    response = to_response(_message(model=None), CHECK, KEY, NOW)
    assert response.fallback_used and not response.usable


def test_text_blocks_are_joined_and_other_blocks_ignored():
    message = _message(
        content=[
            SimpleNamespace(type="text", text='{"ok": '),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            SimpleNamespace(type="text", text="false}"),
        ]
    )
    response = to_response(message, CHECK, KEY, NOW)
    assert response.status == "ok" and response.data == {"ok": False}


def test_parse_answer():
    assert parse_answer('{"ok": true}', CHECK.schema) == ({"ok": True}, [])
    assert parse_answer("[1]", CHECK.schema) == (None, ["top-level JSON value is not an object"])
    data, errors = parse_answer("", CHECK.schema)
    assert data is None and errors[0].startswith("not valid JSON")
    assert parse_answer('{"ok": 1}', CHECK.schema) == (None, ["$.ok: expected boolean, got integer"])
