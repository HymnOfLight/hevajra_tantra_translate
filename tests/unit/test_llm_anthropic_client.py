"""llm.anthropic_client without the SDK: request building, the streaming call against a
fake SDK object, error mapping and the lazy SDK import.

Response normalisation is in ``test_llm_to_response.py``; the same paths against the
real SDK (no network) are in ``test_llm_anthropic_sdk.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hevajra_matrix.llm import anthropic_client as ac
from hevajra_matrix.llm.anthropic_client import FALLBACK_BETA, AnthropicClient, build_kwargs, map_sdk_error
from hevajra_matrix.llm.client import ConfigurationError, LLMRequest, LLMTransportError
from hevajra_matrix.llm.schema import strict

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {
    "thinking",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "stop_sequences",
    "cache_control",
}


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


# -------------------------------------------------------------------------- build_kwargs


def test_build_kwargs_sends_only_the_allowed_parameters():
    kwargs = build_kwargs(_request())
    assert set(kwargs) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert not FORBIDDEN & set(kwargs)
    assert kwargs["model"] == "claude-opus-5-5" and kwargs["max_tokens"] == 128000
    assert kwargs["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": _request().schema},
    }


def test_single_user_turn_without_prefill_and_cache_breakpoints_on_the_stable_prefix():
    kwargs = build_kwargs(_request())
    assert kwargs["system"] == [
        {"type": "text", "text": "SYSTEM-TEXT", "cache_control": {"type": "ephemeral"}}
    ]
    assert [m["role"] for m in kwargs["messages"]] == ["user"]
    context, body = kwargs["messages"][0]["content"]
    assert context == {"type": "text", "text": "CONTEXT-TEXT", "cache_control": {"type": "ephemeral"}}
    assert body == {"type": "text", "text": "BODY-TEXT"}


def test_without_context_the_body_is_the_only_user_block():
    content = build_kwargs(_request(context=None))["messages"][0]["content"]
    assert content == [{"type": "text", "text": "BODY-TEXT"}]


def test_fallback_parameters_only_when_allowed():
    assert "betas" not in build_kwargs(_request()) and "fallbacks" not in build_kwargs(_request())
    kwargs = build_kwargs(_request(allow_fallback=True))
    assert kwargs["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert kwargs["fallbacks"] == "default"
    assert not FORBIDDEN & set(kwargs)


def test_cache_key_material_is_never_sent():
    request = _request(task="TASK-MARK", prompt_sha="SHA-MARK", replicate="REPLICATE-MARK")
    sent = json.dumps(build_kwargs(request))
    assert "TASK-MARK" not in sent and "SHA-MARK" not in sent and "REPLICATE-MARK" not in sent


def test_build_kwargs_is_pure_and_shares_nothing_with_the_request():
    request = _request()
    key = request.key()
    kwargs = build_kwargs(request)
    kwargs["output_config"]["format"]["schema"]["properties"]["ok"]["type"] = "string"
    kwargs["system"][0]["cache_control"]["type"] = "mutated"
    assert request.key() == key
    assert build_kwargs(request) == build_kwargs(_request())


def test_the_api_receives_the_supported_schema_subset():
    schema = strict({"type": "object", "properties": {"q": {"type": "string", "maxLength": 10}}})
    sent = build_kwargs(_request(schema=schema))["output_config"]["format"]["schema"]
    assert sent["properties"]["q"] == {"type": "string", "description": "{maxLength: 10}"}


# ------------------------------------------------------------------ complete / count_tokens


class _Stream:
    def __init__(self, message: Any, request_id: str | None) -> None:
        self._message = message
        self.request_id = request_id

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> Any:
        return self._message


class FakeSDK:
    """Stands in for ``anthropic.Anthropic()``: records every call and its kwargs."""

    def __init__(self, message: Any = None, error: Exception | None = None) -> None:
        self.message = message
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.messages = SimpleNamespace(stream=self._plain_stream, count_tokens=self._count_tokens)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._beta_stream))

    def _call(self, path: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((path, kwargs))
        if self.error is not None:
            raise self.error

    def _plain_stream(self, **kwargs: Any) -> _Stream:
        self._call("messages.stream", kwargs)
        return _Stream(self.message, "req_fake_plain")

    def _beta_stream(self, **kwargs: Any) -> _Stream:
        self._call("beta.messages.stream", kwargs)
        return _Stream(self.message, "req_fake_beta")

    def _count_tokens(self, **kwargs: Any) -> Any:
        self._call("messages.count_tokens", kwargs)
        return SimpleNamespace(input_tokens=1234)


OK_MESSAGE = SimpleNamespace(
    model="claude-opus-5-5",
    stop_reason="end_turn",
    stop_details=None,
    content=[
        SimpleNamespace(type="thinking", thinking="", signature="s"),
        SimpleNamespace(type="text", text='{"ok": true}'),
    ],
    usage=SimpleNamespace(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
    ),
)


def test_complete_streams_the_built_kwargs_on_the_plain_endpoint():
    sdk = FakeSDK(OK_MESSAGE)
    request = _request()
    response = AnthropicClient(sdk=sdk).complete(request)
    assert sdk.calls == [("messages.stream", build_kwargs(request))]
    assert response.status == "ok" and response.usable
    assert response.key == request.key() and response.request_id == "req_fake_plain"
    assert response.created_at.endswith("+00:00")


def test_complete_uses_the_beta_endpoint_only_with_fallback():
    sdk = FakeSDK(OK_MESSAGE)
    request = _request(allow_fallback=True)
    AnthropicClient(sdk=sdk).complete(request)
    ((path, kwargs),) = sdk.calls
    assert path == "beta.messages.stream"
    assert kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"


def test_request_for_another_model_is_refused_before_any_call():
    sdk = FakeSDK(OK_MESSAGE)
    with pytest.raises(ConfigurationError, match="configured for"):
        AnthropicClient(model="claude-opus-5-5", sdk=sdk).complete(_request(model="claude-sonnet-5-5"))
    assert sdk.calls == []


def test_count_tokens_sends_the_request_without_generation_parameters():
    sdk = FakeSDK()
    request = _request(allow_fallback=True)
    assert AnthropicClient(sdk=sdk).count_tokens(request) == 1234
    ((path, kwargs),) = sdk.calls
    kw = build_kwargs(request)
    assert path == "messages.count_tokens"
    assert kwargs == {k: kw[k] for k in ("model", "system", "messages", "output_config")}


# ------------------------------------------------------------------------------- errors


class AnthropicError(Exception):
    pass


class APIStatusError(AnthropicError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# The slice of the SDK's exception hierarchy that map_sdk_error relies on.
BadRequestError = type("BadRequestError", (APIStatusError,), {})
AuthenticationError = type("AuthenticationError", (APIStatusError,), {})
PermissionDeniedError = type("PermissionDeniedError", (APIStatusError,), {})
NotFoundError = type("NotFoundError", (APIStatusError,), {})
RateLimitError = type("RateLimitError", (APIStatusError,), {})
InternalServerError = type("InternalServerError", (APIStatusError,), {})
APIConnectionError = type("APIConnectionError", (AnthropicError,), {})
APITimeoutError = type("APITimeoutError", (APIConnectionError,), {})


FAKE_ERRORS = SimpleNamespace(
    AnthropicError=AnthropicError,
    APIStatusError=APIStatusError,
    BadRequestError=BadRequestError,
    AuthenticationError=AuthenticationError,
    PermissionDeniedError=PermissionDeniedError,
    NotFoundError=NotFoundError,
    RateLimitError=RateLimitError,
    APIConnectionError=APIConnectionError,
)


@pytest.mark.parametrize(
    ("exc", "expected", "fragment"),
    [
        (AuthenticationError("bad key", 401), ConfigurationError, "ANTHROPIC_API_KEY"),
        (PermissionDeniedError("forbidden", 403), ConfigurationError, "PermissionDeniedError"),
        (NotFoundError("no such model", 404), ConfigurationError, "model"),
        (BadRequestError("invalid schema", 400), LLMTransportError, "(400)"),
        (RateLimitError("slow down", 429), LLMTransportError, "(429)"),
        (InternalServerError("oops", 500), LLMTransportError, "HTTP 500"),
        (APITimeoutError("timeout"), LLMTransportError, "APITimeoutError"),
        (AnthropicError("other"), LLMTransportError, "AnthropicError"),
    ],
)
def test_map_sdk_error_most_specific_first(exc, expected, fragment):
    mapped = map_sdk_error(exc, FAKE_ERRORS)
    assert type(mapped) is expected and fragment in str(mapped)


def test_non_sdk_exceptions_are_not_mapped():
    assert map_sdk_error(ValueError("x"), FAKE_ERRORS) is None
    assert map_sdk_error(AuthenticationError("x", 401), None) is None


def test_complete_and_count_tokens_translate_sdk_errors(monkeypatch):
    monkeypatch.setattr(ac, "_installed_sdk", lambda: FAKE_ERRORS)
    failing = AnthropicClient(sdk=FakeSDK(error=RateLimitError("slow down", 429)))
    with pytest.raises(LLMTransportError) as info:
        failing.complete(_request())
    assert isinstance(info.value.__cause__, RateLimitError)
    with pytest.raises(ConfigurationError):
        AnthropicClient(sdk=FakeSDK(error=AuthenticationError("bad", 401))).count_tokens(_request())


def test_foreign_exceptions_propagate_unchanged(monkeypatch):
    monkeypatch.setattr(ac, "_installed_sdk", lambda: FAKE_ERRORS)
    with pytest.raises(KeyError):
        AnthropicClient(sdk=FakeSDK(error=KeyError("bug"))).complete(_request())


# ------------------------------------------------------------------------ lazy SDK import


def test_missing_sdk_fails_on_first_call_not_on_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes "import anthropic" fail
    client = AnthropicClient()
    with pytest.raises(ConfigurationError, match=r"pip install"):
        client.complete(_request())


def test_llm_modules_import_without_the_sdk():
    code = (
        "import sys; sys.modules['anthropic'] = None\n"
        "import hevajra_matrix.llm.anthropic_client as ac\n"
        "import hevajra_matrix.llm.audit, hevajra_matrix.llm.cache, hevajra_matrix.llm.check\n"
        "import hevajra_matrix.llm.fake, hevajra_matrix.llm.schema\n"
        "ac.AnthropicClient()\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def test_importing_the_client_module_does_not_load_the_sdk():
    code = (
        "import sys, hevajra_matrix.llm.anthropic_client as ac\n"
        "ac.AnthropicClient()\n"
        "assert 'anthropic' not in sys.modules, 'anthropic imported eagerly'\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
