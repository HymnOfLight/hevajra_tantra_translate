"""Claude Opus 5.5 through the official Anthropic Python SDK.

Everything except the two methods of ``AnthropicClient`` is pure and unit-tested
without the SDK:

    build_kwargs(request)                     keyword arguments for ``messages.stream``
    to_response(message, request, key, now)   SDK message -> ``LLMResponse``
    map_sdk_error(exc, sdk)                   SDK exception -> ``LLMError`` (or None)

``anthropic`` is imported lazily on the first API call, so the package, the tests and
offline replays from the cache all work without it (``pip install .[claude]``).

Request shape (stable prefix first, for prompt caching)
    system    one text block with a cache breakpoint (task instructions)
    user      optional ``context`` block with a cache breakpoint, then the ``body`` block
Breakpoints are explicit rather than top-level automatic caching, because automatic
caching places the breakpoint after the last block, i.e. after the variable body, so
the large shared context (e.g. the full witness text) would never be reused across
windows.

Never sent: ``thinking`` (Opus 5.5 always thinks adaptively and rejects disabling
it), ``temperature``/``top_p``/``top_k`` (rejected), tools, ``tool_choice`` and
assistant prefill (rejected). Every call streams and keeps only the final message,
which is safe for large ``max_tokens``.

Server-side fallback (``request.allow_fallback``) goes through the beta endpoint with
``fallbacks="default"``. Whatever the path, a response served by another model is
flagged ``fallback_used`` and is never a measurement (see ``LLMResponse.usable``).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from . import schema as schema_mod
from .client import (
    DEFAULT_MODEL,
    ConfigurationError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTransportError,
    Usage,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
CACHE_BREAKPOINT = {"type": "ephemeral"}
# Output cut off before the model finished: the answer is incomplete, not wrong.
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


# --------------------------------------------------------------------------- request


def build_kwargs(request: LLMRequest) -> dict[str, Any]:
    """Return the keyword arguments for ``messages.stream`` (plus beta ones on fallback).

    Pure; the returned structure shares nothing with ``request``. ``replicate``,
    ``task`` and ``prompt_sha`` are cache-key material only and are never sent.
    """
    content: list[dict[str, Any]] = []
    if request.context:
        content.append({"type": "text", "text": request.context, "cache_control": dict(CACHE_BREAKPOINT)})
    content.append({"type": "text", "text": request.body})
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": [{"type": "text", "text": request.system, "cache_control": dict(CACHE_BREAKPOINT)}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": request.effort,
            "format": {"type": "json_schema", "schema": schema_mod.for_api(request.schema)},
        },
    }
    if request.allow_fallback:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


# -------------------------------------------------------------------------- response


def to_response(
    message: Any, request: LLMRequest, key: str, now: str, request_id: str | None = None
) -> LLMResponse:
    """Normalise one final SDK message (or a stand-in with the same attributes).

    Pure. ``message`` may be an SDK model, a ``SimpleNamespace`` or a plain dict.

    Status, decided on ``stop_reason`` (never on ``stop_details``, which may be null):
        refusal    ``"refusal"``; any partial output is discarded
        truncated  ``"max_tokens"`` or ``"model_context_window_exceeded"``
        ok         ``"end_turn"`` with a schema-valid JSON object
        invalid    ``"end_turn"`` with anything else, or any other stop reason

    Only ``text`` blocks are read (thinking and any other block types are ignored).
    After a server-side fallback the answer is taken from the text after the last
    ``fallback`` boundary block when that parses, else from all text (the fallback
    model may have continued the declined model's partial output).

    ``fallback_used`` is True when the served model differs from the requested one,
    when a ``fallback`` block is present, or when ``usage.iterations`` holds a
    ``fallback_message`` entry (the only signal on sticky-routed turns).
    """
    stop_reason = _get(message, "stop_reason") or ""
    served = _get(message, "model") or ""
    blocks = list(_get(message, "content") or ())
    usage_obj = _get(message, "usage")
    common: dict[str, Any] = {
        "key": key,
        "requested_model": request.model,
        "served_model": served,
        "stop_reason": stop_reason,
        "usage": _usage(usage_obj),
        "fallback_used": _fallback_used(served, request.model, blocks, usage_obj),
        "request_id": request_id or _get(message, "_request_id"),
        "created_at": now,
    }
    if stop_reason == "refusal":
        details = _get(message, "stop_details")
        return LLMResponse(
            status="refusal",
            data=None,
            raw_text="",
            refusal_category=_get(details, "category"),
            refusal_explanation=_get(details, "explanation"),
            **common,
        )
    text, data = _answer(blocks, request.schema)
    if stop_reason in TRUNCATION_STOP_REASONS:
        return LLMResponse(status="truncated", data=None, raw_text=text, **common)
    if stop_reason != "end_turn" or data is None:
        return LLMResponse(status="invalid", data=None, raw_text=text, **common)
    return LLMResponse(status="ok", data=data, raw_text=text, **common)


def parse_answer(text: str, schema: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse ``text`` as one JSON object and validate it; return (data, errors).

    ``data`` is None whenever ``errors`` is non-empty. Exposed so that callers can
    explain an ``invalid`` response from its (privately cached) ``raw_text``.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        return None, [f"not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, ["top-level JSON value is not an object"]
    errors = schema_mod.validate(data, schema)
    return (None, errors) if errors else (data, [])


def _answer(blocks: list[Any], schema: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Pick the answer text among the text blocks and parse it (data None if invalid).

    The text after the last ``fallback`` block is preferred; if it does not parse, the
    whole text is tried, since a fallback model may continue a declined partial answer.
    """
    boundary = max((i for i, b in enumerate(blocks) if _get(b, "type") == "fallback"), default=-1)
    texts = [(i, _get(b, "text") or "") for i, b in enumerate(blocks) if _get(b, "type") == "text"]
    whole = "".join(t for _, t in texts)
    after = "".join(t for i, t in texts if i > boundary)
    data, _ = parse_answer(after, schema)
    if data is not None or after == whole:
        return after, data
    return whole, parse_answer(whole, schema)[0]


def _fallback_used(served: str, requested: str, blocks: list[Any], usage: Any) -> bool:
    return (
        served != requested
        or any(_get(b, "type") == "fallback" for b in blocks)
        or any(_get(it, "type") == "fallback_message" for it in (_get(usage, "iterations") or ()))
    )


def _usage(usage: Any) -> Usage:
    """Token usage of the whole call (None -> 0).

    With ``usage.iterations`` (fallback, compaction) the top-level fields cover only
    the attempt that produced the message, so the iterations are summed instead; this
    keeps the budget estimate conservative when a declined attempt was also billed.
    """
    parts = list(_get(usage, "iterations") or ()) or [usage]
    return Usage(**{name: sum(int(_get(part, name) or 0) for part in parts) for name in _USAGE_FIELDS})


def _get(obj: Any, name: str) -> Any:
    """Attribute or mapping access that tolerates None and missing fields."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


# ---------------------------------------------------------------------------- errors


def map_sdk_error(exc: BaseException, sdk: Any) -> LLMError | None:
    """Translate an exception raised by the SDK ``sdk`` (the ``anthropic`` module).

    Most specific first. Credentials, permissions and unknown models are
    ``ConfigurationError`` (fix the setup); every other API failure, including a
    rejected request (400) and rate limits that outlived the SDK's own retries, is
    ``LLMTransportError``. Returns None when ``exc`` is not an SDK exception.
    """
    if sdk is None or not isinstance(exc, sdk.AnthropicError):
        return None
    setup_errors = tuple(
        cls
        for cls in (
            getattr(sdk, "AuthenticationError", None),
            getattr(sdk, "PermissionDeniedError", None),
            getattr(sdk, "NotFoundError", None),
            getattr(sdk, "CredentialsError", None),
        )
        if cls is not None
    )
    if setup_errors and isinstance(exc, setup_errors):
        return ConfigurationError(
            f"Anthropic API setup problem ({type(exc).__name__}): {exc}. "
            "Check ANTHROPIC_API_KEY and the model in config/llm.yaml."
        )
    if isinstance(exc, sdk.BadRequestError):
        return LLMTransportError(f"request rejected by the API (400): {exc}")
    if isinstance(exc, sdk.RateLimitError):
        return LLMTransportError(f"rate limited after the SDK's retries (429): {exc}")
    if isinstance(exc, sdk.APIStatusError):
        return LLMTransportError(f"API error (HTTP {getattr(exc, 'status_code', '?')}): {exc}")
    if isinstance(exc, sdk.APIConnectionError):
        return LLMTransportError(f"could not reach the API ({type(exc).__name__}): {exc}")
    return LLMTransportError(f"Anthropic SDK error ({type(exc).__name__}): {exc}")


def _installed_sdk() -> Any:
    """The ``anthropic`` module, or None when it is not installed (imported lazily)."""
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic


# ---------------------------------------------------------------------------- client


class AnthropicClient:
    """``LLMClient`` backed by the Anthropic API (streaming, structured output).

    ``sdk`` injects a ready SDK client (or a test double exposing
    ``messages.stream``, ``beta.messages.stream`` and ``messages.count_tokens``);
    by default ``anthropic.Anthropic`` is created on first use, with credentials
    resolved by the SDK (e.g. ``ANTHROPIC_API_KEY``). Thread-safe.

    ``model`` is the configured model; a request for any other model is refused
    with ``ConfigurationError`` so that a request builder cannot silently bypass
    ``config/llm.yaml``.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = 4,
        timeout_s: float = 900.0,
        sdk: Any = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._sdk = sdk
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._client_for(request)
        kwargs = build_kwargs(request)
        stream = client.beta.messages.stream if request.allow_fallback else client.messages.stream
        try:
            with stream(**kwargs) as events:
                message = events.get_final_message()
                request_id = getattr(events, "request_id", None)
        except Exception as exc:
            mapped = map_sdk_error(exc, _installed_sdk())
            if mapped is None:
                raise
            raise mapped from exc
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return to_response(message, request, request.key(), now, request_id=request_id)

    def count_tokens(self, request: LLMRequest) -> int:
        """Input tokens of ``request`` (system + context + body + schema), for dry-run costing.

        Free of charge but needs credentials; never generates output.
        """
        client = self._client_for(request)
        kwargs = build_kwargs(request)
        counted = {k: kwargs[k] for k in ("model", "system", "messages", "output_config")}
        try:
            result = client.messages.count_tokens(**counted)
        except Exception as exc:
            mapped = map_sdk_error(exc, _installed_sdk())
            if mapped is None:
                raise
            raise mapped from exc
        return int(_get(result, "input_tokens") or 0)

    def _client_for(self, request: LLMRequest) -> Any:
        if request.model != self.model:
            raise ConfigurationError(
                f"request for {request.model!r} but the client is configured for {self.model!r}"
            )
        with self._lock:
            if self._sdk is None:
                anthropic = _installed_sdk()
                if anthropic is None:
                    raise ConfigurationError(
                        "the Anthropic SDK is not installed; run: pip install 'hevajra-matrix[claude]'"
                    )
                try:
                    self._sdk = anthropic.Anthropic(max_retries=self.max_retries, timeout=self.timeout_s)
                except anthropic.AnthropicError as exc:
                    raise ConfigurationError(f"cannot create the Anthropic client: {exc}") from exc
            return self._sdk
