"""How one ``claude -p --output-format json`` result becomes an ``LLMResponse`` (pure).

Split from ``llm.claude_code_client`` (which runs the CLI) so that the mapping of the CLI's
outcomes - answers, refusals, truncations, invalid answers and run failures - is in one place
and can be checked against the CLI's actual result shapes without running it. The facts it
relies on were checked against Claude Code 2.1.296:

    * ``is_error`` is set on some answers too (``error_max_turns``,
      ``error_max_structured_output_retries``, a capped answer with ``api_error``
      ``max_output_tokens``), so the outcome is classified before deciding to raise;
    * a capped answer carries a synthetic ``stop_reason`` ``"stop_sequence"``;
    * ``stop_reason`` is carried over from earlier messages, so it is trusted only on a run
      that answered (``ANSWER_SUBTYPES``);
    * the plan's limits come as ``api_error`` ``usage_limit_reached`` (or
      ``model_requires_usage_credits``) with messages like "You've hit your session limit";
      a transient 429 says "not your usage limit".
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from . import schema as schema_mod
from .client import LLMRequest, LLMResponse, LLMTransportError, Usage, canonical_json

# Subtypes of a run that produced an answer. Only these can be research data; any other
# subtype (error_during_execution, error_max_budget_usd, ...) is a run failure, whatever
# stop_reason the CLI carried over from an earlier message.
ANSWER_SUBTYPES = frozenset({"success", "error_max_turns", "error_max_structured_output_retries"})
# Answered, but not validly (the CLI sets is_error on them too): cached as "invalid".
INVALID_SUBTYPES = frozenset({"error_max_turns", "error_max_structured_output_retries"})
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
# The CLI reports a capped answer (output or context limit) as this api_error, with is_error
# and a synthetic stop_reason "stop_sequence", and the text "... output token maximum".
TRUNCATION_API_ERROR = "max_output_tokens"
TRUNCATION_TEXT = "output token maximum"
# api_error values after which the same request cannot succeed now: not retried.
USAGE_LIMIT_API_ERRORS = frozenset({"usage_limit_reached", "model_requires_usage_credits"})
MODEL_UNAVAILABLE_API_ERRORS = frozenset({"model_substitution_disabled", "no_allowed_fallback"})
# Text fallback for the plan's limit messages ("You've hit your session limit", "... weekly /
# Opus limit", "You've reached your ... limit", "out of usage credits", "Opus requires usage
# credits"); a transient 429 says "not your usage limit" and is retried.
USAGE_LIMIT_TEXT = re.compile(r"(?:hit|reached) your [\w'\u2019 ]*(?:limit|budget)|out of (?:usage|extra usage)"
                              r"|requires usage credits|usage limit reached", re.IGNORECASE)
NOT_USAGE_LIMIT_TEXT = "not your usage limit"

class UsageLimitReached(LLMTransportError):
    """The subscription's usage allowance is exhausted for now. Finished calls are cached:
    rerun the same command after the limit resets and it continues where it stopped."""


class ModelUnavailable(LLMTransportError):
    """The requested model is not available to this login now and fallback is disabled.
    Not retried."""


def to_response(result: Mapping[str, Any], request: LLMRequest, key: str, now: str,
                provenance: Mapping[str, Any] | None = None) -> LLMResponse:
    """Normalise the JSON result of one ``claude -p --output-format json`` run.

    Only a run that answered (``ANSWER_SUBTYPES``) can be research data; it is classified
    in this order (the CLI sets ``is_error`` on several of these answers too):
        refusal    ``stop_reason == "refusal"`` or ``safety_stops > 0``
        truncated  ``stop_reason`` is ``max_tokens`` / ``model_context_window_exceeded``, or
                   ``api_error == "max_output_tokens"`` (or, with ``is_error`` and no
                   ``api_error``, the CLI's "output token maximum" message); stored with
                   ``stop_reason`` ``max_tokens`` as on the API backend
        invalid    subtype ``error_max_turns`` / ``error_max_structured_output_retries``
        (raised)   ``is_error`` otherwise: a run failure (``_failure``), never cached
        ok         a ``structured_output`` that passes the schema
        invalid    any other success
    Every other subtype is raised as a run failure. ``provenance`` (CLI version, login
    method, seconds) goes into ``client_info`` with the run's own text-free facts.
    """
    subtype = str(result.get("subtype") or "")
    stop_reason = str(result.get("stop_reason") or "")
    api_error = str(result.get("api_error") or "")
    text = str(result.get("result") or "")
    is_error = bool(result.get("is_error"))
    answered = subtype in ANSWER_SUBTYPES
    refused = answered and (stop_reason == "refusal" or int(result.get("safety_stops") or 0) > 0)
    truncated = answered and (stop_reason in TRUNCATION_STOP_REASONS or api_error == TRUNCATION_API_ERROR
                              or (is_error and not api_error and TRUNCATION_TEXT in text.lower()))
    if not answered or (is_error and not (refused or truncated or subtype in INVALID_SUBTYPES)):
        raise failure(result)
    served, fallback_used = served_model(result.get("modelUsage") or {}, request.model)
    output = result.get("structured_output")
    usage = _usage(result.get("usage") or {})
    info = {k: v for k, v in {**(provenance or {}), "subtype": subtype, "num_turns": result.get("num_turns"),
                              "terminal_reason": result.get("terminal_reason"), "api_error": api_error or None,
                              "cli_stop_reason": stop_reason or None,
                              "output_over_max_tokens": usage.output_tokens > request.max_tokens or None}.items()
            if v is not None}
    common: dict[str, Any] = {
        "key": key, "raw_text": canonical_json(output) if output is not None else text,
        "requested_model": request.model, "served_model": served,
        "stop_reason": "max_tokens" if truncated and stop_reason not in TRUNCATION_STOP_REASONS else (
            stop_reason or subtype),
        "usage": usage, "fallback_used": fallback_used, "request_id": result.get("session_id"),
        "created_at": now, "client_info": info,
    }
    if refused:
        return LLMResponse(status="refusal", data=None, refusal_category=None, **common)
    if truncated:
        return LLMResponse(status="truncated", data=None, **common)
    if subtype == "success" and isinstance(output, dict) and not schema_mod.validate(output, request.schema):
        return LLMResponse(status="ok", data=output, **common)
    return LLMResponse(status="invalid", data=None, **common)


def same_model(name: str, requested: str) -> bool:
    """``name`` is ``requested``, possibly with a date suffix (``-20260101``) and/or a context
    tag (``[1m]``); never another model whose id merely starts with ``requested``."""
    return re.fullmatch(rf"{re.escape(requested)}(-\d{{8}})?(\[[^\]]+\])?", name) is not None


def served_model(model_usage: Mapping[str, Any], requested: str) -> tuple[str, bool]:
    """(served model, fallback used) from the result's per-model usage.

    The served model is the one with the most output tokens. Fallback is reported when
    any other model produced output, or when no model usage is reported at all.
    """
    produced = {name: int((stats or {}).get("outputTokens") or 0) for name, stats in model_usage.items()}
    if not produced:
        return "unknown", True
    served = max(produced, key=lambda name: (produced[name], same_model(name, requested)))
    others = any(n > 0 for name, n in produced.items() if not same_model(name, requested))
    return (requested if same_model(served, requested) else served), others


def _usage(usage: Mapping[str, Any]) -> Usage:
    return Usage(input_tokens=int(usage.get("input_tokens") or 0), output_tokens=int(usage.get("output_tokens") or 0),
                 cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
                 cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0))


def is_usage_limit(result: Mapping[str, Any], detail: str) -> bool:
    """The plan's allowance is exhausted (not a transient 429, which is retried)."""
    if result.get("api_error") in USAGE_LIMIT_API_ERRORS:
        return True
    return NOT_USAGE_LIMIT_TEXT not in detail.lower() and USAGE_LIMIT_TEXT.search(detail) is not None


def failure(result: Mapping[str, Any]) -> LLMTransportError:
    detail = " ".join(str(x) for x in [result.get("result") or "", *(result.get("errors") or [])]).strip()
    message = f"claude -p failed ({result.get('subtype')}): {detail[:300] or 'no detail'}"
    if is_usage_limit(result, detail):
        return UsageLimitReached(message + "; finished calls are cached, rerun after the limit resets")
    if result.get("api_error") in MODEL_UNAVAILABLE_API_ERRORS:
        return ModelUnavailable(message + "; the requested model is unavailable to this login and fallback is off")
    return LLMTransportError(message)
