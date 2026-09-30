"""``claude-check``: one tiny call proving that the key, the model and the response path work.

The request is deliberately trivial (a one-field strict schema, effort ``low``, no
fallback) so it costs a fraction of a cent. Send it through ``AnthropicClient``
(optionally wrapped in ``AuditedClient``) rather than through the cache: a cached
answer proves nothing about the live API, which is why the summary reports
``from_cache``.
"""

from __future__ import annotations

from typing import Any

from .client import DEFAULT_MODEL, Effort, LLMClient, LLMRequest, sha256_text
from .schema import strict

CHECK_TASK = "check"
CHECK_SYSTEM = (
    "You are the connectivity check of a research pipeline. Answer with the requested JSON object only."
)
CHECK_BODY = 'Return the JSON object {"ok": true}.'


def build_check_request(
    model: str = DEFAULT_MODEL, effort: Effort = "low", max_tokens: int = 1024, replicate: str = "r1"
) -> LLMRequest:
    """The check request; defaults match ``config/llm.yaml: tasks.check``."""
    return LLMRequest(
        task=CHECK_TASK,
        prompt_sha=sha256_text(CHECK_SYSTEM + "\n" + CHECK_BODY),
        system=CHECK_SYSTEM,
        body=CHECK_BODY,
        schema=strict({"type": "object", "properties": {"ok": {"type": "boolean"}}}),
        effort=effort,
        max_tokens=max_tokens,
        replicate=replicate,
        allow_fallback=False,
        model=model,
    )


def run_check(client: LLMClient, request: LLMRequest | None = None) -> dict[str, Any]:
    """Send the check request and summarise the outcome (no response text).

    ``passed`` is True only for a schema-valid ``{"ok": true}`` served by the
    requested model. Transport and configuration errors propagate as exceptions.
    """
    request = request or build_check_request()
    response = client.complete(request)
    return {
        "passed": response.usable and response.data == {"ok": True},
        "status": response.status,
        "requested_model": response.requested_model,
        "served_model": response.served_model,
        "fallback_used": response.fallback_used,
        "stop_reason": response.stop_reason,
        "refusal_category": response.refusal_category,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
        },
        "request_id": response.request_id,
        "from_cache": response.from_cache,
    }
