"""The LLM boundary: request/response value objects and the ``LLMClient`` protocol.

Every task (collation, components, topics, the experiment) builds an ``LLMRequest``
and receives an ``LLMResponse``; none of them touches the Anthropic SDK directly.
Concrete clients are composed once in the pipeline:

    AuditedClient(CachedClient(AnthropicClient(...)))    # production
    AuditedClient(CachedClient(FakeClient(...)))         # tests

Design notes
    * A refusal, a truncation and a schema-invalid answer are *values*
      (``LLMResponse.status``), not exceptions, because they are data the research
      must count. Exceptions are reserved for transport, configuration, cache misses
      in offline mode and budget exhaustion.
    * Claude Opus 5.5 accepts no sampling parameters, so identical requests may
      return different answers. ``replicate`` is part of the cache key (and never sent
      to the API): replicates measure instrument stability, while the cache makes any
      finished run exactly replayable offline.
    * ``substituted_model`` is True when the served model differs from the requested
      one (server-side fallback). Such output is a reviewer hint only and must never
      enter a measurement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Protocol

DEFAULT_MODEL = "claude-opus-5-5"

Effort = Literal["low", "medium", "high", "xhigh", "max"]
ResponseStatus = Literal["ok", "refusal", "truncated", "invalid"]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing and for prompts (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LLMRequest:
    """One single-turn, stateless, structured-output request.

    ``system`` and ``context`` are the stable, cacheable prefix (static instructions,
    then e.g. the full witness text); ``body`` is the variable part. ``schema`` is a
    strict JSON schema (see ``llm.schema.strict``). ``prompt_sha`` identifies the
    template version so that editing a prompt changes every cache key.
    """

    task: str
    prompt_sha: str
    system: str
    body: str
    schema: Mapping[str, Any]
    effort: Effort
    max_tokens: int
    context: str | None = None
    replicate: str = "r1"
    allow_fallback: bool = False
    model: str = DEFAULT_MODEL

    def key(self) -> str:
        """Content address of the request: sha256 over every field."""
        return sha256_text(canonical_json(asdict(self)))


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    """Normalised outcome of one request.

    ``data`` is the parsed JSON object when ``status == "ok"``; otherwise None.
    ``raw_text`` is kept for the private cache only and is never written to the
    audit log or to committed files (it may contain licence-restricted text).
    """

    key: str
    status: ResponseStatus
    data: Mapping[str, Any] | None
    raw_text: str
    requested_model: str
    served_model: str
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    fallback_used: bool = False
    refusal_category: str | None = None
    refusal_explanation: str | None = None
    request_id: str | None = None
    created_at: str = ""
    from_cache: bool = False

    @property
    def substituted_model(self) -> bool:
        return self.fallback_used or self.served_model != self.requested_model

    @property
    def usable(self) -> bool:
        """True only for a schema-valid answer from the requested model."""
        return self.status == "ok" and not self.substituted_model


class LLMClient(Protocol):
    """Anything that turns an ``LLMRequest`` into an ``LLMResponse``."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class LLMError(Exception):
    """Base class for LLM-layer failures that are not research data."""


class LLMTransportError(LLMError):
    """The API could not be reached or returned an unrecoverable HTTP error."""


class CacheMiss(LLMError):
    """Offline mode was requested and the response is not in the cache."""


class BudgetExceeded(LLMError):
    """A new, uncached call would exceed the configured spending cap."""


class ConfigurationError(LLMError):
    """The SDK is missing or the client is misconfigured."""
