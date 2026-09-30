"""Audit log and spending cap for Claude calls.

One JSONL line per completed call, cache hits included:

    ts, run_id, task, replicate, prompt_sha, schema_sha, effort, key, requested_model,
    served_model, fallback_used, status, stop_reason, refusal_category,
    usage{in, out, cache_read, cache_write}, request_id, from_cache, seconds, est_usd

The log never contains prompt or response text (licence rule), so it can be committed.
It shows, for every published number, which model served which request and at what
estimated cost. A call that raises (transport error, offline cache miss, budget stop)
writes no line.

Budget. Before a call that is not already cached, the client raises
``BudgetExceeded`` once the estimated spend has reached ``budget_usd``. Spend counts
only calls not served from the cache, and it includes what earlier runs recorded in
the same log file, so restarting a run does not reset the cap. With concurrent
workers the cap can be overshot by the calls already in flight.

Cost estimate. Tokens times ``pricing`` in USD per million tokens, with the keys
``input``, ``output``, ``cache_read`` and ``cache_write`` (``config/llm.yaml:
pricing_usd_per_mtok``). A fallback model bills at its own rates, so substituted
calls are estimated at the configured model's prices.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .client import (
    BudgetExceeded,
    ConfigurationError,
    LLMClient,
    LLMRequest,
    LLMResponse,
    Usage,
    canonical_json,
    sha256_text,
)

PRICING_KEYS = ("input", "output", "cache_read", "cache_write")


def estimate_usd(usage: Usage, pricing: Mapping[str, float]) -> float:
    """Estimated cost of ``usage`` in USD (``pricing`` is USD per million tokens)."""
    return (
        usage.input_tokens * pricing["input"]
        + usage.output_tokens * pricing["output"]
        + usage.cache_read_input_tokens * pricing["cache_read"]
        + usage.cache_creation_input_tokens * pricing["cache_write"]
    ) / 1_000_000


def audit_record(
    request: LLMRequest, response: LLMResponse, *, run_id: str, ts: str, seconds: float, est_usd: float
) -> dict[str, Any]:
    """The audit line for one call: identifiers, outcome and usage, never any text."""
    return {
        "ts": ts,
        "run_id": run_id,
        "task": request.task,
        "replicate": request.replicate,
        "prompt_sha": request.prompt_sha,
        "schema_sha": sha256_text(canonical_json(request.schema)),
        "effort": request.effort,
        "key": request.key(),
        "requested_model": request.model,
        "served_model": response.served_model,
        "fallback_used": response.fallback_used,
        "status": response.status,
        "stop_reason": response.stop_reason,
        "refusal_category": response.refusal_category,
        "usage": {
            "in": response.usage.input_tokens,
            "out": response.usage.output_tokens,
            "cache_read": response.usage.cache_read_input_tokens,
            "cache_write": response.usage.cache_creation_input_tokens,
        },
        "request_id": response.request_id,
        "from_cache": response.from_cache,
        "seconds": round(seconds, 3),
        "est_usd": round(est_usd, 6),
    }


class AuditedClient:
    """``LLMClient`` decorator that logs every call and enforces the budget. Thread-safe.

    ``clock`` returns epoch seconds (default ``time.time``); tests pass a fake one.
    """

    def __init__(
        self,
        inner: LLMClient,
        log_path: Path,
        pricing: Mapping[str, float],
        budget_usd: float,
        run_id: str,
        clock: Callable[[], float] | None = None,
    ) -> None:
        missing = [k for k in PRICING_KEYS if k not in pricing]
        if missing:
            raise ConfigurationError(
                f"pricing lacks {missing}; expected USD per MTok for {list(PRICING_KEYS)}"
            )
        self.inner = inner
        self.log_path = Path(log_path)
        self.pricing = {k: float(pricing[k]) for k in PRICING_KEYS}
        self.budget_usd = float(budget_usd)
        self.run_id = run_id
        self.clock = clock or time.time
        self.spent_usd = recorded_spend(self.log_path)
        self._lock = threading.Lock()

    def is_cached(self, request: LLMRequest) -> bool:
        is_cached = getattr(self.inner, "is_cached", None)
        return bool(is_cached(request)) if is_cached is not None else False

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.is_cached(request):
            with self._lock:
                if self.spent_usd >= self.budget_usd:
                    raise BudgetExceeded(
                        f"estimated spend ${self.spent_usd:.2f} has reached the budget of "
                        f"${self.budget_usd:.2f}; uncached {request.task!r} request not sent"
                    )
        started = self.clock()
        response = self.inner.complete(request)
        seconds = self.clock() - started
        cost = estimate_usd(response.usage, self.pricing)
        ts = datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds")
        record = audit_record(request, response, run_id=self.run_id, ts=ts, seconds=seconds, est_usd=cost)
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._lock:
            if not response.from_cache:
                self.spent_usd += cost
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return response


def recorded_spend(log_path: Path) -> float:
    """Sum of ``est_usd`` over the non-cached calls already in the audit log (0 if absent)."""
    if not log_path.is_file():
        return 0.0
    total = 0.0
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not record.get("from_cache"):
                    total += float(record.get("est_usd") or 0.0)
    return total
