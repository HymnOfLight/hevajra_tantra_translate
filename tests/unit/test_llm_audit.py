"""llm.audit: text-free JSONL audit log and the spending cap."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hevajra_matrix.llm.audit import PRICING_KEYS, AuditedClient, estimate_usd, recorded_spend
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.client import (
    BudgetExceeded,
    ConfigurationError,
    LLMRequest,
    LLMTransportError,
    Usage,
    canonical_json,
    sha256_text,
)
from hevajra_matrix.llm.fake import FakeClient, ok_response, refusal_response
from hevajra_matrix.llm.schema import strict

PRICING = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}  # config/llm.yaml
USAGE = Usage(
    input_tokens=10_000,
    output_tokens=2_000,
    cache_read_input_tokens=50_000,
    cache_creation_input_tokens=1_000,
)
USAGE_USD = (10_000 * 4.0 + 2_000 * 20.0 + 50_000 * 0.20 + 1_000 * 5.0) / 1e6  # 0.095
SCHEMA = strict({"type": "object", "properties": {"answer": {"type": "string"}}})
SECRET_PROMPT = ("SYSTEM-SECRET-TEXT", "CONTEXT-SECRET-TEXT", "BODY-SECRET-TEXT")
SECRET_ANSWER = "ANSWER-SECRET-TEXT"
FIELDS = {
    "ts",
    "run_id",
    "task",
    "replicate",
    "prompt_sha",
    "schema_sha",
    "effort",
    "key",
    "requested_model",
    "served_model",
    "fallback_used",
    "status",
    "stop_reason",
    "refusal_category",
    "usage",
    "request_id",
    "from_cache",
    "seconds",
    "est_usd",
}


def _request(replicate: str = "r1") -> LLMRequest:
    system, context, body = SECRET_PROMPT
    return LLMRequest(
        task="collate",
        prompt_sha="p" * 64,
        system=system,
        context=context,
        body=body,
        schema=SCHEMA,
        effort="high",
        max_tokens=64000,
        replicate=replicate,
    )


def _answer(req: LLMRequest):
    return dataclasses.replace(ok_response(req, {"answer": SECRET_ANSWER}, usage=USAGE), request_id="req_1")


class Clock:
    """Advances 2.5 s per reading, starting at 2026-01-01T00:00:00Z."""

    def __init__(self) -> None:
        self.now = 1767225600.0

    def __call__(self) -> float:
        value, self.now = self.now, self.now + 2.5
        return value


def _stack(tmp_path: Path, script=_answer, budget: float = 100.0):
    inner = FakeClient(script)
    audited = AuditedClient(
        CachedClient(inner, tmp_path / "cache"),
        tmp_path / "audit.jsonl",
        PRICING,
        budget_usd=budget,
        run_id="run-1",
        clock=Clock(),
    )
    return audited, inner


def _lines(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()]


def test_estimate_usd():
    assert estimate_usd(USAGE, PRICING) == pytest.approx(USAGE_USD)
    assert estimate_usd(Usage(), PRICING) == 0.0


def test_one_line_per_call_including_cache_hits(tmp_path: Path):
    audited, inner = _stack(tmp_path)
    request = _request()
    audited.complete(request)
    audited.complete(request)
    first, hit = _lines(tmp_path)
    assert len(inner.requests) == 1
    assert set(first) == FIELDS
    assert first == {
        "ts": "2026-01-01T00:00:00+00:00",
        "run_id": "run-1",
        "task": "collate",
        "replicate": "r1",
        "prompt_sha": "p" * 64,
        "schema_sha": sha256_text(canonical_json(SCHEMA)),
        "effort": "high",
        "key": request.key(),
        "requested_model": "claude-opus-5-5",
        "served_model": "claude-opus-5-5",
        "fallback_used": False,
        "status": "ok",
        "stop_reason": "end_turn",
        "refusal_category": None,
        "usage": {"in": 10_000, "out": 2_000, "cache_read": 50_000, "cache_write": 1_000},
        "request_id": "req_1",
        "from_cache": False,
        "seconds": 2.5,
        "est_usd": pytest.approx(USAGE_USD),
    }
    assert (
        hit["from_cache"] is True and hit["key"] == first["key"] and hit["ts"] == "2026-01-01T00:00:05+00:00"
    )


def test_log_never_contains_prompt_or_response_text(tmp_path: Path):
    audited, _ = _stack(tmp_path, script=lambda r: refusal_response(r, "cyber", "EXPLANATION-SECRET-TEXT"))
    audited.complete(_request("r1"))
    audited, _ = _stack(tmp_path)
    audited.complete(_request("r2"))
    audited.complete(_request("r2"))
    text = (tmp_path / "audit.jsonl").read_text("utf-8")
    for secret in (*SECRET_PROMPT, SECRET_ANSWER, "EXPLANATION-SECRET-TEXT"):
        assert secret not in text
    assert [line["status"] for line in _lines(tmp_path)] == ["refusal", "ok", "ok"]
    assert _lines(tmp_path)[0]["refusal_category"] == "cyber"


def test_budget_stops_new_uncached_calls_but_still_serves_the_cache(tmp_path: Path):
    audited, inner = _stack(tmp_path, budget=0.15)  # room for two calls of 0.095
    audited.complete(_request("r1"))
    audited.complete(_request("r1"))  # a hit costs nothing
    assert audited.spent_usd == pytest.approx(USAGE_USD)
    audited.complete(_request("r2"))  # 0.095 < 0.15: allowed, now 0.19
    with pytest.raises(BudgetExceeded):
        audited.complete(_request("r3"))
    assert len(inner.requests) == 2
    assert audited.complete(_request("r1")).from_cache  # cached work stays available
    assert len(_lines(tmp_path)) == 4  # the refused call wrote no line


def test_spend_recorded_by_earlier_runs_counts_against_the_budget(tmp_path: Path):
    audited, _ = _stack(tmp_path, budget=0.10)
    audited.complete(_request("r1"))
    audited.complete(_request("r1"))
    assert recorded_spend(tmp_path / "audit.jsonl") == pytest.approx(USAGE_USD)
    restarted, inner = _stack(tmp_path, budget=0.10)
    assert restarted.spent_usd == pytest.approx(USAGE_USD)
    restarted.complete(_request("r2"))  # 0.095 < 0.10 still allowed
    with pytest.raises(BudgetExceeded):
        restarted.complete(_request("r3"))
    assert recorded_spend(tmp_path / "missing.jsonl") == 0.0


def test_failed_calls_write_no_line(tmp_path: Path):
    def fail(req):
        raise LLMTransportError("synthetic")

    audited, _ = _stack(tmp_path, script=fail)
    with pytest.raises(LLMTransportError):
        audited.complete(_request())
    assert not (tmp_path / "audit.jsonl").exists()


def test_inner_without_is_cached_is_treated_as_uncached(tmp_path: Path):
    audited = AuditedClient(FakeClient(_answer), tmp_path / "a.jsonl", PRICING, budget_usd=0.0, run_id="x")
    assert audited.is_cached(_request()) is False
    with pytest.raises(BudgetExceeded):
        audited.complete(_request())


def test_pricing_must_name_every_rate(tmp_path: Path):
    assert PRICING_KEYS == ("input", "output", "cache_read", "cache_write")
    with pytest.raises(ConfigurationError, match="cache_write"):
        AuditedClient(
            FakeClient(_answer),
            tmp_path / "a.jsonl",
            {"input": 4, "output": 20, "cache_read": 0.2},
            budget_usd=1,
            run_id="x",
        )
