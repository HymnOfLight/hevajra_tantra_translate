"""llm.cache: content-addressed, atomic, offline-replayable response cache."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hevajra_matrix.llm.cache import CachedClient, response_from_dict, response_to_dict
from hevajra_matrix.llm.client import CacheMiss, LLMError, LLMRequest, LLMTransportError, Usage
from hevajra_matrix.llm.fake import (
    FakeClient,
    invalid_response,
    ok_response,
    refusal_response,
    substituted_response,
    truncated_response,
)
from hevajra_matrix.llm.schema import strict

SCHEMA = strict({"type": "object", "properties": {"n": {"type": "integer"}}})
BASE = LLMRequest(
    task="collate",
    prompt_sha="p1",
    system="SYSTEM-TEXT",
    context="CONTEXT-TEXT",
    body="BODY-TEXT",
    schema=SCHEMA,
    effort="high",
    max_tokens=64000,
)


class CountingClient(FakeClient):
    def __init__(self, script=lambda req: {"n": 1}) -> None:
        super().__init__(script)


def test_miss_calls_inner_and_writes_one_file_at_the_documented_path(tmp_path: Path):
    inner = CountingClient()
    cache = CachedClient(inner, tmp_path)
    response = cache.complete(BASE)
    key = BASE.key()
    path = tmp_path / "claude-opus-5-5" / key[:2] / f"{key}.json"
    assert cache.path_for(BASE) == path and path.is_file() and cache.is_cached(BASE)
    assert response.from_cache is False and len(inner.requests) == 1
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [path]  # no temporary files left


def test_hit_returns_the_stored_response_without_calling_inner(tmp_path: Path):
    inner = CountingClient()
    cache = CachedClient(inner, tmp_path)
    first = cache.complete(BASE)
    second = cache.complete(BASE)
    assert len(inner.requests) == 1
    assert second.from_cache is True
    assert dataclasses.replace(second, from_cache=False) == first


@pytest.mark.parametrize(
    "change",
    [
        {"model": "claude-sonnet-5-5"},
        {"effort": "low"},
        {"body": "OTHER"},
        {"context": None},
        {"system": "OTHER"},
        {"replicate": "r2"},
        {"max_tokens": 128000},
        {"allow_fallback": True},
        {"prompt_sha": "p2"},
        {
            "schema": strict(
                {"type": "object", "properties": {"n": {"type": "integer", "description": "count"}}}
            )
        },
    ],
)
def test_every_request_field_changes_the_key(tmp_path: Path, change):
    other = dataclasses.replace(BASE, **change)
    assert other.key() != BASE.key()
    inner = CountingClient()
    cache = CachedClient(inner, tmp_path)
    cache.complete(BASE)
    assert not cache.is_cached(other)
    cache.complete(other)
    assert len(inner.requests) == 2


def test_offline_miss_raises_and_never_calls_inner(tmp_path: Path):
    inner = CountingClient()
    with pytest.raises(CacheMiss):
        CachedClient(inner, tmp_path, offline=True).complete(BASE)
    assert inner.requests == []
    CachedClient(inner, tmp_path).complete(BASE)
    replay = CachedClient(CountingClient(), tmp_path, offline=True).complete(BASE)
    assert replay.from_cache and replay.data == {"n": 1}


@pytest.mark.parametrize(
    "build",
    [
        lambda r: refusal_response(r, "cyber", "synthetic"),
        lambda r: refusal_response(r),
        truncated_response,
        invalid_response,
        lambda r: substituted_response(r, {"n": 2}),
    ],
)
def test_non_ok_outcomes_are_cached_too(tmp_path: Path, build):
    inner = CountingClient(build)
    cache = CachedClient(inner, tmp_path)
    first = cache.complete(BASE)
    replay = cache.complete(BASE)
    assert len(inner.requests) == 1
    assert dataclasses.replace(replay, from_cache=False) == first


def test_exceptions_are_not_cached(tmp_path: Path):
    calls = []

    def script(req):
        calls.append(req)
        if len(calls) == 1:
            raise LLMTransportError("synthetic outage")
        return {"n": 3}

    cache = CachedClient(FakeClient(script), tmp_path)
    with pytest.raises(LLMTransportError):
        cache.complete(BASE)
    assert not cache.is_cached(BASE)
    assert cache.complete(BASE).data == {"n": 3} and len(calls) == 2


def test_entry_holds_no_prompt_text_and_a_small_request_header(tmp_path: Path):
    cache = CachedClient(CountingClient(), tmp_path)
    cache.complete(BASE)
    text = cache.path_for(BASE).read_text("utf-8")
    for prompt_part in ("SYSTEM-TEXT", "CONTEXT-TEXT", "BODY-TEXT"):
        assert prompt_part not in text
    entry = json.loads(text)
    assert entry["cache_format"] == 1
    assert entry["request"] == {
        "task": "collate",
        "model": "claude-opus-5-5",
        "effort": "high",
        "max_tokens": 64000,
        "replicate": "r1",
        "allow_fallback": False,
        "prompt_sha": "p1",
    }
    assert entry["response"]["from_cache"] is False


def test_dict_round_trip():
    response = dataclasses.replace(
        substituted_response(BASE, {"n": 5}, usage=Usage(1, 2, 3, 4)), request_id="req_1", from_cache=True
    )
    data = response_to_dict(response)
    assert json.loads(json.dumps(data)) == data
    assert response_from_dict(data) == response
    with pytest.raises(TypeError):
        response_from_dict({**data, "unknown_field": 1})


def test_corrupt_or_mismatched_entries_fail_loudly(tmp_path: Path):
    cache = CachedClient(CountingClient(), tmp_path)
    cache.complete(BASE)
    path = cache.path_for(BASE)
    entry = json.loads(path.read_text("utf-8"))
    entry["response"]["key"] = "0" * 64
    path.write_text(json.dumps(entry), "utf-8")
    with pytest.raises(LLMError, match="holds key"):
        cache.complete(BASE)
    path.write_text('{"cache_format": 1, "response": {', "utf-8")
    with pytest.raises(LLMError, match="unreadable"):
        cache.complete(BASE)
    path.write_text(json.dumps({**entry, "cache_format": 99}), "utf-8")
    with pytest.raises(LLMError, match="cache_format"):
        cache.complete(BASE)


def test_inner_answer_for_another_key_is_not_written(tmp_path: Path):
    class WrongKey:
        def complete(self, request):
            return ok_response(dataclasses.replace(request, replicate="elsewhere"), {"n": 1})

    cache = CachedClient(WrongKey(), tmp_path)
    with pytest.raises(LLMError, match="inner client answered"):
        cache.complete(BASE)
    assert not cache.is_cached(BASE)


def test_model_ids_must_be_safe_directory_names(tmp_path: Path):
    with pytest.raises(ValueError):
        CachedClient(CountingClient(), tmp_path).path_for(dataclasses.replace(BASE, model="../escape"))
