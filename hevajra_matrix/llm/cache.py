"""Content-addressed response cache: one JSON file per request key.

    <root>/<model>/<key[:2]>/<key>.json

Every response the inner client returns is cached - ok, refusal, truncated and invalid
alike - because each one is a research observation that a replay must reproduce.
Exceptions are never cached. A hit returns the stored response with
``from_cache=True``; in offline mode a miss raises ``CacheMiss`` instead of calling
the API, so any finished run can be replayed without a key.

The key (``LLMRequest.key``) covers every request field, including the replicate tag,
so a new replicate, prompt version, schema or effort is a new entry, never an overwrite.

Entries hold ``raw_text`` (model output that may quote licence-restricted source
text): keep the cache under ``runs/`` (gitignored) and deposit it privately.

Writes are atomic (temporary file, fsync, ``os.replace``), so an interrupted run never
leaves a truncated entry.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .client import CacheMiss, LLMClient, LLMError, LLMRequest, LLMResponse, Usage

CACHE_FORMAT = 1
_SAFE_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class CachedClient:
    """``LLMClient`` decorator that stores every response under its request key."""

    def __init__(self, inner: LLMClient, root: Path, offline: bool = False) -> None:
        self.inner = inner
        self.root = Path(root)
        self.offline = offline

    def path_for(self, request: LLMRequest) -> Path:
        if not _SAFE_MODEL.fullmatch(request.model):
            raise ValueError(f"model id {request.model!r} is not usable as a directory name")
        key = request.key()
        return self.root / request.model / key[:2] / f"{key}.json"

    def is_cached(self, request: LLMRequest) -> bool:
        return self.path_for(request).is_file()

    def complete(self, request: LLMRequest) -> LLMResponse:
        path, key = self.path_for(request), request.key()
        if path.is_file():
            return dataclasses.replace(_read_entry(path, key), from_cache=True)
        if self.offline:
            raise CacheMiss(f"offline: task {request.task!r} key {key[:12]} is not cached under {self.root}")
        response = self.inner.complete(request)
        if response.key != key:
            raise LLMError(f"inner client answered key {response.key[:12]} for request key {key[:12]}")
        _write_atomic(path, _entry_text(request, response))
        return response


def response_to_dict(response: LLMResponse) -> dict[str, Any]:
    """Plain JSON-ready dict of every field (``usage`` as a nested dict)."""
    return dataclasses.asdict(response)


def response_from_dict(data: Mapping[str, Any]) -> LLMResponse:
    """Inverse of ``response_to_dict``; unknown fields raise ``TypeError``."""
    fields = dict(data)
    fields["usage"] = Usage(**fields.get("usage", {}))
    return LLMResponse(**fields)


def _entry_text(request: LLMRequest, response: LLMResponse) -> str:
    entry = {
        "cache_format": CACHE_FORMAT,
        # Small, text-free description of the request, for people browsing the cache.
        "request": {
            "task": request.task,
            "model": request.model,
            "effort": request.effort,
            "max_tokens": request.max_tokens,
            "replicate": request.replicate,
            "allow_fallback": request.allow_fallback,
            "prompt_sha": request.prompt_sha,
        },
        "response": {**response_to_dict(response), "from_cache": False},
    }
    return json.dumps(entry, ensure_ascii=False, sort_keys=True, indent=1) + "\n"


def _read_entry(path: Path, key: str) -> LLMResponse:
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("cache_format") != CACHE_FORMAT:
            raise ValueError(f"cache_format {entry.get('cache_format')!r}, expected {CACHE_FORMAT}")
        response = response_from_dict(entry["response"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LLMError(f"unreadable cache entry {path}: {exc}") from exc
    if response.key != key:
        raise LLMError(f"cache entry {path} holds key {response.key[:12]}, expected {key[:12]}")
    return response


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
