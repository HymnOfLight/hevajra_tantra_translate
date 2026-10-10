"""Claude Opus 5.5 through Claude Code in print mode (``claude -p``), on a Claude subscription.

This is the ``subscription`` backend (``config/llm.yaml: backend``). It lets a researcher
who has a Claude Max plan but no API credits run the LLM tasks against the plan's usage
allowance. It implements the same ``LLMClient`` contract as ``AnthropicClient``, so the
cache, the audit log and every task above it are unchanged.

Everything except ``ClaudeCodeClient.complete`` is pure and unit-tested without the CLI:

    system_text(request)                       system prompt file content (system + context)
    build_argv(request, settings, path)        the ``claude`` command line
    child_env(request, base, config_dir)       the environment of the child process
    to_response(result, request, key, now)     the CLI's JSON result -> ``LLMResponse``

How the call is pinned down (so that it is a single-turn, tool-free, structured request):
    --system-prompt-file   replaces the whole Claude Code system prompt with the task
                           instructions followed by the shared context (the full witness
                           text). The context sits in the system prompt rather than in the
                           user turn because Claude Code caches the system prompt
                           automatically, which keeps the subscription usage of a full
                           collation pass low. The body is sent on stdin.
    --tools ""             no built-in tools (``subscription.tools`` in ``config/llm.yaml``);
                           --strict-mcp-config: no MCP servers
    --setting-sources ""   no user, project or local settings, hooks or CLAUDE.md; the child
                           also runs in an empty temporary directory
    --json-schema          structured output. Claude Code validates the answer and re-prompts
                           on a mismatch; the answer is validated again here.
    --model, --effort      as for the API backend; no --fallback-model (no fallback)
    --max-turns            a small cap, because the structured answer may take a turn
    CLAUDE_CODE_MAX_OUTPUT_TOKENS   ``request.max_tokens``

The child gets an allowlisted environment (``PASSED_ENV``: path, locale, proxy and CA
settings, and the subscription login), never the parent's other variables. So
``ANTHROPIC_API_KEY`` cannot bill the call, and when this runs inside another Claude Code
session (a cloud sandbox) the child inherits none of that session's endpoint, tokens or
markers. The CLI uses its subscription login: ``CLAUDE_CODE_OAUTH_TOKEN`` (from
``claude setup-token``), or the login stored under ``HOME`` (``claude`` then ``/login``).
With the token, the child also gets a fresh ``CLAUDE_CONFIG_DIR`` in its scratch
directory, so no configuration of the host account is read.

Differences from the API backend, all recorded rather than hidden:
    * The instrument is different (Claude Code wraps the request and enforces the schema by
      validate-and-re-prompt instead of constrained decoding), so the backend is part of the
      instrument digest and the cache is kept apart (``<cache>/subscription/``).
    * Served model: read from the result's ``modelUsage``. Any output from a model other
      than the requested one marks the response ``fallback_used``, so it is never measured.
    * Refusals: ``stop_reason == "refusal"`` or a non-zero ``safety_stops`` count.
    * Costs: ``usage`` is real; any dollar figure is the API-equivalent, not a charge.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from . import schema as schema_mod
from .client import ConfigurationError, LLMRequest, LLMResponse, LLMTransportError, Usage, canonical_json

BACKEND = "subscription"
OAUTH_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
PASSED_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE",
    OAUTH_TOKEN, "CLAUDE_CONFIG_DIR",
)
FIXED_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",     # no memory files read or written
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",     # no claude.ai connectors
}
SYSTEM_FILE = "system.txt"
# Result subtypes that are research data (the model did not produce a valid answer);
# every other non-success subtype is a run failure and is raised, never cached.
INVALID_SUBTYPES = frozenset({"error_max_turns", "error_max_structured_output_retries"})
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
USAGE_LIMIT_MARKERS = ("usage limit", "rate limit", "limit reached")

# runner(argv, stdin_text, env, cwd, timeout_s) -> (exit code, stdout, stderr)
Runner = Callable[[list[str], str, Mapping[str, str], Path, float], tuple[int, str, str]]


class UsageLimitReached(LLMTransportError):
    """The subscription's usage allowance is exhausted for now. Finished calls are cached:
    rerun the same command after the limit resets and it continues where it stopped."""


@dataclass(frozen=True)
class CliSettings:
    """``config/llm.yaml: subscription``."""

    command: str = "claude"
    max_turns: int = 3
    timeout_s: float = 900.0
    max_retries: int = 2
    tools: str = ""          # value of --tools: "" disables every built-in tool

    @classmethod
    def from_config(cls, llm: Mapping[str, Any]) -> "CliSettings":
        sub = dict(llm.get("subscription") or {})
        return cls(command=str(sub.get("command", cls.command)), max_turns=int(sub.get("max_turns", cls.max_turns)),
                   timeout_s=float(llm.get("timeout_s", cls.timeout_s)),
                   max_retries=int(sub.get("max_retries", cls.max_retries)),
                   tools=str(sub.get("tools") or ""))


# --------------------------------------------------------------------------- request
def system_text(request: LLMRequest) -> str:
    """The system prompt file: the task instructions, then the shared context if any."""
    return request.system if not request.context else f"{request.system}\n\n{request.context}"


def build_argv(request: LLMRequest, settings: CliSettings, system_file: Path) -> list[str]:
    """The ``claude`` command line for ``request`` (the body goes to stdin)."""
    return [
        settings.command, "-p",
        "--output-format", "json",
        "--model", request.model,
        "--effort", request.effort,
        "--system-prompt-file", str(system_file),
        "--tools", settings.tools,
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--max-turns", str(settings.max_turns),
        "--json-schema", canonical_json(schema_mod.for_api(request.schema)),
    ]


def child_env(request: LLMRequest, base: Mapping[str, str], config_dir: Path | None = None) -> dict[str, str]:
    """The allowlisted part of ``base``, plus the fixed variables and the output cap.

    With ``CLAUDE_CODE_OAUTH_TOKEN`` and no ``CLAUDE_CONFIG_DIR`` of its own, the child's
    configuration directory is ``config_dir`` (a fresh one per call).
    """
    env = {k: v for k, v in base.items() if k in PASSED_ENV}
    if env.get(OAUTH_TOKEN) and "CLAUDE_CONFIG_DIR" not in env and config_dir is not None:
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env.update(FIXED_ENV)
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(request.max_tokens)
    return env


# -------------------------------------------------------------------------- response
def to_response(result: Mapping[str, Any], request: LLMRequest, key: str, now: str) -> LLMResponse:
    """Normalise the JSON result of one ``claude -p --output-format json`` run.

    Status:
        refusal    ``stop_reason == "refusal"`` or ``safety_stops > 0``
        truncated  ``stop_reason`` is ``max_tokens`` or ``model_context_window_exceeded``
        ok         subtype ``success`` with a ``structured_output`` that passes the schema
        invalid    anything else that is research data (no or invalid structured output,
                   ``error_max_turns``, ``error_max_structured_output_retries``)

    Raises ``LLMTransportError`` (``UsageLimitReached`` for an exhausted allowance) when
    the run itself failed: an error subtype not in ``INVALID_SUBTYPES``, or ``is_error``.
    """
    subtype = str(result.get("subtype") or "")
    stop_reason = str(result.get("stop_reason") or "")
    refused = stop_reason == "refusal" or int(result.get("safety_stops") or 0) > 0
    if not refused and (result.get("is_error") or (subtype != "success" and subtype not in INVALID_SUBTYPES)):
        raise _failure(result)
    served, fallback_used = served_model(result.get("modelUsage") or {}, request.model)
    output = result.get("structured_output")
    raw_text = canonical_json(output) if output is not None else str(result.get("result") or "")
    common: dict[str, Any] = {
        "key": key, "raw_text": raw_text, "requested_model": request.model, "served_model": served,
        "stop_reason": stop_reason or subtype, "usage": _usage(result.get("usage") or {}),
        "fallback_used": fallback_used, "request_id": result.get("session_id"), "created_at": now,
    }
    if refused:
        return LLMResponse(status="refusal", data=None, refusal_category=None, **common)
    if stop_reason in TRUNCATION_STOP_REASONS:
        return LLMResponse(status="truncated", data=None, **common)
    if subtype == "success" and isinstance(output, dict) and not schema_mod.validate(output, request.schema):
        return LLMResponse(status="ok", data=output, **common)
    return LLMResponse(status="invalid", data=None, **common)


def same_model(name: str, requested: str) -> bool:
    """``name`` is ``requested``, possibly with a date or context suffix (``-2026...``, ``[1m]``)."""
    return name == requested or name.startswith(requested + "[") or name.startswith(requested + "-")


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


def _failure(result: Mapping[str, Any]) -> LLMTransportError:
    detail = " ".join(str(x) for x in [result.get("result") or "", *(result.get("errors") or [])]).strip()
    message = f"claude -p failed ({result.get('subtype')}): {detail[:300] or 'no detail'}"
    if any(marker in detail.lower() for marker in USAGE_LIMIT_MARKERS):
        return UsageLimitReached(message + "; finished calls are cached, rerun after the limit resets")
    return LLMTransportError(message)


# --------------------------------------------------------------------------- client
def _run(argv: list[str], stdin_text: str, env: Mapping[str, str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
    try:
        done = subprocess.run(argv, input=stdin_text, capture_output=True, text=True, encoding="utf-8",
                              env=dict(env), cwd=cwd, timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"{argv[0]!r} not found: install Claude Code and log in with the subscription") from exc
    except subprocess.TimeoutExpired as exc:
        raise LLMTransportError(f"claude -p timed out after {timeout_s:.0f} s") from exc
    return done.returncode, done.stdout, done.stderr


class ClaudeCodeClient:
    """``LLMClient`` that runs one ``claude -p`` process per request.

    ``runner``, ``sleep`` and ``environ`` are injectable for tests. Transport failures are
    retried ``max_retries`` times with a growing pause; ``UsageLimitReached`` is not.
    """

    def __init__(self, settings: CliSettings | None = None, runner: Runner | None = None,
                 sleep: Callable[[float], None] = time.sleep, environ: Mapping[str, str] | None = None) -> None:
        self.settings = settings or CliSettings()
        self.runner = runner or _run
        self.sleep = sleep
        self.environ = environ

    def version(self) -> str:
        """``claude --version`` (recorded by ``claude-check``), or the error text."""
        try:
            code, stdout, stderr = self.runner([self.settings.command, "--version"], "", {
                k: v for k, v in (os.environ if self.environ is None else self.environ).items() if k in PASSED_ENV},
                Path.cwd(), 60.0)
        except (ConfigurationError, LLMTransportError) as exc:
            return f"unavailable: {exc}"
        return (stdout or stderr).strip()[:200] if code == 0 else f"exit {code}: {(stderr or stdout).strip()[:200]}"

    def complete(self, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return self._once(request)
            except UsageLimitReached:
                raise
            except LLMTransportError:
                if attempt >= self.settings.max_retries:
                    raise
                attempt += 1
                self.sleep(30.0 * attempt)

    def _once(self, request: LLMRequest) -> LLMResponse:
        key = request.key()
        with tempfile.TemporaryDirectory(prefix="hevajra-claude-") as tmp:
            workdir = Path(tmp)
            system_file = workdir / SYSTEM_FILE
            system_file.write_text(system_text(request), encoding="utf-8")
            env = child_env(request, os.environ if self.environ is None else self.environ, workdir / "config")
            code, stdout, stderr = self.runner(build_argv(request, self.settings, system_file), request.body,
                                               env, workdir, self.settings.timeout_s)
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            detail = (stderr or stdout or "").strip()[:300]
            error = _failure({"subtype": f"exit {code}", "result": detail})
            raise error from None
        if not isinstance(result, dict):
            raise LLMTransportError(f"claude -p printed {type(result).__name__}, expected a JSON object")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return to_response(result, request, key, now)
