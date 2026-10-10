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
    --system-prompt-file   replaces Claude Code's default system prompt with the task
                           instructions followed by the shared context (the full witness
                           text). The CLI still puts its own short identity line ("You are a
                           Claude agent, built on Anthropic's Claude Agent SDK.") and an
                           attribution header in front, and adds its own reminder turns for
                           the structured answer; these belong to the instrument and depend
                           on the CLI version, which is therefore recorded with every
                           response (``client_info``) and can be pinned
                           (``subscription.cli_version``). The context sits in the system
                           prompt because Claude Code caches the system prompt automatically.
                           The body is sent on stdin.
    --tools ""             no built-in tools (``subscription.tools`` in ``config/llm.yaml``);
                           --strict-mcp-config: no MCP servers
    --setting-sources ""   no user, project or local settings, hooks or CLAUDE.md; the child
                           also runs in an empty temporary directory
    --json-schema          structured output. Claude Code validates the answer and re-prompts
                           on a mismatch; the answer is validated again here.
    --model, --effort      as for the API backend; no --fallback-model (no fallback)
    --max-turns            a small cap, because the structured answer may take a turn
    CLAUDE_CODE_MAX_OUTPUT_TOKENS   ``request.max_tokens``. When a message hits it, the CLI
                           asks the model to continue (a bounded number of times) instead of
                           stopping, so a truncation the API backend would report can end
                           as an answer here; ``client_info`` records the number of turns and
                           whether the output exceeded ``max_tokens``.

Credentials. The child gets an allowlisted environment (``PASSED_ENV``: path, locale, proxy
and CA settings, and the subscription login), never the parent's other variables, so an
``ANTHROPIC_API_KEY`` in the parent cannot bill the call. The CLI then authenticates with,
in this order of what is present: ``CLAUDE_CODE_OAUTH_TOKEN`` (from ``claude setup-token``),
the login stored under ``HOME`` (``claude`` then ``/login``), or, inside a Claude Code cloud
session, that session's own OAuth token, which the CLI reads from a fixed file
(``SESSION_DIR``) whatever the environment says. All three bill the plan of the account they
belong to. If the cloud session also holds an API key file there, the client refuses to run
unless ``CLAUDE_CODE_OAUTH_TOKEN`` is set, because the CLI could bill that key instead. The
credential source (a label, never the secret) is recorded by ``claude-check`` and in the run
manifest. With ``CLAUDE_CODE_OAUTH_TOKEN`` the child also gets a fresh ``CLAUDE_CONFIG_DIR``
in its scratch directory, so no configuration of the host account is read.

Differences from the API backend, all recorded rather than hidden:
    * The instrument is different (Claude Code wraps the request and enforces the schema by
      validate-and-re-prompt instead of constrained decoding), so the backend is part of the
      instrument digest and the cache is kept apart (``<cache>/subscription/``).
    * Served model: read from the result's ``modelUsage``. Any output from a model other
      than the requested one marks the response ``fallback_used``, so it is never measured.
    * Refusals: ``stop_reason == "refusal"`` or a non-zero ``safety_stops`` count.
    * Run failures are raised and never cached; a usage limit and a timeout are not retried
      (rerunning the command resumes from the cache).
    * Costs: ``usage`` is real; any dollar figure is the API-equivalent, not a charge.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
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
    "DISABLE_AUTOUPDATER": "1",                 # the CLI version must not change mid-run
}
SESSION_DIR = Path("/home/claude/.claude/remote")   # where the CLI finds a cloud session's credentials
SYSTEM_FILE = "system.txt"
# Result subtypes that are research data (the model did not produce a valid answer). The
# CLI sets is_error on them too; they are still answers, cached as "invalid".
INVALID_SUBTYPES = frozenset({"error_max_turns", "error_max_structured_output_retries"})
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
TRUNCATION_API_ERRORS = frozenset({"max_output_tokens", "model_context_window_exceeded"})
TRUNCATION_TEXT = ("output token maximum",)
USAGE_LIMIT_API_ERROR = "usage_limit_reached"
# The plan's own limit messages ("You've hit your session limit", "... weekly limit", "... Opus
# limit", "out of usage credits"); a transient 429 says "not your usage limit" and is retried.
USAGE_LIMIT_TEXT = re.compile(r"hit your [\w ]*limit|out of usage credits|usage limit reached", re.IGNORECASE)
NOT_USAGE_LIMIT_TEXT = "not your usage limit"

# runner(argv, stdin_text, env, cwd, timeout_s) -> (exit code, stdout, stderr)
Runner = Callable[[list[str], str, Mapping[str, str], Path, float], tuple[int, str, str]]


class UsageLimitReached(LLMTransportError):
    """The subscription's usage allowance is exhausted for now. Finished calls are cached:
    rerun the same command after the limit resets and it continues where it stopped."""


class CliTimeout(LLMTransportError):
    """One ``claude -p`` run exceeded ``subscription.timeout_s`` and was killed. Not retried:
    the same request would most likely run as long again."""


@dataclass(frozen=True)
class CliSettings:
    """``config/llm.yaml: subscription``.

    ``timeout_s`` bounds a whole ``claude -p`` run (all turns, wall clock), unlike the API
    backend's ``llm.timeout_s``, which bounds a pause in a stream; so it is much larger.
    ``cli_version``: when set, the client refuses a CLI whose ``--version`` does not contain it.
    """

    command: str = "claude"
    max_turns: int = 3
    timeout_s: float = 3600.0
    max_retries: int = 1
    tools: str = ""          # value of --tools: "" disables every built-in tool
    cli_version: str | None = None

    @classmethod
    def from_config(cls, llm: Mapping[str, Any]) -> "CliSettings":
        sub = dict(llm.get("subscription") or {})
        pin = sub.get("cli_version")
        return cls(command=str(sub.get("command", cls.command)), max_turns=int(sub.get("max_turns", cls.max_turns)),
                   timeout_s=float(sub.get("timeout_s", cls.timeout_s)),
                   max_retries=int(sub.get("max_retries", cls.max_retries)),
                   tools=str(sub.get("tools") or ""), cli_version=str(pin) if pin else None)


def credential_source(environ: Mapping[str, str], session_dir: Path = SESSION_DIR) -> str:
    """Which login the child CLI will use (a label, never the secret)."""
    if environ.get(OAUTH_TOKEN):
        return "CLAUDE_CODE_OAUTH_TOKEN"
    if (session_dir / ".api_key").exists():
        return "cloud session API key file (refused)"
    if (session_dir / ".oauth_token").exists():
        return "cloud session OAuth token"
    return "login stored under HOME"


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
def to_response(result: Mapping[str, Any], request: LLMRequest, key: str, now: str,
                cli_version: str | None = None) -> LLMResponse:
    """Normalise the JSON result of one ``claude -p --output-format json`` run.

    Classified in this order (the CLI sets ``is_error`` on several of these answers too):
        refusal    ``stop_reason == "refusal"`` or ``safety_stops > 0``
        truncated  ``stop_reason`` is ``max_tokens`` / ``model_context_window_exceeded``, or
                   ``api_error`` names the output or context limit
        invalid    subtype ``error_max_turns`` / ``error_max_structured_output_retries``
        (raised)   anything else with ``is_error`` or a non-success subtype: a run failure,
                   ``LLMTransportError`` (``UsageLimitReached`` for an exhausted allowance),
                   never cached
        ok         subtype ``success`` with a ``structured_output`` that passes the schema
        invalid    any other success
    """
    subtype = str(result.get("subtype") or "")
    stop_reason = str(result.get("stop_reason") or "")
    api_error = str(result.get("api_error") or "")
    text = str(result.get("result") or "")
    refused = stop_reason == "refusal" or int(result.get("safety_stops") or 0) > 0
    truncated = (stop_reason in TRUNCATION_STOP_REASONS or api_error in TRUNCATION_API_ERRORS
                 or any(marker in text.lower() for marker in TRUNCATION_TEXT))
    if not (refused or truncated or subtype in INVALID_SUBTYPES) and (result.get("is_error") or subtype != "success"):
        raise _failure(result)
    served, fallback_used = served_model(result.get("modelUsage") or {}, request.model)
    output = result.get("structured_output")
    usage = _usage(result.get("usage") or {})
    info = {k: v for k, v in {"cli_version": cli_version, "subtype": subtype, "num_turns": result.get("num_turns"),
                              "terminal_reason": result.get("terminal_reason"), "api_error": api_error or None,
                              "output_over_max_tokens": usage.output_tokens > request.max_tokens or None}.items()
            if v is not None}
    common: dict[str, Any] = {
        "key": key, "raw_text": canonical_json(output) if output is not None else text,
        "requested_model": request.model, "served_model": served, "stop_reason": stop_reason or subtype,
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
    if result.get("api_error") == USAGE_LIMIT_API_ERROR:
        return True
    return NOT_USAGE_LIMIT_TEXT not in detail.lower() and USAGE_LIMIT_TEXT.search(detail) is not None


def _failure(result: Mapping[str, Any]) -> LLMTransportError:
    detail = " ".join(str(x) for x in [result.get("result") or "", *(result.get("errors") or [])]).strip()
    message = f"claude -p failed ({result.get('subtype')}): {detail[:300] or 'no detail'}"
    if is_usage_limit(result, detail):
        return UsageLimitReached(message + "; finished calls are cached, rerun after the limit resets")
    return LLMTransportError(message)


# --------------------------------------------------------------------------- client
def _run(argv: list[str], stdin_text: str, env: Mapping[str, str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
    """Run ``argv`` in its own process group; on timeout kill the whole group (the CLI and
    anything it started) and raise ``CliTimeout``."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", env=dict(env), cwd=cwd, start_new_session=True)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"{argv[0]!r} not found: install Claude Code and log in with the subscription") from exc
    try:
        stdout, stderr = proc.communicate(stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        proc.communicate()
        raise CliTimeout(f"claude -p ran longer than {timeout_s:.0f} s and was stopped (subscription.timeout_s); "
                         "rerun to continue, finished calls are cached") from None
    except BaseException:
        _kill_group(proc)
        raise
    _kill_group(proc)   # any helper the CLI left behind
    return proc.returncode, stdout, stderr


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class ClaudeCodeClient:
    """``LLMClient`` that runs one ``claude -p`` process per request.

    ``runner``, ``sleep``, ``environ`` and ``session_dir`` are injectable for tests.
    Transport failures are retried ``max_retries`` times with a growing pause;
    ``UsageLimitReached`` and ``CliTimeout`` are not. The CLI version is read once per
    client (``version``), checked against ``cli_version`` when that is set, and recorded in
    every response's ``client_info``.
    """

    def __init__(self, settings: CliSettings | None = None, runner: Runner | None = None,
                 sleep: Callable[[float], None] = time.sleep, environ: Mapping[str, str] | None = None,
                 session_dir: Path = SESSION_DIR) -> None:
        self.settings = settings or CliSettings()
        self.runner = runner or _run
        self.sleep = sleep
        self.environ = environ
        self.session_dir = session_dir
        self._version: str | None = None
        self._lock = threading.Lock()

    @property
    def env(self) -> Mapping[str, str]:
        return os.environ if self.environ is None else self.environ

    def credential_source(self) -> str:
        return credential_source(self.env, self.session_dir)

    def version(self) -> str:
        """``claude --version`` (read once), or the error text."""
        with self._lock:
            if self._version is None:
                try:
                    code, stdout, stderr = self.runner([self.settings.command, "--version"], "", {
                        k: v for k, v in self.env.items() if k in PASSED_ENV}, Path.cwd(), 60.0)
                    text = (stdout or stderr).strip()[:200]
                    self._version = text if code == 0 else f"exit {code}: {text}"
                except (ConfigurationError, LLMTransportError) as exc:
                    self._version = f"unavailable: {exc}"
            return self._version

    def _preflight(self) -> str:
        """Refuse what must not run; return the CLI version."""
        if not self.env.get(OAUTH_TOKEN) and (self.session_dir / ".api_key").exists():
            raise ConfigurationError(f"{self.session_dir}/.api_key exists, so claude -p could bill an API key; "
                                     f"set {OAUTH_TOKEN} (from `claude setup-token`) to use the subscription")
        version = self.version()
        pin = self.settings.cli_version
        if pin and pin not in version:
            raise ConfigurationError(f"Claude Code version {version!r} is not the pinned subscription.cli_version "
                                     f"{pin!r}; a different CLI is a different instrument")
        return version

    def complete(self, request: LLMRequest) -> LLMResponse:
        version = self._preflight()
        attempt = 0
        while True:
            try:
                return self._once(request, version)
            except (UsageLimitReached, CliTimeout):
                raise
            except LLMTransportError:
                if attempt >= self.settings.max_retries:
                    raise
                attempt += 1
                self.sleep(30.0 * attempt)

    def _once(self, request: LLMRequest, version: str | None = None) -> LLMResponse:
        key = request.key()
        with tempfile.TemporaryDirectory(prefix="hevajra-claude-") as tmp:
            workdir = Path(tmp)
            system_file = workdir / SYSTEM_FILE
            system_file.write_text(system_text(request), encoding="utf-8")
            env = child_env(request, self.env, workdir / "config")
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
        return to_response(result, request, key, now, cli_version=version)
