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
                                               (``llm.claude_code_result``)

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
``ANTHROPIC_API_KEY`` in the parent cannot reach it. The CLI itself picks its credential in
this order: an API key file of a Claude Code cloud session (``SESSION_DIR/.api_key``, which
wins over every OAuth token), ``CLAUDE_CODE_OAUTH_TOKEN`` (from ``claude setup-token``), the
cloud session's own OAuth token (``SESSION_DIR/.oauth_token``), then the login stored under
the configuration directory (``claude`` then ``/login``). Policy settings can add an API key
helper on top. So before the first request the client asks the CLI (``claude auth status
--json``, a local call, no model request) and runs only when it reports a first-party
subscription login (``SUBSCRIPTION_AUTH_METHODS``); anything else, and any API key file,
is refused. The reported method is recorded with every response (``client_info``). With
``CLAUDE_CODE_OAUTH_TOKEN`` the child also gets a fresh ``CLAUDE_CONFIG_DIR`` in its scratch
directory, so no configuration of the host account is read.

Model fallback. No ``--fallback-model`` is passed and ``CLAUDE_CODE_NO_MODEL_FALLBACK`` is
set, so the CLI fails (``model_substitution_disabled``, raised, not retried) instead of
answering with another model when the requested one is unavailable. A server-side switch
that still happens is detected from ``modelUsage`` (``fallback_used``).

Interruption. The child runs in the parent's process group, so a signal to the group (Ctrl-C,
a supervisor stopping the job) stops it too. ``abort()`` (called by the CLI's signal
handlers) stops any retry or new request in worker threads.

Differences from the API backend, all recorded rather than hidden:
    * The instrument is different (Claude Code wraps the request and enforces the schema by
      validate-and-re-prompt instead of constrained decoding), so the backend is part of the
      instrument digest and the cache is kept apart (``<cache>/subscription/``).
    * Served model: read from the result's ``modelUsage``. Any output from a model other
      than the requested one marks the response ``fallback_used``, so it is never measured.
    * Refusals: ``stop_reason == "refusal"`` or a non-zero ``safety_stops`` count.
    * Run failures are raised and never cached; a usage limit, an unavailable model and a
      timeout are not retried (rerunning the command resumes from the cache).
    * Costs: ``usage`` is real; any dollar figure is the API-equivalent, not a charge.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from . import schema as schema_mod
from .claude_code_result import ModelUnavailable, UsageLimitReached, failure as _failure, to_response
from .client import ConfigurationError, LLMRequest, LLMResponse, LLMTransportError, canonical_json

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
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",       # fail rather than answer with another model
}
SESSION_DIR = Path("/home/claude/.claude/remote")   # where the CLI finds a cloud session's credentials
# What ``claude auth status --json`` reports for a subscription login (anything else is refused).
SUBSCRIPTION_AUTH_METHODS = frozenset({"oauth_token", "claude.ai"})
SYSTEM_FILE = "system.txt"
ABORTED = threading.Event()   # set by abort(): no retry, no new claude -p run
REAP_TIMEOUT_S = 10.0         # how long a killed child (and its pipes) may take to close


def abort() -> None:
    """Stop every client in this process from retrying or starting another ``claude -p``.
    The CLI's signal handlers call it; running children get the signal with the group."""
    ABORTED.set()


# runner(argv, stdin_text, env, cwd, timeout_s) -> (exit code, stdout, stderr)
Runner = Callable[[list[str], str, Mapping[str, str], Path, float], tuple[int, str, str]]


class CliTimeout(LLMTransportError):
    """One ``claude -p`` run exceeded its time limit and was killed. Not retried: the same
    request would most likely run as long again."""


@dataclass(frozen=True)
class CliSettings:
    """``config/llm.yaml: subscription``.

    ``timeout_s`` bounds a whole ``claude -p`` run (all turns, wall clock), unlike the API
    backend's ``llm.timeout_s``, which bounds a pause in a stream; the limit actually used
    is never below ``max_tokens`` at ``MIN_TOKENS_PER_S`` (``run_timeout``).
    ``cli_version``: when set, the client refuses any CLI whose version is not exactly it.
    """

    command: str = "claude"
    max_turns: int = 3
    timeout_s: float = 7200.0
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


MIN_TOKENS_PER_S = 15.0


def run_timeout(request: LLMRequest, settings: CliSettings) -> float:
    """Wall-clock limit for one run: the configured one, or more for a very large output cap."""
    return max(settings.timeout_s, request.max_tokens / MIN_TOKENS_PER_S)


def version_number(text: str) -> str:
    """``"2.1.296 (Claude Code)"`` -> ``"2.1.296"``."""
    head = text.split()[0] if text.split() else ""
    return head[1:] if head.startswith("v") else head


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


# --------------------------------------------------------------------------- client
def _run(argv: list[str], stdin_text: str, env: Mapping[str, str], cwd: Path, timeout_s: float) -> tuple[int, str, str]:
    """Run ``argv`` in the parent's process group (so a signal to the group stops it too);
    on timeout kill it, reap it within a bounded time and raise ``CliTimeout``."""
    if ABORTED.is_set():
        raise LLMTransportError("aborted")
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", env=dict(env), cwd=cwd)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"{argv[0]!r} not found: install Claude Code and log in with the subscription") from exc
    try:
        stdout, stderr = proc.communicate(stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _stop(proc)
        raise CliTimeout(f"claude -p ran longer than {timeout_s:.0f} s and was stopped (subscription.timeout_s); "
                         "rerun to continue, finished calls are cached") from None
    except BaseException:
        _stop(proc)
        raise
    return proc.returncode, stdout, stderr


def _stop(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and reap it; never block for long on pipes a stray helper still holds."""
    proc.kill()
    try:
        proc.communicate(timeout=REAP_TIMEOUT_S)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pass


class ClaudeCodeClient:
    """``LLMClient`` that runs one ``claude -p`` process per request.

    ``runner``, ``sleep``, ``environ`` and ``session_dir`` are injectable for tests.
    Before the first request the client checks once (``preflight``): no cloud API key file,
    a subscription login reported by ``claude auth status --json``, and the pinned CLI
    version if one is set. Transport failures are retried ``max_retries`` times with a
    growing pause; a usage limit, an unavailable model and a timeout are not, and nothing is
    retried or started after ``abort()``.
    """

    def __init__(self, settings: CliSettings | None = None, runner: Runner | None = None,
                 sleep: Callable[[float], Any] = ABORTED.wait, environ: Mapping[str, str] | None = None,
                 session_dir: Path = SESSION_DIR) -> None:
        self.settings = settings or CliSettings()
        self.runner = runner or _run
        self.sleep = sleep
        self.environ = environ
        self.session_dir = session_dir
        self._version: str | None = None
        self._provenance: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def env(self) -> Mapping[str, str]:
        return os.environ if self.environ is None else self.environ

    def _local(self, args: list[str]) -> tuple[int, str]:
        """A local CLI call (no model request) with the child's environment."""
        with tempfile.TemporaryDirectory(prefix="hevajra-claude-") as tmp:
            env = child_env(LLMRequest("", "", "", "", {}, "low", 1), self.env, Path(tmp) / "config")
            code, stdout, stderr = self.runner([self.settings.command, *args], "", env, Path(tmp), 60.0)
        return code, (stdout or stderr).strip()

    def version(self) -> str:
        """``claude --version`` (read once), or the error text."""
        with self._lock:
            if self._version is None:
                try:
                    code, text = self._local(["--version"])
                    self._version = text[:200] if code == 0 else f"exit {code}: {text[:200]}"
                except (ConfigurationError, LLMTransportError) as exc:
                    self._version = f"unavailable: {exc}"
            return self._version

    def credential_source(self) -> str:
        """The login the CLI reports (``authMethod``), or why it is not usable."""
        if (self.session_dir / ".api_key").exists():
            return "cloud session API key file (refused)"
        try:
            code, text = self._local(["auth", "status", "--json"])
            status = json.loads(text) if code == 0 else {}
        except (ConfigurationError, LLMTransportError, json.JSONDecodeError) as exc:
            return f"unknown ({exc})"
        if not isinstance(status, dict) or not status.get("loggedIn"):
            return "not logged in"
        return f"{status.get('authMethod')} ({status.get('apiProvider')})"

    def preflight(self) -> dict[str, Any]:
        """Check once that the run would bill the subscription with the expected CLI; return
        the provenance recorded with every response. Raises ``ConfigurationError``."""
        if ABORTED.is_set():
            raise LLMTransportError("aborted")
        if self._provenance is not None:
            return self._provenance
        source = self.credential_source()
        method, _, provider = source.partition(" (")
        if method not in SUBSCRIPTION_AUTH_METHODS or provider.rstrip(")") != "firstParty":
            raise ConfigurationError(f"claude -p would not use a first-party Claude subscription login (CLI reports: "
                                     f"{source}); log in with `claude` then /login, or set {OAUTH_TOKEN}")
        version = self.version()
        pin = self.settings.cli_version
        if pin and version_number(version) != pin:
            raise ConfigurationError(f"Claude Code version {version!r} is not the pinned subscription.cli_version "
                                     f"{pin!r}; a different CLI is a different instrument")
        self._provenance = {"cli_version": version_number(version) or version, "auth": source}
        return self._provenance

    def complete(self, request: LLMRequest) -> LLMResponse:
        provenance = self.preflight()
        attempt = 0
        while True:
            try:
                return self._once(request, provenance)
            except (UsageLimitReached, ModelUnavailable, CliTimeout):
                raise
            except LLMTransportError:
                if attempt >= self.settings.max_retries or ABORTED.is_set():
                    raise
                attempt += 1
                self.sleep(30.0 * attempt)
                if ABORTED.is_set():
                    raise

    def _once(self, request: LLMRequest, provenance: Mapping[str, Any] | None = None) -> LLMResponse:
        key = request.key()
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hevajra-claude-") as tmp:
            workdir = Path(tmp)
            system_file = workdir / SYSTEM_FILE
            system_file.write_text(system_text(request), encoding="utf-8")
            env = child_env(request, self.env, workdir / "config")
            code, stdout, stderr = self.runner(build_argv(request, self.settings, system_file), request.body,
                                               env, workdir, run_timeout(request, self.settings))
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            detail = (stderr or stdout or "").strip()[:300]
            error = _failure({"subtype": f"exit {code}", "result": detail})
            raise error from None
        if not isinstance(result, dict):
            raise LLMTransportError(f"claude -p printed {type(result).__name__}, expected a JSON object")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        seconds = round(time.monotonic() - started, 1)
        return to_response(result, request, key, now, provenance={**(provenance or {}), "seconds": seconds})
