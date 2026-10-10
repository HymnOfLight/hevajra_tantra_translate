"""llm.claude_code_client: the subscription backend (``claude -p``) without the real CLI.

Pure parts (command line, environment, result mapping) are tested directly; the client
is tested with an injected runner and, end to end through the CLI, with a fake ``claude``
executable on PATH.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.claude_code_client import (
    CliSettings,
    ClaudeCodeClient,
    UsageLimitReached,
    build_argv,
    child_env,
    served_model,
    system_text,
    to_response,
)
from hevajra_matrix.llm.client import ConfigurationError, LLMRequest, LLMTransportError
from hevajra_matrix.llm.schema import strict
from hevajra_matrix.pipeline import make_client
from hevajra_matrix.prereg import instrument_digests

from test_pipeline_support import context, make_root

MODEL = "claude-opus-5-5"
SCHEMA = strict({"type": "object", "properties": {"ok": {"type": "boolean"}}})


def request(context_text: str | None = "CONTEXT", max_tokens: int = 2048) -> LLMRequest:
    return LLMRequest(task="check", prompt_sha="p", system="SYSTEM", body="BODY", schema=SCHEMA, effort="high",
                      max_tokens=max_tokens, context=context_text)


def result(**overrides) -> dict:
    base = {"type": "result", "subtype": "success", "is_error": False, "stop_reason": "end_turn",
            "session_id": "sess-1", "structured_output": {"ok": True}, "result": "",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
                      "cache_creation_input_tokens": 7},
            "modelUsage": {MODEL: {"inputTokens": 10, "outputTokens": 5}}}
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- request
def test_command_line_pins_a_single_tool_free_structured_call(tmp_path: Path) -> None:
    argv = build_argv(request(), CliSettings(command="claude", max_turns=3), tmp_path / "system.txt")
    assert argv[:2] == ["claude", "-p"]
    flags = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1) if argv[i].startswith("--")
             and not argv[i + 1].startswith("--")}
    assert flags["--output-format"] == "json" and flags["--model"] == MODEL and flags["--effort"] == "high"
    assert flags["--system-prompt-file"] == str(tmp_path / "system.txt") and flags["--max-turns"] == "3"
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv and "--no-session-persistence" in argv
    assert "--fallback-model" not in argv, "no fallback on the subscription backend"
    assert json.loads(flags["--json-schema"])["properties"] == {"ok": {"type": "boolean"}}
    assert "BODY" not in argv and "CONTEXT" not in argv, "texts go through a file and stdin, never argv"


def test_system_file_holds_instructions_then_context() -> None:
    assert system_text(request()) == "SYSTEM\n\nCONTEXT"
    assert system_text(request(context_text=None)) == "SYSTEM"


def test_child_environment_is_an_allowlist(tmp_path: Path) -> None:
    host = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t", "PATH": "/bin", "HOME": "/root",
            "HTTPS_PROXY": "http://proxy", "ANTHROPIC_BASE_URL": "https://session-endpoint", "CLAUDECODE": "1",
            "CLAUDE_CODE_REMOTE": "true", "CLAUDE_SESSION_INGRESS_TOKEN_FILE": "/x", "GH_TOKEN": "g",
            "CLAUDE_EFFORT": "max"}
    env = child_env(request(max_tokens=4096), host, tmp_path / "config")
    assert env["PATH"] == "/bin" and env["HOME"] == "/root" and env["HTTPS_PROXY"] == "http://proxy"
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDECODE",
                 "CLAUDE_CODE_REMOTE", "CLAUDE_SESSION_INGRESS_TOKEN_FILE", "GH_TOKEN", "CLAUDE_EFFORT"):
        assert name not in env, name
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096" and env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert "CLAUDE_CONFIG_DIR" not in env, "a stored login under HOME keeps the host configuration"


def test_an_oauth_token_gets_a_fresh_config_directory(tmp_path: Path) -> None:
    env = child_env(request(), {"CLAUDE_CODE_OAUTH_TOKEN": "tok", "HOME": "/root"}, tmp_path / "config")
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok" and env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "config")
    own = child_env(request(), {"CLAUDE_CODE_OAUTH_TOKEN": "tok", "CLAUDE_CONFIG_DIR": "/mine"}, tmp_path / "c")
    assert own["CLAUDE_CONFIG_DIR"] == "/mine"


# -------------------------------------------------------------------------- response
def test_success_with_a_valid_structured_answer_is_ok() -> None:
    r = to_response(result(), request(), "k", "now")
    assert r.status == "ok" and r.data == {"ok": True} and r.usable
    assert r.served_model == MODEL and not r.fallback_used and r.request_id == "sess-1"
    assert r.usage.cache_read_input_tokens == 100 and r.usage.output_tokens == 5


@pytest.mark.parametrize("overrides", [
    {"structured_output": {"ok": "yes"}},
    {"structured_output": None, "result": "plain text"},
    {"subtype": "error_max_structured_output_retries", "structured_output": None},
    {"subtype": "error_max_turns", "structured_output": None},
])
def test_answers_without_a_valid_structured_output_are_invalid(overrides) -> None:
    r = to_response(result(**overrides), request(), "k", "now")
    assert r.status == "invalid" and r.data is None and not r.usable


@pytest.mark.parametrize("overrides", [{"stop_reason": "refusal"}, {"safety_stops": 1}])
def test_refusals_are_values(overrides) -> None:
    r = to_response(result(**overrides), request(), "k", "now")
    assert r.status == "refusal" and r.data is None


def test_truncation() -> None:
    assert to_response(result(stop_reason="max_tokens"), request(), "k", "now").status == "truncated"


def test_output_from_another_model_is_a_substitution() -> None:
    usage = {MODEL: {"outputTokens": 3}, "claude-opus-4-8": {"outputTokens": 40}}
    r = to_response(result(modelUsage=usage), request(), "k", "now")
    assert r.served_model == "claude-opus-4-8" and r.fallback_used and not r.usable


def test_served_model_reading() -> None:
    assert served_model({f"{MODEL}[1m]": {"outputTokens": 9}}, MODEL) == (MODEL, False)
    assert served_model({}, MODEL) == ("unknown", True)
    assert served_model({MODEL: {"outputTokens": 9}, "claude-haiku-4-5": {"outputTokens": 0}}, MODEL) == (MODEL, False)


@pytest.mark.parametrize("overrides, error", [
    ({"is_error": True, "result": "API Error: 500"}, LLMTransportError),
    ({"subtype": "error_during_execution", "errors": ["boom"]}, LLMTransportError),
    ({"is_error": True, "result": "Claude AI usage limit reached|1760000000"}, UsageLimitReached),
])
def test_run_failures_are_raised_not_cached(overrides, error) -> None:
    with pytest.raises(error):
        to_response(result(**overrides), request(), "k", "now")


# --------------------------------------------------------------------------- client
def test_client_runs_one_process_and_retries_transport_failures(tmp_path: Path) -> None:
    seen, outputs, pauses = [], [(1, "", "connection reset"), (0, json.dumps(result()), "")], []

    def runner(argv, stdin_text, env, cwd, timeout_s):
        system_file = Path(argv[argv.index("--system-prompt-file") + 1])
        seen.append((stdin_text, system_file.read_text(encoding="utf-8"), cwd, env))
        return outputs.pop(0)

    client = ClaudeCodeClient(CliSettings(max_retries=2), runner=runner, sleep=pauses.append,
                              environ={"ANTHROPIC_API_KEY": "k"})
    r = client.complete(request())
    assert r.status == "ok" and r.key == request().key() and len(seen) == 2 and pauses == [30.0]
    stdin_text, system, cwd, env = seen[0]
    assert stdin_text == "BODY" and system == "SYSTEM\n\nCONTEXT" and "ANTHROPIC_API_KEY" not in env
    assert not Path(cwd).exists(), "the scratch directory is removed after the call"


def test_usage_limit_is_not_retried() -> None:
    calls = []

    def runner(*args):
        calls.append(1)
        return 0, json.dumps(result(is_error=True, result="usage limit reached")), ""

    with pytest.raises(UsageLimitReached):
        ClaudeCodeClient(runner=runner, sleep=lambda s: None, environ={}).complete(request())
    assert len(calls) == 1


def test_missing_executable_is_a_configuration_error() -> None:
    client = ClaudeCodeClient(CliSettings(command="/nonexistent/claude-binary"), environ={})
    with pytest.raises(ConfigurationError):
        client.complete(request())


# --------------------------------------------------------------------------- pipeline
FAKE_CLAUDE = '''#!{python}
import json, os, sys
argv = sys.argv[1:]
if argv == ["--version"]:
    print("9.9.9 (Claude Code)")
    sys.exit(0)
log = {log!r}
with open(log, "a") as handle:
    handle.write(json.dumps({{"argv": argv, "stdin": sys.stdin.read(),
                             "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                             "github": os.environ.get("GH_TOKEN")}}) + "\\n")
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "stop_reason": "end_turn",
                  "session_id": "fake", "structured_output": {{"ok": True}},
                  "usage": {{"input_tokens": 3, "output_tokens": 2}},
                  "modelUsage": {{argv[argv.index("--model") + 1]: {{"outputTokens": 2}}}}}}))
'''


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch) -> Path:
    bin_dir, log = tmp_path / "bin", tmp_path / "claude_calls.jsonl"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text(FAKE_CLAUDE.format(python=sys.executable, log=str(log)), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("GH_TOKEN", "must-not-reach-the-cli-either")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-the-cli")
    return log


@pytest.mark.skipif(os.name != "posix", reason="the fake executable is a POSIX script")
def test_claude_check_on_the_subscription_backend(tmp_path: Path, fake_claude: Path, capsys) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["claude-check", "--backend", "subscription", "--root", str(root), "--run-dir", str(run)]) == 0
    summary = json.loads((run / "claude_check.json").read_text(encoding="utf-8"))
    assert summary["passed"] and summary["backend"] == "subscription" and summary["served_model"] == MODEL
    assert summary["claude_code_version"] == "9.9.9 (Claude Code)"
    call = json.loads(fake_claude.read_text(encoding="utf-8").splitlines()[0])
    assert call["api_key"] is None and call["github"] is None and call["argv"][call["argv"].index("--effort") + 1] == "low"
    audit = [json.loads(x) for x in (run / "llm_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(audit) == 1 and audit[0]["served_model"] == MODEL
    assert (root / "runs" / "llm-cache" / "llm_spend.subscription.jsonl").is_file()
    assert not (root / "runs" / "llm-cache" / "llm_spend.jsonl").exists(), "API budget untouched"


def test_subscription_client_composition_and_separate_cache(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    ctx = context(root, tmp_path / "run")
    sub = dataclasses.replace(ctx, settings=dataclasses.replace(
        ctx.settings, llm={**ctx.settings.llm, "backend": "subscription"}))
    client = make_client(sub)
    assert isinstance(client.inner, CachedClient) and isinstance(client.inner.inner, ClaudeCodeClient)
    assert client.inner.root == ctx.settings.path("cache") / "subscription"
    assert client.budget_usd == float("inf")
    api, subscription = instrument_digests(ctx.settings), instrument_digests(sub.settings)
    assert api.keys() == subscription.keys() and all(api[t] != subscription[t] for t in api), \
        "the backend is part of every instrument"


def test_unknown_backend_is_refused(tmp_path: Path, capsys) -> None:
    root = make_root(tmp_path)
    path = root / "config" / "llm.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("backend: api", "backend: local"), encoding="utf-8")
    assert main(["ingest", "--root", str(root), "--run-dir", str(tmp_path / "run")]) == 2
    assert "backend 'local'" in capsys.readouterr().err
