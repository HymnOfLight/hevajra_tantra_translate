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
import time
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.cli import main
from hevajra_matrix.config import load_settings
from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.claude_code_client import (
    CliSettings,
    CliTimeout,
    ClaudeCodeClient,
    UsageLimitReached,
    _run,
    build_argv,
    child_env,
    credential_source,
    same_model,
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
    r = to_response(result(num_turns=2, terminal_reason="completed"), request(), "k", "now", cli_version="2.1.296")
    assert r.status == "ok" and r.data == {"ok": True} and r.usable
    assert r.served_model == MODEL and not r.fallback_used and r.request_id == "sess-1"
    assert r.usage.cache_read_input_tokens == 100 and r.usage.output_tokens == 5
    assert r.client_info == {"cli_version": "2.1.296", "subtype": "success", "num_turns": 2,
                             "terminal_reason": "completed"}


def test_live_check_shape_with_stop_reason_tool_use_is_ok() -> None:
    assert to_response(result(stop_reason="tool_use"), request(), "k", "now").status == "ok"


# Shapes as CLI 2.1.296 emits them: is_error is true on these answers too.
@pytest.mark.parametrize("overrides", [
    {"structured_output": {"ok": "yes"}},
    {"structured_output": None, "result": "plain text"},
    {"subtype": "error_max_structured_output_retries", "is_error": True, "structured_output": None,
     "terminal_reason": "structured_output_retry_exhausted"},
    {"subtype": "error_max_turns", "is_error": True, "structured_output": None, "num_turns": 3},
])
def test_answers_without_a_valid_structured_output_are_invalid(overrides) -> None:
    r = to_response(result(**overrides), request(), "k", "now")
    assert r.status == "invalid" and r.data is None and not r.usable


@pytest.mark.parametrize("overrides", [{"stop_reason": "refusal"}, {"safety_stops": 1},
                                       {"stop_reason": "refusal", "is_error": True}])
def test_refusals_are_values(overrides) -> None:
    r = to_response(result(**overrides), request(), "k", "now")
    assert r.status == "refusal" and r.data is None


@pytest.mark.parametrize("overrides", [
    {"stop_reason": "max_tokens"},
    {"is_error": True, "api_error": "max_output_tokens", "stop_reason": "stop_sequence", "structured_output": None},
    {"is_error": True, "structured_output": None, "stop_reason": "stop_sequence",
     "result": "API Error: Claude's response exceeded the 128000 output token maximum."},
])
def test_truncations_are_values_even_with_is_error(overrides) -> None:
    assert to_response(result(**overrides), request(), "k", "now").status == "truncated"


def test_output_beyond_max_tokens_is_flagged() -> None:
    r = to_response(result(usage={"output_tokens": 5000}), request(max_tokens=2048), "k", "now")
    assert r.status == "ok" and r.client_info["output_over_max_tokens"] is True


def test_output_from_another_model_is_a_substitution() -> None:
    usage = {MODEL: {"outputTokens": 3}, "claude-opus-4-8": {"outputTokens": 40}}
    r = to_response(result(modelUsage=usage), request(), "k", "now")
    assert r.served_model == "claude-opus-4-8" and r.fallback_used and not r.usable


def test_served_model_reading() -> None:
    assert served_model({f"{MODEL}[1m]": {"outputTokens": 9}}, MODEL) == (MODEL, False)
    assert served_model({f"{MODEL}-20260701": {"outputTokens": 9}}, MODEL) == (MODEL, False)
    assert served_model({}, MODEL) == ("unknown", True)
    assert served_model({MODEL: {"outputTokens": 9}, "claude-haiku-4-5": {"outputTokens": 0}}, MODEL) == (MODEL, False)


def test_a_longer_model_id_is_not_the_requested_model() -> None:
    assert not same_model("claude-opus-5-5", "claude-opus-5")
    assert served_model({"claude-opus-5-5": {"outputTokens": 9}}, "claude-opus-5") == ("claude-opus-5-5", True)


LIMIT_TEXTS = ["You've hit your session limit \u00b7 resets 7pm", "You've hit your weekly limit \u00b7 resets Mon",
               "You\u2019ve hit your Opus limit", "You're out of usage credits"]


@pytest.mark.parametrize("text", LIMIT_TEXTS)
def test_plan_limit_messages_stop_the_stage(text) -> None:
    with pytest.raises(UsageLimitReached):
        to_response(result(is_error=True, structured_output=None, result=text), request(), "k", "now")


def test_typed_usage_limit_field_stops_the_stage() -> None:
    with pytest.raises(UsageLimitReached):
        to_response(result(is_error=True, structured_output=None, result="x", api_error="usage_limit_reached",
                           api_error_status=429), request(), "k", "now")


@pytest.mark.parametrize("overrides", [
    {"is_error": True, "result": "API Error: 500"},
    {"subtype": "error_during_execution", "errors": ["boom"]},
    {"is_error": True, "result": "API Error: Server is temporarily limiting requests (not your usage limit)",
     "api_error_status": 429},
    {"is_error": True, "result": "API Error: 529 overloaded", "api_error_status": 529},
])
def test_run_failures_are_raised_not_cached(overrides) -> None:
    with pytest.raises(LLMTransportError) as caught:
        to_response(result(structured_output=None, **overrides), request(), "k", "now")
    assert not isinstance(caught.value, UsageLimitReached), "transient failures stay retryable"


# --------------------------------------------------------------------------- client
def scripted(outputs: list, seen: list | None = None):
    """A runner answering --version and then the scripted (code, stdout, stderr) outputs."""
    def runner(argv, stdin_text, env, cwd, timeout_s):
        if argv[1:] == ["--version"]:
            return 0, "2.1.296 (Claude Code)", ""
        if seen is not None:
            system_file = Path(argv[argv.index("--system-prompt-file") + 1])
            seen.append((stdin_text, system_file.read_text(encoding="utf-8"), cwd, env))
        item = outputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
    return runner


def test_client_runs_one_process_and_retries_transport_failures(tmp_path: Path) -> None:
    seen, pauses = [], []
    runner = scripted([(1, "", "connection reset"), (0, json.dumps(result()), "")], seen)
    client = ClaudeCodeClient(CliSettings(max_retries=2), runner=runner, sleep=pauses.append,
                              environ={"ANTHROPIC_API_KEY": "k"}, session_dir=tmp_path)
    r = client.complete(request())
    assert r.status == "ok" and r.key == request().key() and len(seen) == 2 and pauses == [30.0]
    assert r.client_info["cli_version"] == "2.1.296 (Claude Code)"
    stdin_text, system, cwd, env = seen[0]
    assert stdin_text == "BODY" and system == "SYSTEM\n\nCONTEXT" and "ANTHROPIC_API_KEY" not in env
    assert not Path(cwd).exists(), "the scratch directory is removed after the call"


@pytest.mark.parametrize("payload, status", [
    (result(subtype="error_max_turns", is_error=True, structured_output=None), "invalid"),
    (result(subtype="error_max_structured_output_retries", is_error=True, structured_output=None), "invalid"),
    (result(is_error=True, api_error="max_output_tokens", structured_output=None), "truncated"),
])
def test_research_outcomes_take_one_run_and_no_retry(tmp_path: Path, payload, status) -> None:
    outputs, pauses = [(0, json.dumps(payload), "")], []
    client = ClaudeCodeClient(runner=scripted(outputs), sleep=pauses.append, environ={}, session_dir=tmp_path)
    assert client.complete(request()).status == status and not outputs and not pauses


@pytest.mark.parametrize("failure, error", [
    ((0, json.dumps(result(is_error=True, structured_output=None, result=LIMIT_TEXTS[0])), ""), UsageLimitReached),
    (CliTimeout("too long"), CliTimeout),
])
def test_usage_limit_and_timeout_are_not_retried(tmp_path: Path, failure, error) -> None:
    outputs, pauses = [failure, (0, json.dumps(result()), "")], []
    client = ClaudeCodeClient(CliSettings(max_retries=3), runner=scripted(outputs), sleep=pauses.append,
                              environ={}, session_dir=tmp_path)
    with pytest.raises(error):
        client.complete(request())
    assert len(outputs) == 1 and not pauses


def test_an_api_key_file_of_a_cloud_session_is_refused(tmp_path: Path) -> None:
    (tmp_path / ".api_key").write_text("x")
    calls = []
    client = ClaudeCodeClient(runner=lambda *a: calls.append(a) or (0, "", ""), environ={}, session_dir=tmp_path)
    with pytest.raises(ConfigurationError):
        client.complete(request())
    assert not calls
    assert credential_source({}, tmp_path) == "cloud session API key file (refused)"
    ok = ClaudeCodeClient(runner=scripted([(0, json.dumps(result()), "")]), environ={"CLAUDE_CODE_OAUTH_TOKEN": "t"},
                          session_dir=tmp_path)
    assert ok.complete(request()).status == "ok", "an explicit subscription token is used instead"


def test_credential_source_labels(tmp_path: Path) -> None:
    assert credential_source({"CLAUDE_CODE_OAUTH_TOKEN": "t"}, tmp_path) == "CLAUDE_CODE_OAUTH_TOKEN"
    assert credential_source({}, tmp_path) == "login stored under HOME"
    (tmp_path / ".oauth_token").write_text("x")
    assert credential_source({}, tmp_path) == "cloud session OAuth token"


def test_a_pinned_cli_version_is_enforced(tmp_path: Path) -> None:
    client = ClaudeCodeClient(CliSettings(cli_version="9.0.0"), runner=scripted([]), environ={}, session_dir=tmp_path)
    with pytest.raises(ConfigurationError):
        client.complete(request())
    pinned = ClaudeCodeClient(CliSettings(cli_version="2.1.296"), runner=scripted([(0, json.dumps(result()), "")]),
                              environ={}, session_dir=tmp_path)
    assert pinned.complete(request()).status == "ok"


def test_missing_executable_is_a_configuration_error(tmp_path: Path) -> None:
    client = ClaudeCodeClient(CliSettings(command="/nonexistent/claude-binary"), environ={}, session_dir=tmp_path)
    with pytest.raises(ConfigurationError):
        client.complete(request())


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "grandchild.pid"
    script = f"(sleep 30 & echo $! > {marker}; wait) & sleep 30"
    with pytest.raises(CliTimeout):
        _run(["/bin/sh", "-c", script], "", {"PATH": os.environ["PATH"]}, tmp_path, 1.0)
    pid = int(marker.read_text().strip())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the grandchild survived the timeout")


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


# --------------------------------------------------------------------------- end to end
FAKE_COLLATOR = '''#!{python}
import json, sys, time
sys.path.insert(0, {support!r})
from types import SimpleNamespace
from test_pipeline_support import echo_script
argv = sys.argv[1:]
if argv == ["--version"]:
    print("2.1.296 (Claude Code)")
    sys.exit(0)
started = time.time()
system = open(argv[argv.index("--system-prompt-file") + 1], encoding="utf-8").read()
context = system[system.index("WITNESS TEXT"):]
answer = echo_script(SimpleNamespace(context=context, body=sys.stdin.read()))
time.sleep(0.2)
with open({log!r}, "a") as handle:
    handle.write(json.dumps({{"start": started, "end": time.time()}}) + "\\n")
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "stop_reason": "tool_use",
                  "session_id": "fake", "num_turns": 2, "structured_output": answer,
                  "usage": {{"input_tokens": 50, "output_tokens": 40, "cache_read_input_tokens": 900}},
                  "modelUsage": {{argv[argv.index("--model") + 1]: {{"outputTokens": 40}}}}}}))
'''


@pytest.fixture
def fake_collator(tmp_path: Path, monkeypatch) -> Path:
    bin_dir, log = tmp_path / "bin", tmp_path / "calls.jsonl"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text(FAKE_COLLATOR.format(python=sys.executable, support=str(Path(__file__).parent), log=str(log)),
                   encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return log


@pytest.mark.skipif(os.name != "posix", reason="the fake executable is a POSIX script")
def test_collate_on_the_subscription_backend_end_to_end(tmp_path: Path, fake_collator: Path, capsys) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    cli = lambda *a: main([*a, "--root", str(root), "--run-dir", str(run)])  # noqa: E731
    cli("ingest")
    assert cli("collate", "--backend", "subscription", "--replicates", "2") == 0
    assert (run / "llm_backend.txt").read_text().strip() == "subscription"
    cache = root / "runs" / "llm-cache"
    assert any((cache / "subscription").rglob("*.json")) and not (cache / MODEL).exists()
    audit = [json.loads(x) for x in (run / "llm_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert audit and all(a["status"] == "ok" and a["client_info"]["cli_version"].startswith("2.1.296") for a in audit)
    calls = [json.loads(x) for x in fake_collator.read_text().splitlines()]
    assert len(calls) == len(audit) >= 2
    first_end = min(calls, key=lambda c: c["start"])["end"]
    assert all(c["start"] >= first_end for c in sorted(calls, key=lambda c: c["start"])[1:]), \
        "the first request warms the prompt cache before the others start"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_backend"] == "subscription" and manifest["claude_code_version"].startswith("2.1.296")
    subscription_digest = manifest["instrument_digests"]["collate"]

    # Later stages without --backend adopt the run's backend; a conflicting one is refused.
    assert cli("build") == 0
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_backend"] == "subscription"
    assert manifest["instrument_digests"]["collate"] == subscription_digest
    capsys.readouterr()
    assert cli("build", "--backend", "api") == 2
    assert "produced with the subscription backend" in capsys.readouterr().err
    # Offline replay finds the subscription cache without being told the backend.
    assert cli("collate", "--offline", "--replicates", "2") == 0


def test_prereg_freeze_honours_the_backend_option(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert main(["prereg", "freeze", "--backend", "subscription", "--root", str(root)]) == 0
    doc = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    settings = load_settings(root)
    sub = dataclasses.replace(settings, llm={**settings.llm, "backend": "subscription"})
    assert doc["frozen"] is True and doc["llm_backend"] == "subscription"
    assert doc["instrument_digests"] == instrument_digests(sub) != instrument_digests(settings)


def test_a_pinned_cli_version_enters_the_digest(tmp_path: Path) -> None:
    settings = load_settings(make_root(tmp_path))
    sub = {**settings.llm, "backend": "subscription"}
    plain = instrument_digests(dataclasses.replace(settings, llm=sub))
    pinned = instrument_digests(dataclasses.replace(
        settings, llm={**sub, "subscription": {**sub["subscription"], "cli_version": "2.1.296"}}))
    assert plain["collate"] != pinned["collate"]
