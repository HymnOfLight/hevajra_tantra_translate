"""llm.claude_code_client: the subscription backend (``claude -p``) without the real CLI.

The command line and environment are tested directly; the client with an injected
runner; the process handling with small shell scripts; and the pipeline end to end with a
fake ``claude`` executable on PATH. Result mapping is in ``test_llm_claude_code_result``.
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
from hevajra_matrix.collate import collator
from hevajra_matrix.config import load_settings
from hevajra_matrix.llm import claude_code_client as ccc
from hevajra_matrix.llm.cache import CachedClient, response_to_dict
from hevajra_matrix.llm.claude_code_client import (
    CliSettings,
    CliTimeout,
    ClaudeCodeClient,
    ModelUnavailable,
    UsageLimitReached,
    _run,
    build_argv,
    child_env,
    run_timeout,
    system_text,
)
from hevajra_matrix.llm.client import ConfigurationError, LLMRequest, LLMResponse, LLMTransportError
from hevajra_matrix.llm.schema import strict
from hevajra_matrix.pipeline import make_client, plan_collation
from hevajra_matrix.pipeline.context import cache_root
from hevajra_matrix.prereg import instrument_digests

from test_pipeline_support import context, make_root

MODEL = "claude-opus-5-5"
SCHEMA = strict({"type": "object", "properties": {"ok": {"type": "boolean"}}})
SUBSCRIBED = json.dumps({"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"})


def request(context_text: str | None = "CONTEXT", max_tokens: int = 2048) -> LLMRequest:
    return LLMRequest(task="check", prompt_sha="p", system="SYSTEM", body="BODY", schema=SCHEMA, effort="high",
                      max_tokens=max_tokens, context=context_text)


def result(**overrides) -> dict:
    base = {"type": "result", "subtype": "success", "is_error": False, "stop_reason": "end_turn",
            "session_id": "sess-1", "structured_output": {"ok": True}, "result": "",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "modelUsage": {MODEL: {"inputTokens": 10, "outputTokens": 5}}}
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def not_aborted():
    ccc.ABORTED.clear()
    yield
    ccc.ABORTED.clear()


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
    assert env["CLAUDE_CODE_NO_MODEL_FALLBACK"] == "1" and env["DISABLE_AUTOUPDATER"] == "1"
    assert "CLAUDE_CONFIG_DIR" not in env, "a stored login keeps the host configuration"


def test_an_oauth_token_gets_a_fresh_config_directory(tmp_path: Path) -> None:
    env = child_env(request(), {"CLAUDE_CODE_OAUTH_TOKEN": "tok", "HOME": "/root"}, tmp_path / "config")
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok" and env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "config")
    own = child_env(request(), {"CLAUDE_CODE_OAUTH_TOKEN": "tok", "CLAUDE_CONFIG_DIR": "/mine"}, tmp_path / "c")
    assert own["CLAUDE_CONFIG_DIR"] == "/mine"


def test_run_timeout_is_never_below_the_output_cap_at_a_slow_rate() -> None:
    settings = CliSettings(timeout_s=3600)
    assert run_timeout(request(max_tokens=2048), settings) == 3600
    assert run_timeout(request(max_tokens=128000), settings) == 128000 / ccc.MIN_TOKENS_PER_S


# --------------------------------------------------------------------------- client
def scripted(outputs: list, seen: list | None = None, auth: str = SUBSCRIBED, version: str = "2.1.296 (Claude Code)"):
    """A runner answering --version and auth status, then the scripted outputs in order."""
    def runner(argv, stdin_text, env, cwd, timeout_s):
        if argv[1:] == ["--version"]:
            return 0, version, ""
        if argv[1:] == ["auth", "status", "--json"]:
            return 0, auth, ""
        if seen is not None:
            system_file = Path(argv[argv.index("--system-prompt-file") + 1])
            seen.append((stdin_text, system_file.read_text(encoding="utf-8"), cwd, env, timeout_s))
        item = outputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
    return runner


def client(outputs: list, tmp_path: Path, **kwargs) -> ClaudeCodeClient:
    settings = kwargs.pop("settings", CliSettings())
    return ClaudeCodeClient(settings, runner=scripted(outputs, **kwargs), sleep=lambda s: None, environ={},
                            session_dir=tmp_path)


def test_client_runs_one_process_and_retries_transport_failures(tmp_path: Path) -> None:
    seen, pauses = [], []
    runner = scripted([(1, "", "connection reset"), (0, json.dumps(result()), "")], seen)
    c = ClaudeCodeClient(CliSettings(max_retries=2), runner=runner, sleep=pauses.append,
                         environ={"ANTHROPIC_API_KEY": "k"}, session_dir=tmp_path)
    r = c.complete(request())
    assert r.status == "ok" and r.key == request().key() and len(seen) == 2 and pauses == [30.0]
    assert r.client_info["cli_version"] == "2.1.296" and r.client_info["auth"] == "oauth_token (firstParty)"
    assert "seconds" in r.client_info
    stdin_text, system, cwd, env, timeout_s = seen[0]
    assert stdin_text == "BODY" and system == "SYSTEM\n\nCONTEXT" and "ANTHROPIC_API_KEY" not in env
    assert timeout_s == CliSettings().timeout_s
    assert not Path(cwd).exists(), "the scratch directory is removed after the call"


@pytest.mark.parametrize("payload, status", [
    (result(subtype="error_max_turns", is_error=True, structured_output=None), "invalid"),
    (result(subtype="error_max_structured_output_retries", is_error=True, structured_output=None), "invalid"),
    (result(is_error=True, api_error="max_output_tokens", stop_reason="stop_sequence", structured_output=None),
     "truncated"),
])
def test_research_outcomes_take_one_run_and_no_retry(tmp_path: Path, payload, status) -> None:
    outputs = [(0, json.dumps(payload), "")]
    assert client(outputs, tmp_path).complete(request()).status == status and not outputs


@pytest.mark.parametrize("failure, error", [
    ((0, json.dumps(result(is_error=True, structured_output=None, result="You've hit your session limit")), ""),
     UsageLimitReached),
    ((0, json.dumps(result(is_error=True, structured_output=None, api_error="model_substitution_disabled")), ""),
     ModelUnavailable),
    (CliTimeout("too long"), CliTimeout),
])
def test_usage_limit_unavailable_model_and_timeout_are_not_retried(tmp_path: Path, failure, error) -> None:
    outputs = [failure, (0, json.dumps(result()), "")]
    with pytest.raises(error):
        client(outputs, tmp_path, settings=CliSettings(max_retries=3)).complete(request())
    assert len(outputs) == 1


def test_nothing_is_retried_or_started_after_abort(tmp_path: Path) -> None:
    outputs = [(1, "", "connection reset"), (0, json.dumps(result()), "")]
    c = client(outputs, tmp_path, settings=CliSettings(max_retries=3))
    c.sleep = lambda s: ccc.abort()
    with pytest.raises(LLMTransportError):
        c.complete(request())
    assert len(outputs) == 1, "the retry after the pause did not start"
    with pytest.raises(LLMTransportError, match="aborted"):
        client([(0, json.dumps(result()), "")], tmp_path).complete(request())


# --------------------------------------------------------------------------- preflight
def test_an_api_key_file_of_a_cloud_session_is_refused_even_with_a_token(tmp_path: Path) -> None:
    (tmp_path / ".api_key").write_text("x")
    outputs = [(0, json.dumps(result()), "")]
    c = ClaudeCodeClient(runner=scripted(outputs), environ={"CLAUDE_CODE_OAUTH_TOKEN": "t"}, session_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="API key file"):
        c.complete(request())
    assert outputs, "no request was sent"


@pytest.mark.parametrize("auth", [
    json.dumps({"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty"}),
    json.dumps({"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "bedrock"}),
    json.dumps({"loggedIn": False}),
    "not json",
])
def test_anything_but_a_first_party_subscription_login_is_refused(tmp_path: Path, auth) -> None:
    outputs = [(0, json.dumps(result()), "")]
    with pytest.raises(ConfigurationError, match="subscription login"):
        client(outputs, tmp_path, auth=auth).complete(request())
    assert outputs, "no request was sent"


def test_a_claude_ai_login_is_accepted(tmp_path: Path) -> None:
    auth = json.dumps({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"})
    assert client([(0, json.dumps(result()), "")], tmp_path, auth=auth).complete(request()).status == "ok"


@pytest.mark.parametrize("pin, accepted", [("2.1.296", True), ("2.1.29", False), ("2.1", False), ("9.0.0", False)])
def test_a_pinned_cli_version_must_match_exactly(tmp_path: Path, pin, accepted) -> None:
    c = client([(0, json.dumps(result()), "")], tmp_path, settings=CliSettings(cli_version=pin))
    if accepted:
        assert c.complete(request()).status == "ok"
    else:
        with pytest.raises(ConfigurationError, match="pinned"):
            c.complete(request())


def test_missing_executable_is_a_configuration_error(tmp_path: Path) -> None:
    c = ClaudeCodeClient(CliSettings(command="/nonexistent/claude-binary"), environ={}, session_dir=tmp_path)
    with pytest.raises(ConfigurationError):
        c.complete(request())


# --------------------------------------------------------------------------- processes
@pytest.mark.skipif(os.name != "posix", reason="POSIX shell")
def test_a_timed_out_child_is_stopped_promptly_even_if_a_helper_holds_its_pipes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ccc, "REAP_TIMEOUT_S", 0.5)
    started = time.monotonic()
    with pytest.raises(CliTimeout):
        _run(["/bin/sh", "-c", "(sleep 30) & sleep 30"], "", {"PATH": os.environ["PATH"]}, tmp_path, 0.5)
    assert time.monotonic() - started < 5


@pytest.mark.skipif(os.name != "posix", reason="POSIX shell")
def test_a_normal_run_returns_its_output(tmp_path: Path) -> None:
    assert _run(["/bin/sh", "-c", "cat; echo done"], "in\n", {"PATH": os.environ["PATH"]}, tmp_path, 10) == \
        (0, "in\ndone\n", "")


# --------------------------------------------------------------------------- collator warm-up
class _Recorder:
    """A client that records start/end order and pretends some requests are cached."""

    def __init__(self, cached: set[str]) -> None:
        self.cached, self.events = cached, []

    def is_cached(self, req: LLMRequest) -> bool:
        return req.key() in self.cached

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.events.append(("start", req.key()))
        time.sleep(0.05)
        self.events.append(("end", req.key()))
        return LLMResponse(key=req.key(), status="invalid", data=None, raw_text="", requested_model=req.model,
                           served_model=req.model, stop_reason="end_turn")


def test_the_first_uncached_request_runs_alone_and_order_is_kept(tmp_path: Path) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    main(["ingest", "--root", str(root), "--run-dir", str(run)])
    plan = plan_collation(context(root, run), replicates=2)
    jobs = plan.requests()
    assert len(jobs) >= 3
    recorder = _Recorder({jobs[0][2].key()})
    out = collator.collate(plan.windows, recorder, plan.settings, plan.tags, plan.examples, plan.template, workers=4)
    warm = jobs[1][2].key()
    assert recorder.events[:2] == [("start", warm), ("end", warm)], "the first uncached request ran alone"
    assert {tag: len(parsed) for tag, parsed in out.items()} == {tag: len(plan.windows) for tag in plan.tags}


# --------------------------------------------------------------------------- end to end
FAKE_CLAUDE = '''#!{python}
import json, sys, time
sys.path.insert(0, {support!r})
from types import SimpleNamespace
from test_pipeline_support import echo_script
argv = sys.argv[1:]
if argv == ["--version"]:
    print("2.1.296 (Claude Code)")
    sys.exit(0)
if argv == ["auth", "status", "--json"]:
    print(json.dumps({{"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"}}))
    sys.exit(0)
started = time.time()
system = open(argv[argv.index("--system-prompt-file") + 1], encoding="utf-8").read()
body = sys.stdin.read()
if "WITNESS TEXT" in system:
    answer = echo_script(SimpleNamespace(context=system[system.index("WITNESS TEXT"):], body=body))
else:
    answer = {{"ok": True}}
time.sleep(0.2)
with open({log!r}, "a") as handle:
    handle.write(json.dumps({{"start": started, "end": time.time()}}) + "\\n")
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "stop_reason": "tool_use",
                  "session_id": "fake", "num_turns": 2, "structured_output": answer,
                  "usage": {{"input_tokens": 50, "output_tokens": 40, "cache_read_input_tokens": 900}},
                  "modelUsage": {{argv[argv.index("--model") + 1]: {{"outputTokens": 40}}}}}}))
'''


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch) -> Path:
    bin_dir, log = tmp_path / "bin", tmp_path / "calls.jsonl"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text(FAKE_CLAUDE.format(python=sys.executable, support=str(Path(__file__).parent), log=str(log)),
                   encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-the-cli")
    no_session = tmp_path / "no-session"
    monkeypatch.setattr(ClaudeCodeClient.__init__, "__defaults__",
                        (None, None, ccc.ABORTED.wait, None, no_session))
    return log


posix_only = pytest.mark.skipif(os.name != "posix", reason="the fake executable is a POSIX script")


@posix_only
def test_claude_check_on_the_subscription_backend(tmp_path: Path, fake_claude: Path) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["claude-check", "--backend", "subscription", "--root", str(root), "--run-dir", str(run)]) == 0
    summary = json.loads((run / "claude_check.json").read_text(encoding="utf-8"))
    assert summary["passed"] and summary["backend"] == "subscription" and summary["served_model"] == MODEL
    assert summary["claude_code_version"].startswith("2.1.296")
    assert summary["credential_source"] == "oauth_token (firstParty)"
    assert (run / "llm_backend.txt").read_text().strip() == "subscription"
    assert (root / "runs" / "llm-cache" / "llm_spend.subscription.jsonl").is_file()
    assert not (root / "runs" / "llm-cache" / "llm_spend.jsonl").exists(), "API budget untouched"


@posix_only
def test_collate_on_the_subscription_backend_end_to_end(tmp_path: Path, fake_claude: Path, capsys) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    cli = lambda *a: main([*a, "--root", str(root), "--run-dir", str(run)])  # noqa: E731
    cli("ingest")
    assert cli("collate", "--backend", "subscription", "--replicates", "2") == 0
    assert (run / "llm_backend.txt").read_text().strip() == "subscription"
    cache = root / "runs" / "llm-cache"
    assert any((cache / "subscription").rglob("*.json")) and not (cache / MODEL).exists()
    audit = [json.loads(x) for x in (run / "llm_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert audit and all(a["status"] == "ok" and a["client_info"]["cli_version"] == "2.1.296" for a in audit)
    calls = [json.loads(x) for x in fake_claude.read_text().splitlines()]
    assert len(calls) == len(audit) >= 2
    ordered = sorted(calls, key=lambda c: c["start"])
    assert all(c["start"] >= ordered[0]["end"] for c in ordered[1:]), "the first request warms the prompt cache"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_backend"] == "subscription"
    assert manifest["llm"]["cli_versions"] == ["2.1.296"] and manifest["llm"]["logins"] == ["oauth_token (firstParty)"]
    subscription_digest = manifest["instrument_digests"]["collate"]

    # Later stages without --backend adopt the run's backend; a conflicting one is refused.
    assert cli("build") == 0
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_backend"] == "subscription" and manifest["instrument_digests"]["collate"] == subscription_digest
    capsys.readouterr()
    assert cli("build", "--backend", "api") == 2
    assert "bound to the subscription backend" in capsys.readouterr().err
    # Offline replay finds the subscription cache without being told the backend.
    assert cli("collate", "--offline", "--replicates", "2") == 0


def test_a_failed_attempt_does_not_bind_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root, run = make_root(tmp_path), tmp_path / "run"
    main(["ingest", "--root", str(root), "--run-dir", str(run)])
    assert main(["collate", "--offline", "--root", str(root), "--run-dir", str(run)]) == 2, "empty cache"
    assert not (run / "llm_backend.txt").exists()


def test_subscription_client_composition_and_separate_cache(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    ctx = context(root, tmp_path / "run")
    sub = dataclasses.replace(ctx, settings=dataclasses.replace(
        ctx.settings, llm={**ctx.settings.llm, "backend": "subscription"}))
    made = make_client(sub)
    assert isinstance(made.inner, CachedClient) and isinstance(made.inner.inner, ClaudeCodeClient)
    assert made.inner.root == ctx.settings.path("cache") / "subscription"
    assert made.budget_usd == float("inf")
    api, subscription = instrument_digests(ctx.settings), instrument_digests(sub.settings)
    assert api.keys() == subscription.keys() and all(api[t] != subscription[t] for t in api), \
        "the backend is part of every instrument"


def test_a_pinned_cli_version_has_its_own_cache_and_digest(tmp_path: Path) -> None:
    settings = load_settings(make_root(tmp_path))
    sub = {**settings.llm, "backend": "subscription"}
    plain = dataclasses.replace(settings, llm=sub)
    pinned = dataclasses.replace(settings, llm={**sub, "subscription": {**sub["subscription"], "cli_version": "2.1.296"}})
    assert instrument_digests(plain)["collate"] != instrument_digests(pinned)["collate"]
    assert cache_root(pinned) == cache_root(plain) / "cli-2.1.296"


def test_api_cache_entries_stay_readable_by_older_code() -> None:
    entry = response_to_dict(LLMResponse(key="k", status="ok", data={}, raw_text="", requested_model=MODEL,
                                         served_model=MODEL, stop_reason="end_turn"))
    assert "client_info" not in entry


def test_prereg_freeze_honours_the_backend_option(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert main(["prereg", "freeze", "--backend", "subscription", "--root", str(root)]) == 0
    doc = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    settings = load_settings(root)
    sub = dataclasses.replace(settings, llm={**settings.llm, "backend": "subscription"})
    assert doc["frozen"] is True and doc["llm_backend"] == "subscription"
    assert doc["instrument_digests"] == instrument_digests(sub) != instrument_digests(settings)


def test_unknown_backend_is_refused(tmp_path: Path, capsys) -> None:
    root = make_root(tmp_path)
    path = root / "config" / "llm.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("backend: api", "backend: local"), encoding="utf-8")
    assert main(["ingest", "--root", str(root), "--run-dir", str(tmp_path / "run")]) == 2
    assert "backend 'local'" in capsys.readouterr().err
