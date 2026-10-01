"""The command line: help text, root discovery and run-directory selection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hevajra_matrix.cli import build_parser, main

from test_pipeline_support import REPO, make_root

COMMANDS = ("fetch", "ingest", "baselines", "collate", "build", "evaluate", "perturb", "stats", "report", "topics",
            "sample", "review", "components", "experiment", "claude-check", "prereg", "run")


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    env.pop("ANTHROPIC_API_KEY", None)
    return subprocess.run([sys.executable, "-m", "hevajra_matrix", *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)


def test_help_lists_every_subcommand_with_an_english_line(tmp_path: Path) -> None:
    done = _run("--help", cwd=tmp_path)
    assert done.returncode == 0
    for command in COMMANDS:
        assert f"\n    {command} " in done.stdout, command
    assert done.stdout.isascii()


@pytest.mark.parametrize("path", [("review",), ("topics",), ("prereg",), ("experiment",), ("baselines",)])
def test_nested_commands_have_help(path: tuple[str, ...], capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args([*path, "--help"])
    assert stop.value.code == 0
    assert "usage: hevajra-matrix" in capsys.readouterr().out


def test_every_subparser_has_a_one_line_help() -> None:
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    assert set(action.choices) == set(COMMANDS)
    helps = {c.dest: c.help for c in action._choices_actions}
    assert all(helps[c] and "\n" not in helps[c] for c in COMMANDS)


def test_root_option_works_from_any_directory_and_later_stages_reuse_the_latest_run(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = _run("--root", str(root), "ingest", cwd=elsewhere)
    assert done.returncode == 0, done.stderr
    runs = [p for p in (root / "runs").iterdir() if p.is_dir() and p.name != "llm-cache"]
    assert len(runs) == 1 and runs[0].name.endswith("-nogit")
    later = _run("baselines", "--root", str(root), cwd=elsewhere)     # options also after the command
    assert later.returncode == 0, later.stderr
    assert f"run directory: {runs[0]}" in later.stdout
    assert (runs[0] / "alignments" / "dp_zero.jsonl").is_file()


def test_stage_before_any_run_directory_is_a_clear_error(tmp_path: Path, capsys) -> None:
    root = make_root(tmp_path)
    assert main(["build", "--root", str(root)]) == 2
    assert "no run directory yet" in capsys.readouterr().err


def test_missing_configuration_is_reported(tmp_path: Path, capsys) -> None:
    assert main(["build", "--root", str(tmp_path)]) == 2
    assert "config/run.yaml" in capsys.readouterr().err
