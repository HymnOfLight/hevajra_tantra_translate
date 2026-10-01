"""``prereg freeze``: digests and ``frozen: true`` written in place (comments kept), refusal
when frozen, amendments appended, and the effect on gate G2's digest check."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.cli import main
from hevajra_matrix.config import load_settings
from hevajra_matrix.prereg import PreregError, freeze, instrument_digests, instruments, replace_top_level

from test_pipeline_support import REPO, context, fill_cache, make_root

TASKS = {"collate", "topics", "components", "subject", "scorer"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A root whose preregistration is the repository's own file (with its comments)."""
    root = make_root(tmp_path)
    shutil.copy(REPO / "config" / "preregistration.yaml", root / "config" / "preregistration.yaml")
    return root


def prereg_text(root: Path) -> str:
    return (root / "config" / "preregistration.yaml").read_text(encoding="utf-8")


def test_freeze_writes_digests_and_keeps_keys_order_and_comments(root: Path) -> None:
    before = yaml.safe_load(prereg_text(root))
    assert main(["prereg", "freeze", "--root", str(root)]) == 0
    text = prereg_text(root)
    after = yaml.safe_load(text)
    assert after["frozen"] is True
    assert after["instrument_digests"] == instrument_digests(load_settings(root))
    assert set(after["instrument_digests"]) == TASKS
    assert list(after) == list(before), "keys and their order are kept"
    assert {k: v for k, v in after.items() if k not in ("frozen", "instrument_digests")} == \
           {k: v for k, v in before.items() if k not in ("frozen", "instrument_digests")}
    assert text.startswith("# Pre-registration. Freeze")
    assert "primary_outcome: any                   # any = omission" in text
    assert "# written by `prereg freeze`: task -> InstrumentId digest" in text


def test_frozen_file_is_changed_only_by_an_amendment(root: Path, capsys) -> None:
    freeze(root, today=date(2026, 10, 1))
    assert main(["prereg", "freeze", "--root", str(root)]) == 2
    assert "already frozen" in capsys.readouterr().err
    with pytest.raises(PreregError, match="needs a reason"):
        freeze(root, amend="  ")
    doc = freeze(root, amend="raise the audit\nsize", today=date(2026, 10, 2))
    assert doc["amendments"] == [{"date": "2026-10-02", "reason": "raise the audit size", "changed": [],
                                  "was_frozen": True}]
    again = freeze(root, amend="second", today=date(2026, 10, 3))
    assert [a["reason"] for a in again["amendments"]] == ["raise the audit size", "second"]
    assert yaml.safe_load(prereg_text(root)) == again


def test_digest_changes_with_the_instrument(root: Path) -> None:
    settings = load_settings(root)
    base = instruments(settings)["collate"]
    llm = root / "config" / "llm.yaml"
    llm.write_text(llm.read_text(encoding="utf-8").replace("collate:    {effort: high", "collate:    {effort: xhigh"),
                   encoding="utf-8")
    changed = instruments(load_settings(root))["collate"]
    assert changed.effort == "xhigh" and changed.digest() != base.digest()
    assert instruments(load_settings(root))["topics"] == instruments(settings)["topics"]


def test_replace_top_level_keeps_neighbouring_blocks() -> None:
    text = "a: 1  # note\nb:\n  - x\n  - y\n\n# comment\nc: 3\n"
    out = replace_top_level(text, "b", {"k": "v"})
    assert out == "a: 1  # note\nb:\n  k: v\n\n# comment\nc: 3\n"
    assert replace_top_level(out, "a", 2).startswith("a: 2")


def test_freezing_makes_gate_g2_accept_the_digest(tmp_path: Path) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    reasons = json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))["reasons"]
    assert "G2: the instrument digest is not the preregistered one" in reasons
    freeze(root)
    assert main(["evaluate", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    reasons = json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))["reasons"]
    assert "G2: the instrument digest is not the preregistered one" not in reasons
