"""Which evaluation gates, and how often the test set counts as scored (synthesis 5.4, 5.5).

Only the Claude consensus scored on test gold gates: dev-gold and baselines-only evaluations
never replace its ``gate.json``/``scores.json``. Re-gating identical output on identical gold
(to refresh G2/G3 after review) does not append a second ledger record. ``run`` does not
abort when test gold exists before the freeze, and ``report`` flags a gate older than the
matrix it describes.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.matrix.build import segment_fingerprint
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.pipeline.store import read_alignment
from hevajra_matrix.prereg import freeze
from hevajra_matrix.topics import TopicLabel, write_labels

from test_pipeline_support import context, fill_cache, make_root

LEDGER = Path("data") / "ledger" / "test_evaluations.jsonl"


def cli(root: Path, run: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--run-dir", str(run), "--offline"])


def fill_gold(root: Path, run: Path, gold_set: str) -> None:
    """Export the ``gold_set`` sheets, answer them with the consensus links, import them."""
    assert cli(root, run, "review", "export", "--task", "gold", "--set", gold_set) == 0
    links = read_alignment(run / "alignments" / "claude.jsonl").by_ref()
    for ref_sheet in sorted((run / "review" / "gold").glob(f"{gold_set}_*.ref.csv")):
        wit_sheet = ref_sheet.with_name(ref_sheet.name.replace(".ref.csv", ".wit.csv"))
        with wit_sheet.open(encoding="utf-8-sig", newline="") as fh:
            handle = {r["seg_id"]: r["handle"] for r in csv.DictReader(fh)}
        with ref_sheet.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            rows, columns = list(reader), reader.fieldnames
        for row in rows:
            link = links[row["unit_id"]]
            row["links"] = " ".join(handle[w] for w in link.wit_ids)
            row["relation"] = str(link.relation)
        with ref_sheet.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        assert cli(root, run, "review", "import", "--task", "gold", "--file", str(ref_sheet), "--annotator", "ann",
                   "--date", "2026-09-30") == 0


@pytest.fixture
def collated(tmp_path: Path) -> tuple[Path, Path]:
    """A finished offline run with test windows drawn and test and dev gold annotated."""
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    assert cli(root, run, "sample", "windows") == 0
    fill_gold(root, run, "test")
    fill_gold(root, run, "dev")
    return root, run


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_before_the_freeze_falls_back_to_dev_gold_and_still_reports(collated, capsys) -> None:
    root, run = collated
    assert cli(root, run, "evaluate") == 2, "test gold is refused against Claude before the freeze"
    capsys.readouterr()
    assert cli(root, run, "run") == 0
    out = capsys.readouterr().out
    assert "scoring dev gold instead" in out and "report:" in out
    assert (run / "stats" / "estimates.json").is_file() and (run / "summary.md").is_file()
    gate = _json(run / "evaluation" / "gate.json")
    assert gate["gating"] is False and gate["gold_set"] == "dev" and gate["level"] == 0
    assert gate["reasons"][0].startswith("G1: not a gating evaluation (dev gold)")
    assert not (root / LEDGER).exists()


def test_only_the_test_gold_claude_evaluation_gates_and_is_ledgered_once(collated, capsys) -> None:
    root, run = collated
    freeze(root)
    assert cli(root, run, "evaluate") == 0
    gate = _json(run / "evaluation" / "gate.json")
    scores = _json(run / "evaluation" / "scores.json")
    assert gate["gating"] is True and gate["gold_set"] == "test" and scores["gold_set"] == "test"
    assert len((root / LEDGER).read_text(encoding="utf-8").splitlines()) == 1

    capsys.readouterr()
    assert cli(root, run, "evaluate") == 0, "re-gating, e.g. after review verdicts"
    assert "already recorded; not appended again" in capsys.readouterr().out
    assert len((root / LEDGER).read_text(encoding="utf-8").splitlines()) == 1, "the same scoring counts once"
    assert _json(run / "evaluation" / "gate.json")["confirmatory"] == gate["confirmatory"]

    for extra in (["--gold", "dev"], ["--baselines-only"], ["--gold", "dev", "--baselines-only"]):
        assert cli(root, run, "evaluate", *extra) == 0
        assert _json(run / "evaluation" / "gate.json") == gate, f"{extra} replaced the test gate"
        assert _json(run / "evaluation" / "scores.json")["gold_set"] == "test", f"{extra} replaced the test scores"
    dev = _json(run / "evaluation" / "scores.dev.json")
    assert dev["gold_set"] == "dev" and "claude:consensus" in dev["sources"]
    assert set(_json(run / "evaluation" / "scores.test_baselines.json")["sources"]) == {"dp:zero", "dp:anchor"}
    assert len((root / LEDGER).read_text(encoding="utf-8").splitlines()) == 1


def test_topic_labels_added_after_the_test_scoring_do_not_make_a_new_scoring(collated) -> None:
    # Campaign order: evaluate once on test gold, then label topics, then re-gate after
    # review. The labels add refusal-rate breakdowns by topic group to the metrics; that
    # must not count as a second scoring of the test set (it cost the confirmatory flag).
    root, run = collated
    freeze(root)
    assert cli(root, run, "evaluate") == 0
    gate = _json(run / "evaluation" / "gate.json")
    texts = load_texts(context(root, run))
    write_labels(root / "data" / "annotations" / "topics" / f"{texts.reference_id}.csv",
                 [TopicLabel(u.id, segment_fingerprint(u), topics=frozenset({"neutral"}), coder="c", date="2026-10-01")
                  for u in texts.units])
    assert cli(root, run, "evaluate") == 0
    scores = _json(run / "evaluation" / "scores.json")["sources"]["claude:consensus"]["metrics"]
    assert "refusal_rate:neutral" in scores
    assert len((root / LEDGER).read_text(encoding="utf-8").splitlines()) == 1
    assert _json(run / "evaluation" / "gate.json")["confirmatory"] == gate["confirmatory"]


def test_a_new_scoring_of_the_test_set_is_ledgered(collated) -> None:
    root, run = collated
    freeze(root)
    assert cli(root, run, "evaluate") == 0
    ledger = root / LEDGER
    record = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    record["metrics"]["claude:consensus"]["link_f1"] = 0.5          # a different (earlier) scoring
    ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert cli(root, run, "evaluate") == 0
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2
    assert "scored 2 times" in _json(run / "evaluation" / "gate.json")["scope_note"]


def test_report_flags_a_gate_older_than_the_matrix(collated, capsys) -> None:
    root, run = collated
    assert cli(root, run, "evaluate", "--gold", "dev") == 0
    cells = run / "matrix" / "cells.jsonl"
    later = (run / "evaluation" / "gate.json").stat().st_mtime + 60
    os.utime(cells, (later, later))
    capsys.readouterr()
    assert cli(root, run, "report") == 0
    assert "WARNING evaluation/gate.json predates matrix/cells.jsonl" in capsys.readouterr().out
    assert "predates matrix/cells.jsonl" in (run / "summary.md").read_text(encoding="utf-8")
