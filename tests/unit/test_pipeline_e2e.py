"""End to end on the mini fixtures: ingest -> baselines -> collate (offline, from a cache
written by a FakeClient) -> build -> evaluate -> stats -> report, through the CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.core.io import read_jsonl

from test_pipeline_support import context, fill_cache, make_root

DECIMAL = re.compile(r"(?<![\w.:])-?\d+\.\d+(?![\w.])")


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("e2e")
    root, run = make_root(tmp), tmp / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    assert fill_cache(context(root, run)) > 0
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


def test_every_stage_wrote_its_files(finished: tuple[Path, Path]) -> None:
    _, run = finished
    for rel in ("manifest.json", "llm_audit.jsonl", "ingest/report.json", "ingest/segments_zh_T0892_song.jsonl",
                "alignments/dp_zero.jsonl", "alignments/dp_anchor.jsonl", "alignments/claude.r1.jsonl",
                "alignments/claude.r3.jsonl", "alignments/claude.jsonl", "alignments/shuffled.jsonl",
                "collation/consensus.json", "matrix/cells.csv", "matrix/cells.jsonl", "matrix/machine_cells.jsonl",
                "evaluation/gate.json", "evaluation/scores.json", "stats/estimates.json", "summary.md",
                "status_strip.svg", "chapter_heatmap.svg"):
        assert (run / rel).is_file(), rel


def test_offline_replay_is_all_cache_hits_and_the_manifest_records_it(finished: tuple[Path, Path]) -> None:
    _, run = finished
    audit = read_jsonl(run / "llm_audit.jsonl")
    assert audit and all(r["from_cache"] for r in audit) and {r["task"] for r in audit} == {"collate"}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["offline"] is True
    assert manifest["stages"] == ["ingest", "baselines", "collate", "build", "evaluate", "stats", "report"]
    assert manifest["llm"]["cache_hits"] == len(audit) and manifest["llm"]["cache_misses"] == 0
    assert manifest["llm"]["served_models"] == {"collate": ["claude-opus-5-5"]}
    assert set(manifest["instrument_digests"]) == {"collate", "topics", "components", "subject", "scorer"}
    assert {"raw:cbeta", "raw:derge"} <= set(manifest["inputs"])
    assert all(len(rec["sha256"]) == 64 for rec in manifest["inputs"].values())
    assert "preregistration.yaml" in manifest["config_sha256"]


def test_level_0_report_prints_counts_but_no_rate_interval_or_delta(finished: tuple[Path, Path]) -> None:
    _, run = finished
    gate = json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))
    assert gate["level"] == 0          # digest not preregistered, no perturbations: G2 fails
    text = (run / "summary.md").read_text(encoding="utf-8")
    assert "Report level 0 (DESCRIPTIVE)" in text
    assert "## Counts (unvalidated instrument output)" in text
    assert "| outcome | Claude consensus | P1 length-only DP | B0 anchor DP |" in text
    assert not DECIMAL.search(text), DECIMAL.search(text)
    assert "%" not in text and "Delta" not in text and "\u0394" not in text
    assert not re.search(r"\[\s*-?\d", text), "no interval at level 0"
    assert "P2 shuffled" not in text.split("## Human-verified")[0].split("| outcome |")[1]
    assert "## Alignment scores" not in text and "## Witness-only material" not in text


def test_report_scope_estimands_and_human_counts(finished: tuple[Path, Path]) -> None:
    root, run = finished
    text = (run / "summary.md").read_text(encoding="utf-8")
    for line in ("relative to Derge D417-418 (revised by gZhon nu dpal), not to the Sanskrit",
                 "Chinese = Ming-based Taisho T0892; transmission changes are included",
                 "units = provisional Derge pada/clause segments"):
        assert f"- {line}" in text
    assert "E3 decomposition of deviations (Vorlage, shared, residual): NOT_ESTIMABLE: G4: no manuscript column; " \
           "reference is the co-witness" in text
    assert re.search(r"E1 share of units deviating \(D_any\): NOT_ESTIMABLE: G\d: ", text)
    assert "## Human-verified counts (grade A)" in text
    assert "**Exploratory**" in text
    estimates = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))
    assert estimates["E1_any"]["not_estimable"].startswith("G3: ")     # no phase-2 verdicts
    assert estimates["E3"]["not_estimable"] == "G4: no manuscript column; reference is the co-witness"


def test_counts_match_the_echo_collator(finished: tuple[Path, Path]) -> None:
    _, run = finished
    consensus = read_jsonl(run / "alignments" / "claude.jsonl")[1:]
    units = [r for r in consensus if r["ref_id"]]
    absent = sum(1 for r in units if r["relation"] == "no_counterpart")
    text = (run / "summary.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| absent (no counterpart) |"))
    assert row.split("|")[2].strip() == str(absent) and absent > 0
    grades = json.loads((run / "collation" / "consensus.json").read_text(encoding="utf-8"))["grades"]
    assert set(grades.values()) == {"B"}, "three identical replicates are unanimous"


def test_offline_miss_is_a_clear_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    assert main(["collate", "--offline", "--root", str(root), "--run-dir", str(run)]) == 2
    assert "not cached" in capsys.readouterr().err
