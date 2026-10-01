"""The human-review loop on a finished offline fixture run: verification plan -> blind sheet
-> import -> reveal -> import -> rebuild -> stats and report; and the two topic sheets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.review import verdicts as verdict_files
from hevajra_matrix.topics import load_codebook, load_labels

from test_pipeline_support import context, fill_cache, fill_prelabel_cache, make_root

WITNESS = "zh_T0892_song"


@pytest.fixture
def finished(tmp_path: Path) -> tuple[Path, Path]:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


def cli(root: Path, run: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--run-dir", str(run), "--offline"])


def fill(sheet: Path, **values: str) -> int:
    """Set columns of every row of a sheet (BOM kept); returns the number of rows."""
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, columns = list(reader), reader.fieldnames
    for row in rows:
        row.update(values)
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def test_verification_blind_reveal_and_back_into_the_matrix(finished, capsys) -> None:
    root, run = finished
    assert cli(root, run, "sample", "verification") == 0
    assert cli(root, run, "sample", "verification") == 2, "a plan is drawn once"
    assert cli(root, run, "review", "export", "--task", "verify") == 0
    blind = run / "review" / "verify.blind.csv"
    header = blind.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "machine_" not in header, "blind sheets show no machine output"
    n = fill(blind, blind_relation="no_counterpart")
    assert n == 5, "the echo collator's five no_counterpart units are a census class"
    assert cli(root, run, "review", "import", "--task", "verify", "--file", str(blind), "--annotator", "ann",
               "--date", "2026-09-30") == 0
    committed = root / "data" / "annotations" / "verdicts" / WITNESS / "verify.csv"
    saved = verdict_files.read_file(committed)
    assert len(saved) == 5 and all(v.stratum == "pos:absent:other" and v.inclusion_prob == 1.0 for v in saved)
    assert "ref_text" not in committed.read_text(encoding="utf-8").splitlines()[0]

    capsys.readouterr()
    assert cli(root, run, "review", "status") == 0
    assert "pos:absent:other final 0/5 (blind 5)" in capsys.readouterr().out

    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "verify") == 0
    reveal = run / "review" / "verify.reveal.csv"
    fill(reveal, final_relation="no_counterpart", final_wit_loci="", final_flags="")
    assert cli(root, run, "review", "reveal", "--file", str(reveal), "--date", "2026-10-01") == 0
    final = verdict_files.read_file(committed)
    assert all(v.final_relation == "no_counterpart" and v.machine_relation == "no_counterpart" for v in final)
    assert all(v.instrument_digest for v in final)

    assert cli(root, run, "build") == 0
    cells = [json.loads(line) for line in (run / "matrix" / "cells.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for c in cells if c["grade"] == "A") == 5
    assert cli(root, run, "stats") == 0
    estimates = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))
    # the five machine negatives (neg:B:other) have no audit verdict yet: imputing them would
    # report the Dirichlet prior, so E1 is NOT_ESTIMABLE and names the stratum
    for name in ("E1_any", "E1_any_blind", "E2_any"):
        assert estimates[name]["not_estimable"] == "G3: no phase-2 sample verdict in stratum neg:B:other (5 unverified)"
    details = json.loads((run / "stats" / "details.json").read_text(encoding="utf-8"))
    assert details["uncalibrated"] == {"final": {"neg:B:other": 5}, "blind": {"neg:B:other": 5}}
    assert cli(root, run, "report") == 0
    text = (run / "summary.md").read_text(encoding="utf-8")
    assert "| absent (no counterpart) | 5 |" in text.split("## Human-verified counts (grade A)")[1]

    assert cli(root, run, "sample", "audit") == 0
    plan = (run / "review" / "plan_audit.csv").read_text(encoding="utf-8").splitlines()
    assert len(plan) == 1 + 5 and all(",audit," in line for line in plan[1:])
    assert cli(root, run, "evaluate") == 0
    gate = json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))
    assert any("neg:B:other has 5 unverified" in r for r in gate["reasons"]), "G3 sees the uncalibrated stratum"

    # blind-only audit verdicts calibrate the blind column but are not final decisions (A1)
    assert cli(root, run, "review", "export", "--task", "audit") == 0
    audit_blind = run / "review" / "audit.blind.csv"
    with audit_blind.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, columns = list(reader), reader.fieldnames
    for row in rows:
        row.update(blind_relation="equivalent", blind_wit_loci=row["zh_context_loci"].split()[0])
    with audit_blind.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 5
    assert cli(root, run, "review", "import", "--task", "audit", "--file", str(audit_blind), "--annotator", "ann",
               "--date", "2026-10-02") == 0
    assert cli(root, run, "stats") == 0
    estimates = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))
    assert estimates["E1_any"]["not_estimable"].startswith("G3: no phase-2 sample verdict in stratum neg:B:other")
    assert estimates["E1_any_blind"]["not_estimable"] is None
    assert estimates["E1_any_blind"]["point"] == pytest.approx(0.5)      # 5 absent + 5 equivalent, all verified

    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "audit") == 0
    audit_reveal = run / "review" / "audit.reveal.csv"
    assert fill(audit_reveal) == 5                         # final columns come prefilled with the blind decision
    assert cli(root, run, "review", "reveal", "--file", str(audit_reveal), "--date", "2026-10-03") == 0
    assert cli(root, run, "build") == 0
    assert cli(root, run, "stats") == 0
    estimates = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))
    assert estimates["E1_any"]["not_estimable"] is None and estimates["E1_any"]["point"] == pytest.approx(0.5)
    assert estimates["E1_any_prior"]["not_estimable"] is None                   # prior sensitivity (symmetric in D)


def test_a_blind_sheet_reimported_after_the_reveal_keeps_the_final_decisions(finished, capsys) -> None:
    # Re-importing a blind sheet used to replace the revealed verdicts wholesale: every final
    # decision was dropped (300 of 300 on the real-text campaign) and the blind answer could
    # be rewritten after the reviewer had seen the machine output.
    root, run = finished
    assert cli(root, run, "sample", "verification") == 0
    assert cli(root, run, "review", "export", "--task", "verify") == 0
    blind = run / "review" / "verify.blind.csv"
    fill(blind, blind_relation="no_counterpart")
    args = ("review", "import", "--task", "verify", "--file", str(blind), "--annotator", "ann")
    assert cli(root, run, *args) == 0
    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "verify") == 0
    assert cli(root, run, "review", "reveal", "--file", str(run / "review" / "verify.reveal.csv")) == 0
    committed = root / "data" / "annotations" / "verdicts" / WITNESS / "verify.csv"
    revealed = committed.read_text(encoding="utf-8")
    assert all(v.final_relation == "no_counterpart" for v in verdict_files.read_file(committed))

    assert cli(root, run, *args) == 0, "the same blind sheet again is harmless"
    assert committed.read_text(encoding="utf-8") == revealed
    fill(blind, blind_relation="unresolved")
    capsys.readouterr()
    assert cli(root, run, *args) == 2
    assert "a blind decision is fixed once revealed" in capsys.readouterr().err
    assert committed.read_text(encoding="utf-8") == revealed, "nothing was imported"


def test_topic_prelabels_first_and_blind_second_coder(finished) -> None:
    root, run = finished
    assert fill_prelabel_cache(context(root, run, offline=True)) == 1
    assert cli(root, run, "topics", "prelabel") == 0
    prelabels = [json.loads(x) for x in (run / "topics" / "prelabels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(prelabels) == 10 and all(p["topics"] == ["neutral"] and p["reason"] is None for p in prelabels)
    assert cli(root, run, "topics", "export") == 0
    first = run / "review" / "topics_first.csv"
    with first.open(encoding="utf-8-sig", newline="") as fh:
        assert {r["prelabel_topics"] for r in csv.DictReader(fh)} == {"neutral"}
    assert fill(first, topics="neutral") == 10
    assert cli(root, run, "topics", "import", "--file", str(first), "--annotator", "coder-a",
               "--date", "2026-09-30") == 0
    labels_path = root / "data" / "annotations" / "topics" / "bo_derge_D417_418.csv"
    codebook = load_codebook(root / "data" / "codebook" / "topics.yaml")
    labels = load_labels(labels_path, codebook)
    assert len(labels) == 10 and all(x.prelabel_topics == {"neutral"} for x in labels.values())
    assert cli(root, run, "review", "export", "--task", "topics_second") == 0
    second = run / "review" / "topics_second.second.csv"
    assert "prelabel" not in second.read_text(encoding="utf-8-sig").splitlines()[0], "critique A6"
    assert fill(second, topics="neutral") == 2, "20% of the labelled units"
    assert cli(root, run, "review", "import", "--task", "topics_second", "--file", str(second), "--annotator",
               "coder-b", "--date", "2026-10-01") == 0
    labels = load_labels(labels_path, codebook)
    assert sum(1 for x in labels.values() if x.second_topics) == 2
    assert cli(root, run, "review", "export", "--task", "topics") == 2, "nothing left to label"
