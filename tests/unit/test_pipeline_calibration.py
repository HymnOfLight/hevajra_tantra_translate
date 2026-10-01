"""Two-phase facts behind G3 and the estimates, on a finished offline fixture run.

The echo collator yields 5 machine-ABSENT units (census stratum ``pos:absent:other``) and
5 machine negatives (``neg:B:other``, fewer than the audit floor of 40). Checked here:
the deferred NULL recall is computed from the two-phase data (it used to be None, which
failed G3 whenever G1 deferred the NULL metrics), a fully audited census stratum smaller
than the floor meets it, and strata are frozen when a plan is drawn so that topic labels
imported later do not move unverified units into strata no verdict calibrates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.matrix.build import segment_fingerprint
from hevajra_matrix.pipeline import human_data as ann
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.pipeline.measure import _calibration, read_built_cells
from hevajra_matrix.topics import TopicLabel, write_labels

from test_pipeline_support import context, fill_cache, make_root


@pytest.fixture
def finished(tmp_path: Path) -> tuple[Path, Path]:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


def cli(root: Path, run: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--run-dir", str(run), "--offline"])


def rewrite(sheet: Path, rows_fn) -> int:
    """Rewrite a sheet's rows with ``rows_fn(rows) -> rows`` (BOM kept); returns the row count."""
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, columns = list(reader), reader.fieldnames
    rows = rows_fn(rows)
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def verify_all_absent(root: Path, run: Path) -> None:
    assert cli(root, run, "sample", "verification") == 0
    assert cli(root, run, "review", "export", "--task", "verify") == 0
    blind = run / "review" / "verify.blind.csv"
    rewrite(blind, lambda rows: [{**r, "blind_relation": "no_counterpart"} for r in rows])
    assert cli(root, run, "review", "import", "--task", "verify", "--file", str(blind), "--annotator", "ann",
               "--date", "2026-09-30") == 0
    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "verify") == 0
    reveal = run / "review" / "verify.reveal.csv"
    rewrite(reveal, lambda rows: [{**r, "final_relation": "no_counterpart", "final_wit_loci": "", "final_flags": ""}
                                  for r in rows])
    assert cli(root, run, "review", "reveal", "--file", str(reveal), "--date", "2026-10-01") == 0


def audit(root: Path, run: Path, decisions: list[str]) -> None:
    """Audit the first ``len(decisions)`` sampled negatives with these blind relations, then reveal."""
    assert cli(root, run, "sample", "audit") == 0
    assert cli(root, run, "review", "export", "--task", "audit") == 0
    sheet = run / "review" / "audit.blind.csv"

    def decide(rows):
        out = []
        for row, relation in zip(rows, decisions):
            loci = row["zh_context_loci"].split()[0] if relation != "no_counterpart" else ""
            out.append({**row, "blind_relation": relation, "blind_wit_loci": loci})
        return out

    assert rewrite(sheet, decide) == len(decisions)
    assert cli(root, run, "review", "import", "--task", "audit", "--file", str(sheet), "--annotator", "ann",
               "--date", "2026-10-02") == 0
    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "audit") == 0
    assert cli(root, run, "review", "reveal", "--file", str(run / "review" / "audit.reveal.csv"),
               "--date", "2026-10-03") == 0
    assert cli(root, run, "build") == 0


def calibration(root: Path, run: Path):
    ctx = context(root, run, offline=True)
    texts = load_texts(ctx)
    return _calibration(ctx, texts, read_built_cells(ctx), ann.load_topics(ctx, texts))


def test_deferred_null_recall_and_a_census_audit_below_the_floor(finished) -> None:
    root, run = finished
    verify_all_absent(root, run)
    audit(root, run, ["no_counterpart", "equivalent", "equivalent", "equivalent", "equivalent"])
    found = calibration(root, run)
    # every unit is verified: 6 truly absent units, of which the machine marked 5
    assert found.null_recall == pytest.approx(5 / 6)
    assert found.null_precision == pytest.approx(1.0)
    assert found.audit == {"neg:B:other": 40}, "a completed census of 5 units meets the floor of 40"
    assert cli(root, run, "evaluate") == 0
    gate = json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))
    assert not [r for r in gate["reasons"] if "audit stratum" in r]


def test_an_incomplete_census_still_fails_the_audit_floor(finished) -> None:
    root, run = finished
    verify_all_absent(root, run)
    audit(root, run, ["equivalent", "equivalent", "equivalent"])
    found = calibration(root, run)
    assert found.audit == {"neg:B:other": 3}
    # two negatives are imputed from x_h = 3 NONDEV: recall is 5/5 unless a draw makes one ABSENT
    assert found.null_recall is not None and 0.5 < found.null_recall < 1.0


def test_no_null_recall_without_phase_2_sample_verdicts(finished) -> None:
    root, run = finished
    assert calibration(root, run).null_recall is None


def test_strata_are_frozen_when_the_plan_is_drawn(finished, capsys) -> None:
    root, run = finished
    verify_all_absent(root, run)
    out = capsys.readouterr().out
    assert "WARNING topic labels are incomplete (0/10)" in out
    snapshot = json.loads((root / "data" / "annotations" / "verdicts" / "zh_T0892_song" / "strata_verify.json").read_text(encoding="utf-8"))
    assert snapshot["families"] == ["pos"] and snapshot["topics_complete"] is False
    assert len(snapshot["strata"]) == 10
    audit(root, run, ["equivalent", "equivalent", "equivalent"])      # two negatives stay unverified
    assert cli(root, run, "stats") == 0
    before = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))["E1_any"]
    assert before["not_estimable"] is None

    # topic labels imported after sampling: every unit becomes "sensitive"
    ctx = context(root, run, offline=True)
    texts = load_texts(ctx)
    write_labels(ann.topic_labels_path(ctx, texts.reference_id),
                 [TopicLabel(s.id, segment_fingerprint(s), frozenset({"sexual"}), coder="c", date="2026-10-04")
                  for s in texts.units])
    assert ann.load_topics(ctx, texts).complete
    assert cli(root, run, "build") == 0
    assert cli(root, run, "stats") == 0
    after = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))["E1_any"]
    # without the frozen strata the two unverified negatives move to neg:B:sensitive, which no
    # verdict calibrates (NOT_ESTIMABLE, or before that fix: imputed from the prior alone)
    assert after["not_estimable"] is None and after["point"] == before["point"]
    assert calibration(root, run).uncalibrated == {}


def test_plans_and_strata_are_committed_and_survive_a_new_run(finished, tmp_path) -> None:
    # Plans and strata snapshots lived only in the run directory: a new run lost them, so
    # G3 saw no plan and the frozen strata were gone.
    root, run = finished
    verify_all_absent(root, run)
    committed = root / "data" / "annotations" / "verdicts" / "zh_T0892_song"
    plan, strata = committed / "plan_verify.csv", committed / "strata_verify.json"
    assert plan.is_file() and strata.is_file() and not (run / "review" / "plan_verify.csv").exists()
    assert plan.read_text(encoding="utf-8").splitlines()[0] == "item_id,task,unit_id,stratum,inclusion_prob,priority"
    ctx = context(root, run, offline=True)
    texts = load_texts(ctx)
    assert {v.batch_id for v in ann.review_verdicts(ctx, texts)} == {"verify"}, "a plan is not a verdict file"

    run2 = tmp_path / "run2"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run2)]) == 0
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run2)]) == 0
    assert cli(root, run2, "sample", "verification") == 2, "the committed plan is drawn once for every run"
    assert calibration(root, run2).planned == calibration(root, run).planned == {"pos:absent:other": 5}

    # a run made before plans were committed: its plan and strata are read from the run dir
    (run / "review").mkdir(exist_ok=True)
    plan.rename(run / "review" / "plan_verify.csv")
    strata.rename(run / "review" / "strata_verify.json")
    assert calibration(root, run).planned == {"pos:absent:other": 5}
    assert calibration(root, run2).planned == {}
    frozen = json.loads((run / "review" / "strata_verify.json").read_text(encoding="utf-8"))["strata"]
    assert ann.sampling_strata(ctx, texts.witness_id, {}) == {u: s for u, s in frozen.items() if s.startswith("pos:")}
