"""Stage outputs that must not be lost, merged or misread: experiment phases kept apart,
review sheets never silently overwritten, every export says what it wrote, the E4
diagnostics and the MDE on real labels, the level-0 heatmap and the credential check."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hevajra_matrix.cli import build_parser, main
from hevajra_matrix.core.io import read_jsonl, write_jsonl
from hevajra_matrix.core.types import Estimate
from hevajra_matrix.evaluation.gate import GateReport
from hevajra_matrix.experiments.overattribution.score import TrialOutcome, outcome_record
from hevajra_matrix.pipeline import e4 as e4_stage
from hevajra_matrix.pipeline import human_data as ann
from hevajra_matrix.pipeline import results
from hevajra_matrix.pipeline import review as review_stage
from hevajra_matrix.pipeline.context import has_api_key, load_texts
from hevajra_matrix.pipeline.measure import design_mde, read_built_cells
from hevajra_matrix.pipeline.store import gate_to_dict
from hevajra_matrix.report import markdown, svg
from hevajra_matrix.review import sampling, sheets
from hevajra_matrix.stats import twophase

from test_pipeline_calibration import audit, verify_all_absent
from test_pipeline_support import context, fill_cache, make_root

ITEMS = """item_id,phase,arm,pair_id,chapter,unit_ids,omission_origin,zh_context_from,zh_context_to,synthetic_facts,note
s1,main,sensitive,p1,I.1,D417:1b.2.1,constructed,T0892:0587c13.1,T0892:0587c13.3,true,
n1,main,neutral,p1,I.1,D417:1b.1.6,real,T0892:0587c11.1,T0892:0587c11.2,true,
s2,pilot,sensitive,p2,I.1,D417:1b.2.1,constructed,T0892:0587c13.1,T0892:0587c13.3,true,
n2,pilot,neutral,p2,I.1,D417:1b.1.6,real,T0892:0587c11.1,T0892:0587c11.2,true,
"""


def cli(root: Path, run: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--run-dir", str(run), "--offline"])


@pytest.fixture
def ingested(tmp_path: Path) -> tuple[Path, Path]:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


@pytest.fixture
def finished(ingested: tuple[Path, Path]) -> tuple[Path, Path]:
    root, run = ingested
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


# --------------------------------------------------------------------------- experiment phases
def _records(prefix: str) -> list[dict]:
    out = []
    for arm in ("sensitive", "neutral"):
        for pair in range(4):
            for condition in ("E0", "EP", "EW"):
                over = arm == "sensitive" and condition != "EW"
                out.append(outcome_record(TrialOutcome(
                    trial_id=f"{prefix}{arm[0]}{pair}:{condition}:r1", item_id=f"{prefix}{arm[0]}{pair}", arm=arm,
                    condition=condition, evidence="none", replicate="r1", omission_origin="real", status="ok",
                    scorer_status="ok", served_model="m", explanation="x", primary="source_text", stances={},
                    disputes_premise=False, y_over=over, y_any=over, y_uptake=not over, premise_ok=True)))
    return out


def test_pilot_and_main_experiment_outputs_are_kept_apart(ingested, capsys) -> None:
    root, run = ingested
    (root / "data" / "experiments" / "overattribution" / "items.csv").write_text(ITEMS, encoding="utf-8")
    assert cli(root, run, "experiment", "overattribution", "plan") == 0
    assert cli(root, run, "experiment", "overattribution", "plan", "--phase", "pilot") == 0
    main_trials = read_jsonl(run / "experiments" / "overattribution" / "trials.jsonl")
    pilot_trials = read_jsonl(run / "experiments" / "overattribution" / "pilot" / "trials.jsonl")
    assert {t["item_id"] for t in main_trials} == {"s1", "n1"} and {t["item_id"] for t in pilot_trials} == {"s2", "n2"}

    base = run / "experiments" / "overattribution"
    write_jsonl(base / "responses.jsonl", _records("m"))
    write_jsonl(base / "pilot" / "responses.jsonl", _records("p"))
    capsys.readouterr()
    assert cli(root, run, "experiment", "overattribution", "score", "--phase", "pilot") == 0
    out = capsys.readouterr().out
    assert "outcome basis unvalidated" in out and "(pilot, confirmatory test not interpreted)" in out
    assert not (base / "results.json").exists(), "a pilot score never writes the main results"
    assert json.loads((base / "pilot" / "results.json").read_text(encoding="utf-8"))["phase"] == "pilot"
    assert cli(root, run, "experiment", "overattribution", "score") == 0
    assert "H1 (confirmatory)" in capsys.readouterr().out
    main_results = json.loads((base / "results.json").read_text(encoding="utf-8"))
    assert main_results["phase"] == "main"
    assert len(read_jsonl(base / "responses.jsonl")) == 24, "the pilot never overwrote the main responses"


# --------------------------------------------------------------------------- review sheets
def test_review_exports_say_what_they_wrote_and_never_overwrite_a_sheet(finished, capsys) -> None:
    root, run = finished
    verify_all_absent(root, run)       # exports, fills and imports the blind and the reveal sheet
    sheet = run / "review" / "verify.blind.csv"
    sheet.write_text(sheet.read_text(encoding="utf-8-sig") + "reviewer work\n", encoding="utf-8-sig")
    capsys.readouterr()
    assert cli(root, run, "review", "export", "--task", "verify") == 2
    assert "may hold a reviewer's work" in capsys.readouterr().err
    assert "reviewer work" in sheet.read_text(encoding="utf-8-sig"), "the filled sheet survived"
    assert cli(root, run, "review", "export", "--task", "verify", "--force") == 0

    capsys.readouterr()
    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "verify") == 2, "already exported"
    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "verify", "--force") == 0
    assert re.search(r"review export reveal: 5 items -> .*verify\.reveal\.csv", capsys.readouterr().out)
    assert cli(root, run, "topics", "export") == 0
    assert re.search(r"review export topics: 10 units -> .*topics_first\.csv", capsys.readouterr().out)


def test_sheet_tasks_have_one_definition() -> None:
    assert review_stage.EXPORT_TASKS == sheets.SHEET_TASKS
    sub = {a.dest: a for a in build_parser()._subparsers._group_actions}["command"].choices["review"]
    actions = {a.dest: a for a in sub._subparsers._group_actions}["action"].choices
    task = {a.dest: a for a in actions["import"]._actions}["task"]
    assert tuple(task.choices) == review_stage.IMPORT_TASKS == tuple(t for t in sheets.SHEET_TASKS if t != "reveal")


# --------------------------------------------------------------------------- E4 diagnostics and the MDE
def test_tost_margin_is_the_preregistered_mde() -> None:
    assert e4_stage.tost_margin({"stats": {"mde": 0.07, "tost_margin": 0.07}}) == (0.07, None)
    margin, warning = e4_stage.tost_margin({"stats": {"mde": 0.07, "tost_margin": 0.10}})
    assert margin == 0.07 and "differs from stats.mde" in warning


def test_e4_diagnostics_and_the_mde_on_real_labels(finished) -> None:
    root, run = finished
    verify_all_absent(root, run)
    audit(root, run, ["no_counterpart", "equivalent", "equivalent", "equivalent", "equivalent"])
    ctx = context(root, run, offline=True)
    texts = load_texts(ctx)
    cells, machine = read_built_cells(ctx), read_built_cells(ctx, machine=True)
    verdicts = ann.current_verdicts(ann.review_verdicts(ctx, texts), texts)
    strata = sampling.machine_strata(machine, {}, texts.ref_kinds)
    draws = twophase.draws(cells, verdicts, strata, 20, 1, "final", texts.ref_kinds)
    groups = {u.id: ("sensitive", "neutral", "frame")[i % 3] for i, u in enumerate(texts.units)}
    estimate, d = e4_stage.contrast_with_diagnostics(texts, cells, machine, groups, verdicts, draws, "any", 1, 50,
                                                     0.1, "scope")
    assert {"permutation_p", "equivalent_within_margin", "margin", "overlap", "naive", "attenuation",
            "matched_rd", "misclassification", "negative_control"} <= set(d)
    assert d["margin"] == 0.1 and isinstance(d["matched_rd"]["pairs"], int)
    rows = d["misclassification"]
    assert rows and {r["factor"] for r in rows} == {"chapter", "tertile"}
    audited = [r for r in rows if r["stratum"] == "neg:B:other" and r["factor"] == "chapter"]
    assert sum(r["n"] for r in audited) == 5 and sum(r["errors"] for r in audited) == 1, "one verified miss"
    assert d["negative_control"]["name"] == "E4_frame_vs_neutral"

    power = e4_stage.mde_on_labels(texts, machine, groups, "any", 1, n_sim=20)
    assert power["units"] == sum(1 for g in groups.values() if g in ("sensitive", "neutral"))
    assert power["power_curve"] and power["mde_on_labels"] is None, "ten units cannot detect any effect"
    (run / "stats").mkdir(exist_ok=True)
    (run / "stats" / "power.json").write_text(json.dumps(power), encoding="utf-8")
    assert design_mde(ctx) is None, "G4 then reports the MDE as not reached"
    (run / "stats" / "power.json").write_text(json.dumps({**power, "mde_on_labels": 0.14}), encoding="utf-8")
    assert design_mde(ctx) == 0.14, "the larger of the preregistered and the simulated MDE"


def test_e4_names_the_uncalibrated_stratum_rather_than_missing_verdicts(finished) -> None:
    # Verification verdicts exist but the audit stratum has no phase-2 sample: E1 and E4 are
    # NOT_ESTIMABLE for that reason; E4 used to say "no phase-2 verification or audit verdict".
    root, run = finished
    verify_all_absent(root, run)
    assert cli(root, run, "build") == 0
    (run / "evaluation" / "gate.json").write_text(json.dumps(gate_to_dict(_level2())), encoding="utf-8")
    est = results.stats(context(root, run, offline=True))
    assert est["E1_any"].not_estimable.startswith("G3: no phase-2 sample verdict in stratum neg:B:other")
    assert est["E4"].not_estimable == est["E1_any"].not_estimable


def _level2() -> GateReport:
    return GateReport(level=2, confirmatory=True, passed={"G0": True, "G1": True, "G2": True, "G3": True},
                      reasons=())


def test_report_prints_revision_rate_and_e4_diagnostics_only_beside_numbers() -> None:
    est = {name: Estimate(name, 0.2, 0.1, 0.3, 10, "s", ("x",)) for name in ("E1_any", "E1_any_blind", "E4")}
    details = {"margin": 0.1, "equivalent_within_margin": False, "permutation_p": 0.4,
               "overlap": {"n_strata": 4, "n_overlap_strata": 3, "n_units": 20, "n_used": 18, "dropped": ["a", "b"]},
               "naive": 0.05, "attenuation": 0.15, "matched_rd": {"difference": 0.1, "pairs": 6},
               "negative_control": {"name": "E4_frame_vs_neutral", "point": 0.01, "lo": -0.1, "hi": 0.1,
                                    "not_estimable": None},
               "misclassification": [{"stratum": "neg:B:other", "factor": "chapter", "level": "I.1", "n": 5,
                                      "errors": 1, "rate": 0.2}]}
    inputs = markdown.ReportInputs(run_id="r", reference="bo", witness="zh", scope=("s",), estimates=est,
                                   revision={"pos:absent:other": {"n": 10, "outcome_changed": 3}}, e4_details=details)
    text = markdown.render(inputs, _level2())
    assert "revised at reveal (outcome class changed): 3 of 10 verdicts (0.300)" in text
    assert "3 of 4 chapter x length strata hold both exposures; 18 of 20 units used, 2 dropped" in text
    assert "naive Delta on machine labels 0.050; corrected minus naive 0.150" in text
    assert "caliper-matched difference 0.100 (6 pairs)" in text and "frame vs neutral: 0.010" in text
    assert "| neg:B:other | chapter | I.1 | 5 | 1 | 0.200 |" in text
    level0 = markdown.render(inputs, GateReport(level=0, confirmatory=False, passed={"G1": False},
                                                reasons=("G1: failed",)))
    assert "naive Delta" not in level0 and "revised at reveal" not in level0


def test_level_0_heatmap_does_not_shade_by_share(finished) -> None:
    root, run = finished
    cells = read_built_cells(context(root, run, offline=True))
    svg.chapter_heatmap(cells, run / "h1.svg", "level 1")

    def opacities(name: str) -> set[str]:
        return set(re.findall(r'fill-opacity="([0-9.]+)"', (run / name).read_text(encoding="utf-8")))

    assert opacities("chapter_heatmap.svg") <= {"0.00", "0.25"}, "the level-0 report shades uniformly"
    assert len(opacities("h1.svg")) > 2, "from level 1 the shading shows each status's share"


# --------------------------------------------------------------------------- credentials
def test_an_auth_token_counts_as_credentials(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not has_api_key()
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    assert has_api_key()
