"""The stats stage reports E1_any_conv beside E1_any on finished offline fixture runs.

E1_any_conv imputes conventions from the human verdicts only (``stats.anyconv``): with no
human convention it equals E1_any exactly, on a collate.v1 run and on a collate.v2 run whose
machine conventions no verdict confirms; human conventions raise it at least by the share of
verified NONDEV units carrying one. Manski any_conv exists only for a collate.v2 run, and
any_conv can never be the primary outcome.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.core.types import Grade, Status
from hevajra_matrix.pipeline import results
from hevajra_matrix.pipeline.context import StageError
from hevajra_matrix.pipeline.measure import primary_outcome, read_built_cells
from hevajra_matrix.pipeline.store import write_cells

from hevajra_matrix.llm.cache import CachedClient
from hevajra_matrix.llm.fake import FakeClient
from hevajra_matrix.pipeline import plan_collation
from hevajra_matrix.pipeline.instrument import collated_prompt
from hevajra_matrix.review.verdicts import read_file

from test_pipeline_calibration import audit, cli, rewrite, verify_all_absent
from test_pipeline_collate_v2 import select_prompt
from test_pipeline_evaluate_conventions import echo_names
from test_pipeline_support import context, fill_cache, make_root

ZERO = {"standing_rendering": 0, "transcribed_name": 0, "unsupported": 0, "human": 0}


@pytest.fixture
def finished(tmp_path: Path) -> tuple[Path, Path]:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    fill_cache(context(root, run))
    assert main(["run", "--offline", "--root", str(root), "--run-dir", str(run)]) == 0
    return root, run


def estimates(run: Path) -> dict:
    return json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))


def details(run: Path) -> dict:
    return json.loads((run / "stats" / "details.json").read_text(encoding="utf-8"))


def test_without_phase_2_e1_any_conv_is_not_estimable_like_e1_any(finished) -> None:
    _, run = finished
    est = estimates(run)
    assert est["E1_any_conv"]["not_estimable"] == est["E1_any"]["not_estimable"] == results.NO_PHASE2
    assert "E1_any_conv_blind" not in est and "E1_any_conv_prior" not in est


def sample_verdicts(root: Path) -> list:
    files = sorted(f for f in (root / "data" / "annotations" / "verdicts").rglob("*.csv")
                   if not f.name.startswith("plan_"))
    return [v for f in files for v in read_file(f) if v.task in ("verify", "audit")]


def same(a: dict, b: dict) -> bool:
    return (a["point"], a["lo"], a["hi"], a["n"], a["sources"]) == (b["point"], b["lo"], b["hi"], b["n"], b["sources"])


def test_v1_run_without_human_conventions_gives_e1_any_conv_equal_to_e1_any(finished) -> None:
    root, run = finished
    verify_all_absent(root, run)
    audit(root, run, ["equivalent", "equivalent", "equivalent"])
    assert cli(root, run, "stats") == 0
    est = estimates(run)
    assert est["E1_any"]["not_estimable"] is None and same(est["E1_any"], est["E1_any_conv"])
    assert "E1_any_conv_blind" not in est and "E1_any_conv_prior" not in est and "E2_any_conv" not in est
    assert collated_prompt(context(root, run, offline=True)) == "collate.v1"
    assert set(details(run)["manski"]) == {"any", "cov"}, "collate.v1 records no conventions: no Manski any_conv"
    n = len(sample_verdicts(root))
    assert details(run)["any_conv"] == {"verdicts": n, "unannotated": 0, "with_convention": 0,
                                        "nondev_with_convention": 0, "verified_with_convention": 0}
    assert details(run)["conventions"] == ZERO


def test_v1_run_imputes_human_conventions_and_reports_the_unannotated_verdicts(finished) -> None:
    root, run = finished
    verify_all_absent(root, run)
    # two audited negatives verified as transcribed names: the sheet's convention column
    assert cli(root, run, "sample", "audit") == 0
    assert cli(root, run, "review", "export", "--task", "audit") == 0
    sheet = run / "review" / "audit.blind.csv"

    def decide(rows):
        return [{**r, "blind_relation": "equivalent", "blind_wit_loci": r["zh_context_loci"].split()[0],
                 "blind_convention": "transcribed_name" if i < 2 else ""} for i, r in enumerate(rows[:3])]

    assert rewrite(sheet, decide) == 3
    assert cli(root, run, "review", "import", "--task", "audit", "--file", str(sheet), "--annotator", "ann",
               "--date", "2026-10-02") == 0
    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "audit") == 0
    assert cli(root, run, "review", "reveal", "--file", str(run / "review" / "audit.reveal.csv"),
               "--date", "2026-10-03") == 0
    # a verdict file written before the convention columns: the verify verdicts lose them ("")
    for path in (root / "data" / "annotations" / "verdicts").rglob("verify.csv"):
        rewrite(path, lambda rows: [{**r, "machine_convention": "", "blind_convention": "", "final_convention": ""}
                                    for r in rows])
    assert cli(root, run, "build") == 0
    assert cli(root, run, "stats") == 0
    est = estimates(run)
    a, b = est["E1_any"], est["E1_any_conv"]
    n = a["n"]
    # each draw: D_any_conv >= D_any + the two verified NONDEV transcribed names
    assert b["point"] >= a["point"] + 2 / n - 1e-12 and b["lo"] >= a["lo"] + 2 / n - 1e-12
    assert set(details(run)["manski"]) == {"any", "cov"}
    verify = [v for v in sample_verdicts(root) if v.task == "verify"]
    assert verify and all(v.final_convention == "" for v in verify)
    assert details(run)["any_conv"] == {"verdicts": len(sample_verdicts(root)), "unannotated": len(verify),
                                        "with_convention": 2, "nondev_with_convention": 2,
                                        "verified_with_convention": 2}
    assert details(run)["conventions"] == {**ZERO, "transcribed_name": 2, "human": 2}
    ctx = context(root, run, offline=True)
    inputs = results.report_inputs(ctx)
    assert inputs.conventions == details(run)["conventions"] and inputs.any_conv == details(run)["any_conv"]


def test_v2_run_calibrates_the_machine_conventions_and_has_manski_any_conv(tmp_path: Path) -> None:
    root, run = make_root(tmp_path), tmp_path / "run"
    assert main(["ingest", "--root", str(root), "--run-dir", str(run)]) == 0
    select_prompt(root, "collate.v2")
    ctx = context(root, run)
    cache = CachedClient(FakeClient(echo_names), ctx.settings.path("cache"))
    for _, _, request in plan_collation(ctx).requests():
        cache.complete(request)
    assert cli(root, run, "run") == 0
    machine = [c for c in read_built_cells(context(root, run, offline=True)) if "transcribed_name" in c.flags]
    assert machine, "echo_names gives every second linked unit a transcribed name"
    verify_all_absent(root, run)
    audit(root, run, ["equivalent", "equivalent", "equivalent"])     # the humans record no convention
    assert cli(root, run, "stats") == 0
    est = estimates(run)
    assert est["E1_any"]["not_estimable"] is None and same(est["E1_any"], est["E1_any_conv"])
    manski = details(run)["manski"]
    assert set(manski) == {"any", "cov", "any_conv"}, "collate.v2: Manski any_conv reads the built cells"
    assert manski["any_conv"][0] >= manski["any"][0]


def test_any_conv_is_never_the_primary_outcome() -> None:
    def ctx(prereg):
        return SimpleNamespace(settings=SimpleNamespace(prereg=prereg))

    assert primary_outcome(ctx({})) == "any" and primary_outcome(ctx({"primary_outcome": "cov"})) == "cov"
    for bad in ("any_conv", "absent"):
        with pytest.raises(StageError, match="primary_outcome"):
            primary_outcome(ctx({"primary_outcome": bad}))


def test_stats_refuses_a_preregistration_naming_any_conv_primary(finished) -> None:
    root, run = finished
    ctx = context(root, run, offline=True)
    ctx.settings.prereg["primary_outcome"] = "any_conv"
    with pytest.raises(StageError, match="primary_outcome 'any_conv'"):
        results.stats(ctx)
