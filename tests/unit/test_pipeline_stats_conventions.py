"""The stats stage reports E1_any_conv beside E1_any on a finished offline fixture run.

The fixture run uses collate.v1 (no convention flags), so E1_any_conv must equal E1_any
exactly. Adding convention flags to the built matrix then raises it by exactly the share of
verified NONDEV units that carry one; any_conv can never be the primary outcome.
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

from test_pipeline_calibration import audit, cli, verify_all_absent
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


def test_v1_run_gives_e1_any_conv_equal_to_e1_any_then_conventions_raise_it(finished) -> None:
    root, run = finished
    verify_all_absent(root, run)
    audit(root, run, ["equivalent", "equivalent", "equivalent"])
    assert cli(root, run, "stats") == 0
    est = estimates(run)
    a, b = est["E1_any"], est["E1_any_conv"]
    assert a["not_estimable"] is None
    assert (a["point"], a["lo"], a["hi"], a["n"], a["sources"]) == (b["point"], b["lo"], b["hi"], b["n"], b["sources"])
    assert "E1_any_conv_blind" not in est and "E1_any_conv_prior" not in est
    assert "E2_any_conv" not in est and set(details(run)["manski"]) == {"any", "cov", "any_conv"}
    assert details(run)["manski"]["any_conv"] == details(run)["manski"]["any"]
    assert details(run)["conventions"] == ZERO

    # the same matrix with two verified NONDEV units carrying a (human) convention
    ctx = context(root, run, offline=True)
    cells = read_built_cells(ctx)
    verified = [c.unit_id for c in cells if c.grade is Grade.A and c.status is Status.PRESENT
                and c.relation == "equivalent"][:2]
    assert len(verified) == 2
    flagged = [replace(c, flags=c.flags | {"transcribed_name"}) if c.unit_id in verified else c for c in cells]
    write_cells(ctx.path("matrix", "cells.jsonl"), flagged)
    est = results.stats(ctx)
    n = est["E1_any"].n
    assert est["E1_any"].point == pytest.approx(a["point"]), "D_any ignores conventions"
    assert est["E1_any_conv"].point == pytest.approx(a["point"] + 2 / n)
    assert est["E1_any_conv"].lo == pytest.approx(a["lo"] + 2 / n)
    assert details(run)["conventions"] == {**ZERO, "transcribed_name": 2, "human": 2}
    assert results.report_inputs(ctx).conventions == details(run)["conventions"]


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
