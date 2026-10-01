"""Real texts through the CLI: ingest, baselines and a Claude-free evaluation into a
temporary run directory. Skipped unless HEVAJRA_RAW_DIR points at the fetched files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hevajra_matrix.cli import main
from hevajra_matrix.core.io import read_jsonl

pytestmark = pytest.mark.realdata
REPO = Path(__file__).resolve().parents[2]
INGEST_SENTINELS = {"S5a_note_vorlage_statement", "S5b_note_substitution_0589a09",
                    "S5c_note_substitution_0589a13", "S5d_note_title_0587c06", "S6_D418_colophon_is_paratext",
                    "S7_I7_melapaka_gap_shared"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import os

    raw = os.environ["HEVAJRA_RAW_DIR"]
    run = tmp_path_factory.mktemp("realdata") / "run"
    common = ["--root", str(REPO), "--run-dir", str(run)]
    assert main(["ingest", "--raw-dir", raw, *common]) == 0
    assert main(["baselines", *common]) == 0
    assert main(["evaluate", "--gold", "dev", "--baselines-only", *common]) == 0
    assert main(["report", *common]) == 0
    return run


def test_ingest_counts_and_g0(run_dir: Path) -> None:
    report = json.loads((run_dir / "ingest" / "report.json").read_text(encoding="utf-8"))
    zh, bo = report["zh_T0892_song"], report["bo_derge_D417_418"]
    assert (zh["notes"], zh["footnotes"], zh["unclassified_notes"], zh["duplicate_ids"]) == (521, 181, [], [])
    assert (bo["variants"], bo["paratext"], bo["content_segments"], bo["duplicate_ids"]) == (26, 3, 3042, [])
    g0 = json.loads((run_dir / "ingest" / "g0.json").read_text(encoding="utf-8"))
    assert g0 == {"passed": True, "reasons": []}
    gate = json.loads((run_dir / "evaluation" / "gate.json").read_text(encoding="utf-8"))
    assert gate["passed"]["G0"] is True and gate["level"] == 0, "no instrument yet: level 0"
    assert not [r for r in gate["reasons"] if r.startswith("G0")]


def test_ingest_stage_sentinels_pass(run_dir: Path) -> None:
    results = {r["sentinel_id"]: r for r in read_jsonl(run_dir / "ingest" / "sentinels.jsonl")}
    assert set(results) == INGEST_SENTINELS
    assert all(r["passed"] for r in results.values()), [r for r in results.values() if not r["passed"]]
    assert sum(1 for r in results.values() if r["blocking"]) == 5


def test_baselines_cover_every_reference_unit(run_dir: Path) -> None:
    for name in ("dp_zero", "dp_anchor"):
        links = read_jsonl(run_dir / "alignments" / f"{name}.jsonl")[1:]
        assert len({r["ref_id"] for r in links if r["ref_id"]}) == 3042, name


def test_level_0_report_on_the_real_texts(run_dir: Path) -> None:
    text = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Report level 0 (DESCRIPTIVE)" in text and "G0 passed" in text
    # B0 was 122 before the v0.3 review fixes to Tibetan syllables after a visarga and to
    # fused / multiplied Tibetan numerals (lengths and num: anchors changed).
    assert "| absent (no counterpart) | 1 | 120 |" in text, "P1 and B0 counts, side by side"
    assert "E3 decomposition of deviations (Vorlage, shared, residual): NOT_ESTIMABLE: G4: no manuscript column" \
        in text
    assert "| T0892:0592a27.n1 | vorlage_statement |" in text
