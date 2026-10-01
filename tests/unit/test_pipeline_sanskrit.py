"""Sanskrit reference mode, offline end to end on synthetic fixtures (data/fixtures/sa_mini/).

With ``data/reference/sa_snellgrove1959.tsv`` present, ingest makes the Sanskrit units the
reference and both the Derge and T0892 aligned witnesses; every per-witness stage runs for
each; E3 is computed from the manuscript readings with the other witness as co-witness.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.cli import main
from hevajra_matrix.core.io import read_jsonl
from hevajra_matrix.evaluation import gold as gold_sets
from hevajra_matrix.evaluation.windows import TestWindow
from hevajra_matrix.pipeline import StageError, perturb, plan_collation
from hevajra_matrix.pipeline import human_data as ann
from hevajra_matrix.pipeline import instrument
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.pipeline.store import read_alignment

from test_pipeline_support import context, echo_script, fill_all, make_root, make_sanskrit_root

SA, BO, ZH = "sa_snellgrove1959", "bo_derge_D417_418", "zh_T0892_song"


def cli(root: Path, run: Path, *args: str) -> int:
    return main([*args, "--root", str(root), "--run-dir", str(run)])


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("sanskrit")
    root, run = make_sanskrit_root(tmp), tmp / "run"
    assert cli(root, run, "ingest") == 0
    assert fill_all(context(root, run)) > 0
    assert cli(root, run, "run", "--offline") == 0
    return root, run


def details(run: Path, witness: str = ZH) -> dict:
    base = run if witness == ZH else run / "witnesses" / witness
    return json.loads((base / "stats" / "details.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- ingest and layout
def test_ingest_makes_the_sanskrit_units_the_reference(finished: tuple[Path, Path]) -> None:
    root, run = finished
    ctx = context(root, run)
    assert ctx.sanskrit_mode and ctx.reference == SA and ctx.witnesses == (ZH, BO)
    texts = load_texts(ctx)
    assert texts.reference_id == SA and texts.witness_id == ZH
    assert [u.id for u in texts.units] == ["I.1.p01", "I.1.1", "I.1.2", "I.1.3", "I.2.1", "I.2.2", "II.1.p01",
                                           "II.1.1", "II.1.2", "II.2.1"], "flagged units are no matrix rows"
    report = json.loads((run / "ingest" / "report.json").read_text(encoding="utf-8"))[SA]
    assert report["flagged"] == {"LACUNA": 1, "ABSENT": 1} and report["content_segments"] == 10
    bo = load_texts(ctx.for_witness(BO))
    assert bo.reference_id == SA and bo.witness_id == BO and bo.units == texts.units


def test_every_witness_has_its_own_outputs(finished: tuple[Path, Path]) -> None:
    _, run = finished
    for base, witness in ((run, ZH), (run / "witnesses" / BO, BO)):
        for rel in ("alignments/dp_zero.jsonl", "alignments/claude.jsonl", "collation/replicates.json",
                    "matrix/cells.jsonl", "evaluation/gate.json", "stats/estimates.json", "summary.md"):
            assert (base / rel).is_file(), (witness, rel)
        consensus = read_alignment(base / "alignments" / "claude.jsonl")
        assert (consensus.reference, consensus.witness) == (SA, witness)
        dp = read_alignment(base / "alignments" / "dp_anchor.jsonl")
        assert (dp.reference, dp.witness) == (SA, witness)
    assert not (run / "witnesses" / BO / "ingest").exists(), "the texts are shared"
    assert {r["task"] for r in read_jsonl(run / "llm_audit.jsonl")} == {"collate"}


def test_requests_pair_the_sanskrit_reference_with_each_witness(finished: tuple[Path, Path]) -> None:
    root, run = finished
    ctx = context(root, run)
    for witness, lang, core in ((BO, "Tibetan", {"D417:1", "D417:2"}), (ZH, "Chinese", {"pin1", "pin2"})):
        cplan = plan_collation(ctx.for_witness(witness))
        first = cplan.windows[0]
        assert (first.chapter, first.core_locals) == ("I.1", frozenset(core))
        _, _, request = cplan.requests()[0]
        assert "REFERENCE CHUNK (Sanskrit)" in request.body and f"WITNESS TEXT ({lang})" in request.context
        assert "D417:" not in request.body and "I.1.p01" not in request.body, "handles only, no coordinates"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert {"collate", "collate:sa-bo", "collate:sa-zh"} <= set(manifest["instrument_digests"])


def test_scope_lines_name_the_sanskrit_reference(finished: tuple[Path, Path]) -> None:
    _, run = finished
    for path in (run / "summary.md", run / "witnesses" / BO / "summary.md"):
        text = path.read_text(encoding="utf-8")
        assert f"- relative to the Sanskrit reference {SA} (a printed edition, not a manuscript)" in text
        assert f"- units = the units of {SA} (Snellgrove numbering)" in text
        assert "relative to Derge D417-418" not in text
    assert f"witness `{BO}`" in (run / "witnesses" / BO / "summary.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- E3
def test_e3_decomposes_the_chinese_with_the_derge_as_co_witness(finished: tuple[Path, Path]) -> None:
    _, run = finished
    d = details(run)["e3"]
    assert d["estimable"] and d["cowitness"] == BO and d["cowitness_revised"]
    assert d["manuscripts"] == ["C", "K", "Na"] and d["editions_ignored"] == ["sa_tripathi_negi2001"]
    assert d["by_unit"] == {"I.1.3": "shared_revised", "II.1.p01": "residual", "II.1.1": "vorlage_ms",
                            "II.1.2": "insufficient", "II.2.1": "residual"}
    assert sum(d["counts"].values()) == d["n_deviating"] == 5
    assert d["shared"]["observed"] == 1 and d["shared"]["excess"] == pytest.approx(1 - d["shared"]["expected"])
    estimates = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))
    assert estimates["E3"]["point"] == 5 and estimates["E3"]["not_estimable"] is None
    assert estimates["E3.phi_S"]["point"] == pytest.approx(d["shared"]["phi"])
    gate = json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))
    assert "E3" not in gate["not_estimable"], "G4 passes for E3: manuscripts and a distinct co-witness"


def test_editions_are_never_manuscripts(finished: tuple[Path, Path]) -> None:
    """II.2.1 is absent in an edition's reading only: it stays residual, not vorlage_ms, and
    the edition is no informative column (I.1.3 has two manuscripts, not three)."""
    _, run = finished
    d = details(run)["e3"]
    assert d["by_unit"]["II.2.1"] == "residual"
    assert "sa_tripathi_negi2001" not in d["manuscripts"]


def test_a2_bound_counts_unverified_shared_units_as_not_shared(finished: tuple[Path, Path]) -> None:
    _, run = finished
    d = details(run)["e3"]
    assert d["verified_deviating"] == 0
    assert d["bound"]["unverified_shared"] == 1 and d["bound"]["by_unit"]["I.1.3"] == "residual"
    assert d["bound"]["counts"]["shared_revised"] == 0 and d["bound"]["shared"]["observed"] == 0


def test_the_derge_column_is_decomposed_against_the_chinese(finished: tuple[Path, Path]) -> None:
    _, run = finished
    d = details(run, BO)["e3"]
    assert d["cowitness"] == ZH and not d["cowitness_revised"]
    assert d["by_unit"] == {"I.1.3": "shared"}, "the Chinese is not revised: plain 'shared'"


def test_witness_tree_over_reference_derge_and_chinese(finished: tuple[Path, Path]) -> None:
    _, run = finished
    tree = details(run)["distance"]
    assert tree["labels"] == sorted([SA, BO, ZH])
    assert tree["newick"].endswith(";") and all(len(row) == 3 for row in tree["matrix"])
    assert details(run, BO)["distance"] == tree


def test_verified_shared_units_survive_the_a2_bound(tmp_path: Path) -> None:
    """Gold that decides I.1.3 in both columns (grade A) keeps it shared in the bound."""
    root, run = make_sanskrit_root(tmp_path), tmp_path / "run"
    assert cli(root, run, "ingest") == 0
    ctx = context(root, run)
    unit = next(u for u in load_texts(ctx).units if u.id == "I.1.3")
    for witness in (ZH, BO):
        _write_gold(ann.gold_dir(ctx.for_witness(witness), witness), "dev",
                    [(unit.id, unit.fingerprint, "", "no_counterpart")])
    assert ann.gold_dir(ctx, ZH) == root / "data" / "annotations" / "by_reference" / SA / "gold" / ZH
    fill_all(ctx)
    assert cli(root, run, "run", "--offline", "--gold", "dev") == 0
    d = details(run)["e3"]
    assert d["verified_deviating"] == 1 and d["bound"]["unverified_shared"] == 0
    assert d["bound"]["by_unit"]["I.1.3"] == "shared_revised"


# --------------------------------------------------------------------------- not estimable
def test_without_readings_e3_is_not_estimable_for_want_of_a_manuscript(tmp_path: Path) -> None:
    root, run = make_sanskrit_root(tmp_path, readings=False), tmp_path / "run"
    assert cli(root, run, "ingest") == 0
    fill_all(context(root, run))
    assert cli(root, run, "run", "--offline") == 0
    estimates = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))
    assert estimates["E3"]["not_estimable"] == "G4: no manuscript column"
    assert "NOT_ESTIMABLE: G4: no manuscript column" in (run / "summary.md").read_text(encoding="utf-8")


def test_edition_readings_alone_are_no_manuscript_column(tmp_path: Path) -> None:
    root, run = make_sanskrit_root(tmp_path), tmp_path / "run"
    readings = root / "data" / "reference" / "readings"
    shutil.rmtree(readings)
    readings.mkdir()
    (readings / "I.1.tsv").write_text("unit_id\tms\tstatus\treading\tsource\tnote\n"
                                      "I.1.3\tsa_tripathi_negi2001\tabsent\t\t\t\n", encoding="utf-8")
    assert cli(root, run, "ingest") == 0
    fill_all(context(root, run))
    assert cli(root, run, "run", "--offline") == 0
    estimates = json.loads((run / "stats" / "estimates.json").read_text(encoding="utf-8"))
    assert estimates["E3"]["not_estimable"] == ("G4: no manuscript column (the readings of sa_tripathi_negi2001 "
                                                "are editions, never manuscripts)")


def test_an_unknown_manuscript_id_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, run = make_sanskrit_root(tmp_path), tmp_path / "run"
    with (root / "data" / "reference" / "readings" / "I.1.tsv").open("a", encoding="utf-8") as fh:
        fh.write("I.1.1\tXYZ\tpresent\t\t\t\n")
    assert cli(root, run, "ingest") == 0
    fill_all(context(root, run))
    capsys.readouterr()
    assert cli(root, run, "run", "--offline") == 2
    assert "['XYZ'] are not in registry" in capsys.readouterr().err


def test_derge_reference_with_derge_keyed_readings_is_descriptive_only(tmp_path: Path) -> None:
    """Under the provisional Derge reference the readings are used (Derge segment ids), but
    the reference would be its own co-witness: E3 stays NOT_ESTIMABLE with that reason."""
    root, run = make_root(tmp_path), tmp_path / "run"
    readings = root / "data" / "reference" / "readings"
    readings.mkdir(parents=True)
    (readings / "I.1.tsv").write_text("unit_id\tms\tstatus\treading\tsource\tnote\n"
                                      "D417:1b.2.3\tC\tpresent\t\t\t\nD417:1b.2.3\tK\tabsent\t\t\t\n",
                                      encoding="utf-8")
    assert cli(root, run, "ingest") == 0
    from test_pipeline_support import fill_cache
    fill_cache(context(root, run))
    assert cli(root, run, "run", "--offline") == 0
    d = details(run)["e3"]
    assert d["reason"] == "G4: reference is the co-witness" and not d["estimable"]
    assert d["manuscripts"] == ["C", "K"] and d["cowitness"] == BO
    assert d["by_unit"]["D417:1b.2.3"] == "vorlage_ms", "the Derge mantra unit deviates in the echo collation"
    assert set(d["by_unit"].values()) <= {"vorlage_ms", "insufficient"}, "no co-witness: never shared or residual"
    gate = json.loads((run / "evaluation" / "gate.json").read_text(encoding="utf-8"))
    assert gate["not_estimable"]["E3"] == "G4: reference is the co-witness"


# --------------------------------------------------------------------------- CLI, examples, ledger
def test_witness_option_runs_one_witness(finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    root, run = finished
    capsys.readouterr()
    assert cli(root, run, "report", "--witness", BO) == 0
    out = capsys.readouterr().out
    assert "witnesses/bo_derge_D417_418/summary.md" in out and "run/summary.md" not in out
    assert cli(root, run, "report", "--witness", "sa_tripathi_negi2001") == 2
    assert cli(root, run, "review", "status", "--witness", BO) == 0


def test_a_pair_without_examples_cannot_be_collated(tmp_path: Path) -> None:
    root, run = make_sanskrit_root(tmp_path), tmp_path / "run"
    (root / "data" / "codebook" / "collate_examples.sa-bo.yaml").unlink()
    assert cli(root, run, "ingest") == 0
    with pytest.raises(StageError, match=r"collate_examples\.sa-bo\.yaml is missing"):
        plan_collation(context(root, run).for_witness(BO))


def test_each_reference_and_witness_pair_has_its_own_ledger(finished: tuple[Path, Path]) -> None:
    root, run = finished
    ctx = context(root, run)
    ledger = root / "data" / "ledger"
    assert ann.ledger_path(ctx) == ledger / f"test_evaluations.{SA}.{ZH}.jsonl"
    assert ann.ledger_path(ctx.for_witness(BO)) == ledger / f"test_evaluations.{SA}.{BO}.jsonl"


# --------------------------------------------------------------------------- P-negation
def test_perturb_negation_reports_polarity_recall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev gold pairs the negated I.1.1 with a Chinese line made negative; P-negation strips
    the witness negator and scores whether the reader reports the polarity change."""
    root = make_sanskrit_root(tmp_path)
    raw = root / "data" / "raw" / "T18n0892.xml"
    raw.write_text(raw.read_text(encoding="utf-8").replace("\u5982\u662f\u6211\u805e", "\u4e0d\u662f\u6211\u805e"),
                   encoding="utf-8")
    run_yaml = root / "config" / "run.yaml"
    doc = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
    doc["windows"]["neighbours"] = 0          # so that the wrong-window perturbation has a far chapter
    run_yaml.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    run = tmp_path / "run"
    assert cli(root, run, "ingest") == 0
    ctx = context(root, run, offline=True)
    unit = next(u for u in load_texts(ctx).units if u.id == "I.1.1")
    _write_gold(ann.gold_dir(ctx, ZH), "dev", [(unit.id, unit.fingerprint, "T0892:0587c11.1", "equivalent")])
    fill_all(ctx)
    assert cli(root, run, "run", "--offline", "--gold", "dev") == 0

    def reader(request):        # flips polarity when the witness text holds no negator any more
        answer = echo_script(request)
        if "\u4e0d" not in (request.context or ""):
            for u in answer["units"]:
                u["polarity_flip"] = u["relation"] == "equivalent"
        return answer

    from hevajra_matrix.llm.fake import FakeClient
    monkeypatch.setattr(instrument, "make_client", lambda c: FakeClient(reader))
    out = perturb(ctx, negation=True)
    assert out["negation"] == {"hits": 1, "n": 1}
    assert any(key.endswith("-neg") for key in out["windows"])
    saved = json.loads((run / "evaluation" / "perturbations.json").read_text(encoding="utf-8"))
    assert saved["negation"] == {"hits": 1, "n": 1}
    scores = json.loads((run / "evaluation" / "scores.dev.json").read_text(encoding="utf-8"))
    reliability = scores["sources"]["claude:consensus"]["reliability"]
    assert reliability["n"] == 1 and set(reliability["table"]) == {"high", "medium", "low"}
    assert scores["sources"]["dp:zero"]["reliability"] is None, "a control gives no confidence"


def test_perturb_negation_needs_dev_gold(finished: tuple[Path, Path]) -> None:
    root, run = finished
    with pytest.raises(StageError, match="draws its pairs from dev gold"):
        instrument.negation_windows(context(root, run), load_texts(context(root, run)), [], None)  # type: ignore[arg-type]


def _write_gold(directory: Path, name: str, rows: list[tuple[str, str, str, str]]) -> None:
    """A one-window gold set: rows of (unit id, fingerprint, witness ids, relation)."""
    gold_rows = tuple(gold_sets.GoldRow(set_name=name, window_id="d1", row_type="ref", unit_id=u, fingerprint=fp,
                                        wit_ids=tuple(w.split()), relation=rel, annotator="test", date="2026-10-01")
                      for u, fp, w, rel in rows)
    gold_sets.save(gold_sets.GoldSet(name, directory.name, gold_rows), directory / f"{name}.csv")
    first, last = rows[0][0], rows[-1][0]
    gold_sets.write_windows([TestWindow("d1", first, last, len(rows), first.rsplit(".", 1)[0])],
                            directory / "windows.csv")
