import csv
import json
import shutil
from pathlib import Path

from hevajra_matrix.pipeline import run
from hevajra_matrix.registry import DATA_DIR

RAW_FILES = {"T18n0892.xml": "mini_cbeta.xml", "derge_rgyud_bum_nga.txt": "mini_derge.txt"}


def _stage(tmp_path: Path, fixtures: Path, with_sanskrit: bool) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    for dst, src in RAW_FILES.items():
        shutil.copy(fixtures / src, raw / dst)
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR / "anchors", data / "anchors")
    shutil.copytree(DATA_DIR / "concordance", data / "concordance")
    shutil.copy(DATA_DIR / "witnesses.yaml", data / "witnesses.yaml")
    if with_sanskrit:
        (data / "reference").mkdir()
        shutil.copy(fixtures / "sa_edition_a.tsv", data / "reference" / "sa_snellgrove1959.tsv")
        shutil.copy(fixtures / "sa_edition_b.tsv", data / "reference" / "sa_tripathi_negi2001.tsv")
    return raw, data, tmp_path / "out"


def test_pipeline_provisional_reference(tmp_path, fixtures):
    raw, data, out = _stage(tmp_path, fixtures, with_sanskrit=False)
    m = run(raw, out, data_dir=data)
    assert m.reference_grade == "provisional" and m.reference == "bo_derge_D417_418"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["zh_title_note"] == "大幻化普通儀軌三十一分中略出二無我法"
    assert (out / "cells.csv").exists() and (out / "summary.md").exists() and (out / "heatmap_coverage.svg").exists()
    rows = list(csv.DictReader((out / "cells.csv").open(encoding="utf-8")))
    assert rows and all(r["witness"] == "zh_T0892_song" for r in rows)
    assert "临时参照" in (out / "summary.md").read_text(encoding="utf-8")


def test_pipeline_gold_reference_with_two_editions(tmp_path, fixtures):
    raw, data, out = _stage(tmp_path, fixtures, with_sanskrit=True)
    m = run(raw, out, data_dir=data)
    assert m.reference_grade == "gold" and m.reference == "sa_snellgrove1959"
    assert set(m.witnesses) >= {"sa_tripathi_negi2001", "zh_T0892_song", "bo_derge_D417_418"}
    # second edition's flags become statuses
    assert m.get("I.1.2", "sa_tripathi_negi2001").status == "ABSENT"
    assert m.get("I.2.1", "sa_tripathi_negi2001").status == "LACUNA"
    # chapter II.1 is mapped to zh pin 11 in the concordance; the mini fixture has no pin 11 → UNALIGNED, not ABSENT
    assert m.get("II.1.1", "zh_T0892_song").status == "UNALIGNED"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    decs = {d["witness"]: d for d in manifest["decompositions"]}
    assert set(decs) == {"zh_T0892_song", "bo_derge_D417_418"}
    lengths = list(csv.DictReader((out / "structure_lengths.csv").open(encoding="utf-8")))
    assert any(r["merged_group"] == "I.11+II.1" for r in lengths)
    review = list(csv.DictReader((out / "alignment_review.csv").open(encoding="utf-8")))
    assert review and {"ref_units", "wit_text", "review_status"} <= set(review[0])


def test_pipeline_with_hybrid_similarity_toy_encoder(tmp_path, fixtures):
    raw, data, out = _stage(tmp_path, fixtures, with_sanskrit=True)

    def toy(texts):
        return [[float("金剛" in t or "vajra" in t or "རྡོ་རྗེ" in t), 1.0] for t in texts]

    m = run(raw, out, data_dir=data, similarity="hybrid", embedding_model="toy", encode=toy)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["similarity_backend"] == {"kind": "hybrid", "embedding_model": "toy"}
    assert m.cells


def test_llm_subcommands_dry_run_on_pipeline_output(tmp_path, fixtures):
    from hevajra_matrix.cli import main

    raw, data, out = _stage(tmp_path, fixtures, with_sanskrit=True)
    run(raw, out, data_dir=data)
    d = tmp_path / "derived"
    assert main(["llm-judge", "--llm", "mock", "--review", str(out / "alignment_review.csv"), "--out", str(d / "review_llm.csv"),
                 "--ref-lang", "sa", "--wit-lang", "zh"]) == 0
    assert main(["llm-extract", "--llm", "mock", "--cells", str(out / "cells.csv"), "--data", str(data),
                 "--witness", "zh_T0892_song", "--out", str(d / "components.jsonl"), "--limit", "3"]) == 0
    assert len((d / "components.jsonl").read_text(encoding="utf-8").splitlines()) <= 3
    assert main(["llm-attribute", "--llm", "mock", "--cells", str(out / "cells.csv"), "--witness", "zh_T0892_song",
                 "--out", str(d / "attr.csv")]) == 0
    assert main(["llm-probe", "--llm", "mock", "--passages", str(out / "cells.csv"), "--out", str(d / "probe.json"),
                 "--min-chars", "10", "--prefix-chars", "4"]) == 0
    probe = json.loads((d / "probe.json").read_text(encoding="utf-8"))
    assert probe["model_id"] == "mock" and probe["memorised_rate"] == 0.0
    items = d / "items.csv"
    items.write_text("unit,sa_witness,sa_text,zh_span,zh_text,cowitness_fact\nI.1.1,sa,x,0587c,y,the Tibetan also lacks this unit\n", encoding="utf-8")
    assert main(["llm-counterfactual", "--llm", "mock,mock", "--items", str(items), "--out", str(d / "cf.csv")]) == 0
    assert (d / "cf.csv").exists()
