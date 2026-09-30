import csv
from pathlib import Path

from hevajra_matrix.ids import parse_unit_id, sort_key
from hevajra_matrix.registry import DATA_DIR
from hevajra_matrix.transfer import sinitic as S

CONFIG = DATA_DIR / "transfer" / "laozi"


def test_laozi_scheme_registered_and_sortable():
    u = parse_unit_id("L.81.p03")
    assert u.part == "L" and u.chapter == 81 and u.kind == "prose"
    assert sorted(["L.19.p01", "L.1.p02", "I.1.1"], key=sort_key) == ["I.1.1", "L.1.p02", "L.19.p01"]


def test_variants_normalise_taboo_and_phonetic_but_not_particles():
    vt = S.load_variants(CONFIG / "variants.yaml")
    assert vt.normalize("非恆道也") == "非常道也"
    assert vt.normalize("亡為") == "無為"
    assert vt.normalize("弗敢") == "弗敢"           # particle groups are opt-out
    assert vt.explained("無執故無失", "亡執故亡失") == ["無=亡", "無=亡"]


def test_parse_witness_file_orders_lacuna_and_half_chapters(fixtures: Path, tmp_path: Path):
    p = tmp_path / "w.txt"
    p.write_text("// c\n#64下 @簡10\n為之者敗之，□□□失之。\n#1?\n道可道也。\n", encoding="utf-8")
    segs = S.parse_witness_file(p, "w")
    assert [s.chapter for s in segs] == ["L.64", "L.64", "L.1"]
    assert segs[0].extra["native"] == "簡10" and segs[0].extra["half"] == "下"
    assert segs[1].kind == "lacuna" and segs[2].extra["uncertain"] is True
    assert S.chapter_order(segs) == [64, 1]


def test_order_deviation_detects_de_before_dao():
    od = S.order_deviation([64, 1, 19], [1, 19, 64])
    assert od["kendall_tau_distance"] > 0.6 and od["de_before_dao_share"] == 1.0
    assert S.order_deviation([19, 64], [1, 19, 64])["kendall_tau_distance"] == 0.0


def test_run_transfer_on_fixtures(fixtures: Path, tmp_path: Path):
    out = tmp_path / "laozi"
    m = S.run_transfer(CONFIG, fixtures / "laozi", out)
    assert m.reference == "zh_wangbi" and m.reference_grade == "coordinate_only"
    assert set(m.witnesses) == {"zh_heshanggong", "zh_mawangdui_B", "zh_guodian_A"}
    assert m.chapters[0] == "L.1" and len(m.chapters) == 81

    # anthology witness: chapters it never included are NA, not ABSENT
    assert m.get("L.1.p01", "zh_guodian_A").status == "NA"
    # 民之從事，常於幾成而敗之 has no counterpart in Guodian 64下 → ABSENT
    absent = [u for u in m.ordered_units() if m.get(u, "zh_guodian_A") and m.get(u, "zh_guodian_A").status == "ABSENT"]
    assert absent == ["L.64.p06"]
    # taboo variant 恆/常 is explained by normalisation: d_lex 0 for 非常道 ↔ 非恆道也 is not required, but raw > normalised
    rows = list(csv.DictReader((out / "alignment_review.csv").open(encoding="utf-8")))
    r = next(r for r in rows if r["witness"] == "zh_mawangdui_B" and r["ref_text"].startswith("非常道"))
    assert float(r["d_lex_raw"]) > float(r["d_lex"]) and "常=恆" in r["variants_explained"]
    # Guodian 19 reorders 絕巧棄利 / 盜賊亡有: repaired by the non-monotone pass with d_ord > 0
    c = m.get("L.19.p05", "zh_guodian_A")
    assert c.bead == "1:1*" and c.d_ord > 0 and c.similarity == 1.0
    # chapter order: Mawangdui puts 64 (德) before 1 (道)
    orders = {r["witness"]: r for r in csv.DictReader((out / "chapter_order.csv").open(encoding="utf-8"))}
    assert float(orders["zh_mawangdui_B"]["kendall_tau_distance"]) > 0 and float(orders["zh_guodian_A"]["kendall_tau_distance"]) == 0
    # heshanggong ≈ wangbi: the closest pair in the distance matrix
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "coordinate_only" in summary and "UPGMA" in summary
    for name in ("cells.csv", "wide_d_lex.csv", "heatmap_d_lex.svg", "reorder_repairs.csv", "manifest.json"):
        assert (out / name).exists()


def test_cli_transfer(fixtures: Path, tmp_path: Path):
    from hevajra_matrix.cli import main

    assert main(["transfer", "--config", str(CONFIG), "--texts", str(fixtures / "laozi"), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "cells.csv").exists()
