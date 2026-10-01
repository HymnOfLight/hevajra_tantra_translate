"""Gold format, windows and scoring (evaluation.gold, evaluation.windows)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hevajra_matrix.collate.verify import FLAG_HINT, FLAG_UNKNOWN_HANDLE
from hevajra_matrix.config import ConfigError
from hevajra_matrix.core.types import Alignment, Diagnostic, Link, Relation, Segment, WitnessOnlyKind
from hevajra_matrix.evaluation import gold as G
from hevajra_matrix.evaluation.windows import TestWindow

REF, WIT = "bo_ref", "zh_wit"
EQ, AB, NC = Relation.EQUIVALENT, Relation.ABRIDGED, Relation.NO_COUNTERPART


def row(unit, rel, wit=(), window="w01", **kw) -> G.GoldRow:
    return G.GoldRow(set_name="test", window_id=window, row_type="ref", unit_id=unit, fingerprint="f" + unit[-1],
                     wit_ids=tuple(wit), relation=rel, annotator="ab", date="2026-10-01", **kw)


def only(seg, kind, window="w01") -> G.GoldRow:
    return G.GoldRow(set_name="test", window_id=window, row_type="witness_only", unit_id=f"+{seg}", fingerprint="x",
                     wit_ids=(seg,), witness_only_kind=kind, annotator="ab", date="2026-10-01")


# Hand case (see test_score_by_hand): 5 units, 2 witness-only segments, one excluded unit.
ROWS = (
    row("D:1a.1.1", "equivalent", ["T:1.1"]),
    row("D:1a.2.1", "equivalent", ["T:2.1", "T:3.1"]),
    row("D:1a.3.1", "no_counterpart"),
    row("D:1a.4.1", "abridged", ["T:5.1"], window="w02"),
    row("D:1a.5.1", "no_counterpart", window="w02"),
    row("D:1a.6.1", "lacuna", window="w02"),
    only("T:9.1", "addition"),
    only("T:10.1", "translator_note", window="w02"),
)
WINDOWS = (TestWindow("w01", "D:1a.1.1", "D:1a.3.1", 3, "I.1", 5, "2026-10-01"),
           TestWindow("w02", "D:1a.4.1", "D:1a.6.1", 3, "I.2", 5, "2026-10-01"))
GOLD = G.GoldSet("test", WIT, ROWS, WINDOWS, REF)
KINDS = {f"D:1a.{i}.1": "verse_line" for i in range(1, 7)}

PRED = Alignment("dp:zero", REF, WIT, (
    Link("D:1a.1.1", ("T:1.1",), EQ),
    Link("D:1a.2.1", ("T:2.1",), AB),
    Link("D:1a.3.1", ("T:4.1",), EQ),
    Link("D:1a.5.1", (), NC),
    Link(None, ("T:9.1",), WitnessOnlyKind.ADDITION),
))


# --------------------------------------------------------------------------- format
def test_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / WIT / "test.csv"
    G.save(GOLD, path)
    G.write_windows(WINDOWS, path.with_name("windows.csv"))
    back = G.load(path, reference=REF)
    assert back == GOLD
    assert back.source == "gold:test" and back.alignment.source == "gold:test"
    assert back.units() == tuple(f"D:1a.{i}.1" for i in range(1, 6))          # the lacuna row is not scored
    assert [link.ref_id for link in back.alignment.links if link.ref_id] == list(back.units())
    assert len(back.alignment.witness_only()) == 2


def test_columns_exactly_as_specified():
    assert G.GOLD_COLUMNS == ("set", "window_id", "row_type", "unit_id", "fingerprint", "wit_ids", "relation",
                              "polarity_flip", "flags", "witness_only_kind", "annotator", "date", "minutes")


@pytest.mark.parametrize("bad, message", [
    (row("D:1.1", "equivalent"), "needs witness segment ids"),
    (row("D:1.1", "no_counterpart", ["T:1.1"]), "names no witness segment"),
    (row("D:1.1", "shortened", ["T:1.1"]), "is not one of"),
    (row("D:1.1", "equivalent", ["T:1.1"], flags=frozenset({"typo"})), "unknown flag"),
    (dataclasses.replace(only("T:1.1", "addition"), unit_id="+T:2.1"), "unit_id is"),
    (dataclasses.replace(only("T:1.1", "addition"), witness_only_kind="gloss"), "witness_only_kind"),
    (dataclasses.replace(row("D:1.1", "equivalent", ["T:1.1"]), row_type="unit"), "row_type"),
])
def test_invalid_rows_are_refused(tmp_path: Path, bad, message):
    path = tmp_path / WIT / "test.csv"
    G.save(G.GoldSet("test", WIT, (bad,)), path)
    G.write_windows(WINDOWS, path.with_name("windows.csv"))
    with pytest.raises(G.GoldError, match=message):
        G.load(path)


def test_set_level_checks(tmp_path: Path):
    path = tmp_path / WIT / "test.csv"
    G.write_windows(WINDOWS, path.with_name("windows.csv"))
    G.save(G.GoldSet("test", WIT, (row("D:1.1", "equivalent", ["T:1.1"]), only("T:1.1", "addition"))), path)
    with pytest.raises(G.GoldError, match="both linked and witness-only"):
        G.load(path)
    G.save(G.GoldSet("test", WIT, (row("D:1.1", "no_counterpart"), row("D:1.1", "no_counterpart"))), path)
    with pytest.raises(G.GoldError, match="more than once"):
        G.load(path)
    G.save(G.GoldSet("test", WIT, (row("D:1.1", "no_counterpart", window="w09"),)), path)
    with pytest.raises(G.GoldError, match="not in windows.csv"):
        G.load(path)
    with pytest.raises(G.GoldError, match="gold set must be"):
        G.load(tmp_path / WIT / "other.csv")
    path.with_name("windows.csv").unlink()
    with pytest.raises(G.GoldError, match="windows.csv is missing"):
        G.load(path)


def test_verdict_roundtrip():
    verdicts = G.to_verdicts(GOLD)
    assert {v.batch_id for v in verdicts} == {"test_w01", "test_w02"}
    assert all(v.task == "gold" for v in verdicts)
    back = G.from_verdicts(verdicts, "test", WIT, REF)
    assert back.rows == GOLD.rows
    with pytest.raises(G.GoldError, match="batch dev_"):
        G.from_verdicts(verdicts, "dev", WIT)


# --------------------------------------------------------------------------- scoring
def test_score_by_hand():
    s = G.score(PRED, GOLD, ref_kinds=KINDS)
    m = s.metrics()
    # links: tp = 2 (T:1.1, T:2.1); predicted 3; gold 4
    assert m["link_precision"] == pytest.approx(2 / 3)
    assert m["link_recall"] == pytest.approx(2 / 4)
    assert m["link_f1"] == pytest.approx(4 / 7)
    # NULL: predicted {1a.5}; gold {1a.3, 1a.5}
    assert m["null_precision"] == 1.0 and m["null_recall"] == 0.5
    # witness-only: T:9.1 found with the right kind, T:10.1 missed
    assert m["witness_only_recall"] == 0.5 and m["witness_only_kind_agreement"] == 1.0
    # status: (P,P) (P,PA) (A,P) (PA,UNALIGNED) (A,A): observed 0.4, chance 7/25
    assert m["status_kappa"] == pytest.approx((0.4 - 7 / 25) / (1 - 7 / 25))
    assert m["status_agreement:PRESENT"] == pytest.approx(2 * 1 / (2 + 2))
    assert m["status_agreement:ABSENT"] == pytest.approx(2 * 1 / (2 + 1))
    assert m["status_agreement:PARTIAL"] == 0.0
    # D_any: gold (n, n, d, d, d); pred (n, d, n, unresolved, d): observed 2/5; chance (2*2 + 3*2)/25
    assert m["dany_kappa"] == pytest.approx((0.4 - 10 / 25) / (1 - 10 / 25))
    assert m["relation_recall:equivalent"] == 0.5 and m["relation_recall:no_counterpart"] == 0.5
    assert m["relation_recall:abridged"] == 0.0
    assert s.excluded == ("D:1a.6.1",)
    assert s.counts() == {"units": 5, "windows": 2, "excluded": 1, "gold_links": 4, "gold_null": 2,
                          "gold_witness_only": 2, "unresolved": 1}
    assert s.confusion()[("PARTIAL", "UNALIGNED")] == 1


def test_gold_scores_perfectly_against_itself():
    s = G.score(GOLD.alignment, GOLD, ref_kinds=KINDS)
    m = s.metrics()
    for name in ("link_precision", "link_recall", "link_f1", "null_precision", "null_recall",
                 "witness_only_recall", "status_kappa", "dany_kappa"):
        assert m[name] == 1.0, name
    assert m["refusal_rate"] == 0.0 and m["quote_failure_rate"] == 0.0


def test_transliterated_mantra_is_no_deviation():
    gold = G.GoldSet("test", WIT, (row("D:1a.1.1", "transliterated", ["T:1.1"]),), WINDOWS[:1])
    pred = Alignment("x", REF, WIT, (Link("D:1a.1.1", ("T:1.1",), Relation.TRANSLITERATED),))
    (mantra,) = G.score(pred, gold, ref_kinds={"D:1a.1.1": "mantra"}).units
    (prose,) = G.score(pred, gold, ref_kinds={"D:1a.1.1": "prose"}).units
    assert (mantra.gold_dev, prose.gold_dev) == (False, True)
    with pytest.raises(ValueError, match="ref_kinds"):
        G.score(pred, gold, ref_kinds={})


def test_refusals_quote_failures_and_invalid_handles():
    pred = Alignment("claude:consensus", REF, WIT, (
        Link("D:1a.1.1", ("T:1.1",), EQ, flags=frozenset({FLAG_UNKNOWN_HANDLE})),
        Link("D:1a.2.1", ("T:2.1", "T:3.1"), EQ),
    ))
    diagnostics = [Diagnostic("verification_failed", ("D:1a.3.1",), detail="I.1#1 D:1a.3.1: V4 ref_quote is not"),
                   Diagnostic("verification_failed", ("D:1a.2.1",), detail="I.1#1: V5 wit_quote is required")]
    s = G.score(pred, GOLD, ref_kinds=KINDS, diagnostics=diagnostics,
                unresolved={"D:1a.3.1": "refused:bio", "D:1a.4.1": "truncated"},
                topic_groups={"D:1a.3.1": "sensitive", "D:1a.4.1": "sensitive", "D:1a.1.1": "neutral"})
    m = s.metrics()
    assert m["refusal_rate"] == pytest.approx(1 / 5)
    assert m["refusal_rate:sensitive"] == 0.5 and m["refusal_rate:neutral"] == 0.0
    assert m["quote_failure_rate"] == pytest.approx(1 / 5)
    assert m["invalid_handle_rate"] == pytest.approx(1 / 5)
    assert {u.unit_id: u.reason for u in s.units if u.pred_relation is None}["D:1a.5.1"] == "unassessed"


def test_substituted_model_hints_are_never_scored():
    pred = Alignment("x", REF, WIT, (Link("D:1a.1.1", ("T:1.1",), EQ, flags=frozenset({FLAG_HINT})),))
    with pytest.raises(ValueError, match="never scored"):
        G.score(pred, GOLD, ref_kinds=KINDS)


def test_units_restriction_and_windows():
    s = G.score(PRED, GOLD, ["D:1a.1.1", "D:1a.2.1"], ref_kinds=KINDS)
    assert s.windows() == ("w01",) and len(s.witness_only) == 1
    with pytest.raises(ValueError, match="not scored"):
        G.score(PRED, GOLD, ["D:1a.6.1"], ref_kinds=KINDS)
    full = G.score(PRED, GOLD, ref_kinds=KINDS)
    doubled = full.on_windows(["w01", "w01"])
    assert len(doubled.units) == 6 and G.link_f1(doubled) == pytest.approx(G.link_f1(full.on_windows(["w01"])))


def test_interval_estimates_and_mcnemar():
    s = G.score(PRED, GOLD, ref_kinds=KINDS)
    est = G.interval_estimates(s, n_boot=200, seed=1, names=("link_f1", "status_kappa"))
    assert est["link_f1"].point == pytest.approx(4 / 7) and est["link_f1"].lo <= est["link_f1"].hi
    assert est["link_f1"].scope == "gold:test" and est["link_f1"].sources == ("dp:zero",)
    perfect = G.score(GOLD.alignment, GOLD, ref_kinds=KINDS)
    result = G.mcnemar(G.status_correct(perfect), G.status_correct(s))
    assert (result.only_a, result.only_b) == (3, 0)


def test_human_kappa():
    second_rows = tuple(dataclasses.replace(r, set_name="test_second") for r in ROWS[:3])
    second = G.GoldSet("test_second", WIT, second_rows, WINDOWS[:1])
    assert G.human_kappa(GOLD, second, KINDS) == 1.0
    changed = (dataclasses.replace(second_rows[0], relation="no_counterpart", wit_ids=()),) + second_rows[1:]
    assert G.human_kappa(GOLD, dataclasses.replace(second, rows=changed), KINDS) < 1.0


# --------------------------------------------------------------------------- windows
def seg(uid: str, chapter: str, start: str) -> Segment:
    return Segment(id=uid, witness=REF, lang="bo", text="x", start=start, end=start, kind="verse_line",
                   chapter=chapter)


def test_dev_region_units():
    segs = ([seg(f"D417:1a.{i}.1", "I.1", f"1a.{i}") for i in range(1, 6)]
            + [seg(f"D418:2a.{i}.1", "II.3", f"2a.{i}") for i in range(1, 10)]
            + [seg(f"D418:3a.{i}.1", "II.11", f"3a.{i}") for i in range(1, 5)]
            + [seg(f"D418:4a.{i}.1", "II.12", f"4a.{i}") for i in range(1, 3)])
    regions = [{"chapter": "I.1"}, {"chapter": "II.3", "around": "D418:2a.5", "radius_units": 2},
               {"chapter": "I.1", "first_units": 2}, {"chapter": "II.11", "last_units": 2, "id": "II.11-12"},
               {"chapter": "II.12", "id": "II.11-12"}]
    out = G.dev_region_units(segs, regions)
    assert out["I.1"] == tuple(f"D417:1a.{i}.1" for i in range(1, 6))
    assert out["II.3-around-D418-2a.5"] == tuple(f"D418:2a.{i}.1" for i in range(3, 8))
    assert out["I.1-first2"] == ("D417:1a.1.1", "D417:1a.2.1")
    assert out["II.11-12"] == ("D418:3a.3.1", "D418:3a.4.1", "D418:4a.1.1", "D418:4a.2.1")
    assert G.dev_region_units(segs, [{"chapter": "II.12", "last_units": 5}]) == {
        "II.12-last5": ("D418:4a.1.1", "D418:4a.2.1")}                  # longer than the chapter: all of it
    window = G.region_window("II.11-12", out["II.11-12"], {s.id: s.chapter for s in segs})
    assert (window.chapter, window.n_units, window.seed) == ("II.11+II.12", 4, None)
    for bad in ({"chapter": "I.9"}, {"chapter": "I.1", "radius": 3}, {"chapter": "I.1", "first_units": 2,
                                                                        "last_units": 1},
                {"chapter": "II.3", "around": "D418:2a.5"}, {"chapter": "I.1", "first_units": -1}):
        with pytest.raises(ConfigError):
            G.dev_region_units(segs, [bad])
    with pytest.raises(ValueError, match="starts on"):
        G.dev_region_units(segs, [{"chapter": "II.3", "around": "D418:9b.9", "radius_units": 1}])


def test_draw_test_windows():
    units = {"A": [f"a{i}" for i in range(40)], "B": [f"b{i}" for i in range(12)], "C": [f"c{i}" for i in range(4)]}
    exclude = {f"a{i}" for i in range(10, 15)}
    windows = G.draw_test_windows(units, n=4, width=5, exclude=exclude, seed=11, drawn_at="2026-10-01")
    assert windows == G.draw_test_windows(units, 4, 5, exclude, 11, "2026-10-01")
    assert [w.window_id for w in windows] == ["w01", "w02", "w03", "w04"]
    covered = [u for w in windows for u in G.window_units(w, units)]
    assert len(covered) == len(set(covered)) == 20                     # no overlap
    assert not set(covered) & exclude and not any(u.startswith("c") for u in covered)
    assert all(w.seed == 11 and w.n_units == 5 for w in windows)
    with pytest.raises(ValueError, match="frame holds only"):
        G.draw_test_windows(units, n=20, width=5, exclude=exclude, seed=1)


def test_draw_is_proportional_to_chapter_size():
    units = {"big": [f"b{i}" for i in range(900)], "small": [f"s{i}" for i in range(100)]}
    picks = [G.draw_test_windows(units, 1, 1, (), seed)[0].chapter for seed in range(400)]
    assert 0.83 < picks.count("big") / 400 < 0.97
