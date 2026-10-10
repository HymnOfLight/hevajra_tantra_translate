"""Scoring collate.v2 conventions against gold (evaluation.scores, evaluation.gold.score):
convention precision and recall per convention and overall, the D_any_conv caliber,
relation agreement, and n/a (never 0) for a source without conventions."""

from __future__ import annotations

import pytest

from hevajra_matrix.core.types import Alignment, Link, Relation
from hevajra_matrix.evaluation import gold as G
from hevajra_matrix.evaluation import scores as S

from test_evaluation_gold import GOLD, KINDS, PRED, REF, WINDOWS, WIT, row

EQ, SUB, PARA = Relation.EQUIVALENT, Relation.SUBSTITUTION, Relation.PARAPHRASE
TN, SR, UNSUPPORTED = "transcribed_name", "standing_rendering", "convention_unsupported"

# Hand case: six units in two windows; unit 5 has no annotated convention.
CONV_GOLD = G.GoldSet("test", WIT, (
    row("D:1a.1.1", "equivalent", ["T:1.1"], convention=TN),
    row("D:1a.2.1", "equivalent", ["T:2.1"], convention=SR),
    row("D:1a.3.1", "substitution", ["T:3.1"], convention="none"),
    row("D:1a.4.1", "equivalent", ["T:4.1"], window="w02", convention=TN),
    row("D:1a.5.1", "paraphrase", ["T:5.1"], window="w02"),
    row("D:1a.6.1", "equivalent", ["T:6.1"], window="w02", convention="none"),
), WINDOWS, REF)

CONV_PRED = Alignment("claude:consensus", REF, WIT, (
    Link("D:1a.1.1", ("T:1.1",), EQ, flags=frozenset({TN})),                  # right
    Link("D:1a.2.1", ("T:2.1",), EQ, flags=frozenset({TN})),                  # wrong convention: a miss both ways
    Link("D:1a.3.1", ("T:3.1",), SUB),
    # D:1a.4.1 unresolved: a recall miss, outside the precision denominator
    Link("D:1a.5.1", ("T:5.1",), PARA, flags=frozenset({SR})),                # gold not annotated: left out
    Link("D:1a.6.1", ("T:6.1",), EQ, flags=frozenset({TN, UNSUPPORTED})),     # gold none: a precision miss
))


def scored(conventions: bool) -> S.AlignmentScores:
    return G.score(CONV_PRED, CONV_GOLD, ref_kinds=KINDS, conventions=conventions)


def test_convention_precision_and_recall_by_hand():
    s = scored(True)
    m = s.metrics()
    # given a convention, gold annotated: units 1 (hit), 2 (TN for SR), 6 (TN for none)
    assert m["convention_precision"] == pytest.approx(1 / 3)
    # gold conventions: 1 (hit), 2 (missed: TN given), 4 (unresolved)
    assert m["convention_recall"] == pytest.approx(1 / 3)
    assert m["convention_f1"] == pytest.approx(1 / 3)
    assert m["convention_precision:transcribed_name"] == pytest.approx(1 / 3)
    assert m["convention_precision:standing_rendering"] is None, "its only prediction has no gold annotation"
    assert m["convention_recall:transcribed_name"] == 0.5
    assert m["convention_recall:standing_rendering"] == 0.0
    # every scored unit given a convention (1, 2, 5, 6), one of them unsupported
    assert m["convention_unsupported_rate"] == 0.25
    counts = s.counts()
    assert (counts["gold_convention_annotated"], counts["gold_conventions"], counts["pred_conventions"]) == (5, 3, 4)


def test_d_any_conv_caliber_and_relation_agreement_by_hand():
    m = scored(True).metrics()
    # annotated units 1, 2, 3, 4, 6: gold (d, d, d, d, n); pred (d, d, d, unresolved, d)
    assert m["dany_conv_kappa"] == pytest.approx((3 / 5 - 16 / 25) / (1 - 16 / 25))
    assert m["dany_conv_agreement:dev"] == pytest.approx(2 * 3 / (4 + 4))
    assert m["dany_conv_agreement:nondev"] == 0.0
    # relations (eq, eq, sub, eq, para, eq) against (eq, eq, sub, UNALIGNED, para, eq)
    assert m["relation_kappa"] == pytest.approx((5 / 6 - 14 / 36) / (1 - 14 / 36))


def test_a_source_without_conventions_reports_na_never_zero():
    """Baselines, the placebo, external alignments and a v1 consensus: the same links (the
    flags are ignored, never read as a v2 'none'), so every convention metric is None."""
    with_c, without = scored(True), scored(False)
    m = without.metrics()
    for name in ("convention_precision", "convention_recall", "convention_f1", "convention_unsupported_rate",
                 "convention_precision:transcribed_name", "convention_precision:standing_rendering",
                 "convention_recall:transcribed_name", "convention_recall:standing_rendering"):
        assert m[name] is None, name
    assert without.counts()["pred_conventions"] is None
    assert all(u.pred_convention is None and not u.convention_unsupported for u in without.units)
    # D_any_conv of such a source is its own D_any: (n, n, d, unresolved, n) against (d, d, d, d, n)
    assert m["dany_conv_kappa"] == pytest.approx((2 / 5 - 7 / 25) / (1 - 7 / 25))
    assert [u.pred_dev_conv for u in without.units] == [u.pred_dev for u in without.units]
    # the primary caliber and the relations never depend on conventions
    for name in ("dany_kappa", "relation_kappa", "status_kappa", "link_f1"):
        assert m[name] == with_c.metrics()[name], name


def test_a_convention_never_changes_d_any_but_sets_d_any_conv():
    by_unit = {u.unit_id: u for u in scored(True).units}
    first = by_unit["D:1a.1.1"]
    assert (first.gold_dev, first.gold_dev_conv, first.pred_dev, first.pred_dev_conv) == (False, True, False, True)
    assert (first.gold_convention, first.pred_convention) == (TN, TN)
    assert by_unit["D:1a.4.1"].pred_convention is None and by_unit["D:1a.4.1"].pred_dev_conv is None
    assert by_unit["D:1a.5.1"].gold_convention is None and by_unit["D:1a.5.1"].gold_dev_conv is None
    assert by_unit["D:1a.6.1"].convention_unsupported
    mantra = G.GoldSet("test", WIT, (row("D:1a.1.1", "transliterated", ["T:1.1"], convention="none"),), WINDOWS[:1])
    (unit,) = G.score(mantra.alignment, mantra, ref_kinds={"D:1a.1.1": "mantra"}, conventions=True).units
    assert (unit.gold_dev, unit.gold_dev_conv, unit.pred_dev_conv) == (False, False, False)


def test_gold_without_annotated_conventions_leaves_the_convention_metrics_undefined():
    m = G.score(PRED, GOLD, ref_kinds=KINDS, conventions=True).metrics()
    for name in ("convention_precision", "convention_recall", "convention_f1", "dany_conv_kappa",
                 "dany_conv_agreement:dev", "convention_unsupported_rate"):
        assert m[name] is None, name
    assert m["relation_kappa"] is not None


def test_gold_scores_perfectly_against_itself_on_conventions():
    m = G.score(CONV_GOLD.alignment, CONV_GOLD, ref_kinds=KINDS, conventions=True).metrics()
    for name in ("convention_precision", "convention_recall", "convention_f1", "relation_kappa",
                 "convention_recall:transcribed_name", "convention_recall:standing_rendering",
                 "dany_conv_agreement:dev", "dany_conv_agreement:nondev", "dany_conv_kappa"):
        assert m[name] == 1.0, name
    assert m["convention_unsupported_rate"] == 0.0


def test_convention_metrics_get_window_cluster_intervals():
    s = scored(True)
    assert s.on_windows(["w01"]).conventions is True
    est = G.interval_estimates(s, n_boot=200, seed=1)
    assert set(est) == set(s.metrics())
    for key in ("convention_precision", "convention_recall", "convention_recall:transcribed_name",
                "dany_conv_kappa", "relation_kappa", "dany_conv_agreement:dev"):
        assert est[key].point == s.metrics()[key], key
    rec = est["convention_recall:transcribed_name"]
    assert (rec.lo, rec.hi) == (0.0, 1.0), "w01 recalls its name, w02 misses its one"
    none = est["convention_precision:standing_rendering"]
    assert (none.point, none.lo, none.hi) == (None, None, None)
    na = G.interval_estimates(scored(False), n_boot=50, seed=1)
    assert (na["convention_precision"].point, na["convention_precision"].lo) == (None, None)
