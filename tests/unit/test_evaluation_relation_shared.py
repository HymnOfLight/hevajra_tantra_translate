"""relation_kappa_shared (evaluation.scores): the relation agreement on the units the v1 and
v2 codebooks code alike, the caliber the section-7 adoption rule compares. Linked gold units
with an annotated convention ``none`` whose gold relation v2 did not recode (``generalised``,
``category_name_omitted``); computed for every source alike, with window-cluster intervals."""

from __future__ import annotations

import pytest

from hevajra_matrix.core.types import Alignment, Link, Relation
from hevajra_matrix.evaluation import gold as G
from hevajra_matrix.evaluation import scores as S

from test_evaluation_gold import GOLD, KINDS, PRED, REF, WIT, row

EQ, SUB, NC = Relation.EQUIVALENT, Relation.SUBSTITUTION, Relation.NO_COUNTERPART
CNO, TRANSLIT = Relation.CATEGORY_NAME_OMITTED, Relation.TRANSLITERATED
UNITS = [f"D:1a.{i}.1" for i in range(1, 9)]
WINDOWS = (G.TestWindow("w01", UNITS[0], UNITS[3], 4, "I.1", 5, "2026-10-01"),
           G.TestWindow("w02", UNITS[4], UNITS[7], 4, "I.2", 5, "2026-10-01"))
SHARED_KINDS = {u: "verse_line" for u in UNITS}

SHARED_GOLD = G.GoldSet("test", WIT, (
    row(UNITS[0], "equivalent", ["T:1.1"], convention="none"),                       # shared
    row(UNITS[1], "substitution", ["T:2.1"], convention="none"),                     # shared
    row(UNITS[2], "substitution", ["T:3.1"], convention="none"),                     # shared
    row(UNITS[3], "generalised", ["T:4.1"], convention="none"),                      # recoded by v2
    row(UNITS[4], "equivalent", ["T:5.1"], window="w02", convention="none"),         # shared
    row(UNITS[5], "category_name_omitted", ["T:6.1"], window="w02", convention="none"),   # v1 caliber
    row(UNITS[6], "no_counterpart", window="w02", convention="none"),                # not linked
    row(UNITS[7], "equivalent", ["T:8.1"], window="w02", convention="transcribed_name"),  # a convention
), WINDOWS, REF)

SHARED_PRED = Alignment("claude:consensus", REF, WIT, (
    Link(UNITS[0], ("T:1.1",), EQ),
    Link(UNITS[1], ("T:2.1",), SUB),
    Link(UNITS[2], ("T:3.1",), EQ),
    Link(UNITS[3], ("T:4.1",), CNO),          # v1 coding of a merged list: left out
    # UNITS[4] unresolved: UNALIGNED, kept in the shared units
    Link(UNITS[5], ("T:6.1",), CNO),
    Link(UNITS[6], ("T:7.1",), EQ),           # gold null: left out
    Link(UNITS[7], ("T:8.1",), TRANSLIT),     # v1 coding of a transcribed name: left out
))


def scored(conventions: bool) -> S.AlignmentScores:
    return G.score(SHARED_PRED, SHARED_GOLD, ref_kinds=SHARED_KINDS, conventions=conventions)


def test_shared_kappa_by_hand():
    m = scored(True).metrics()
    # shared units 1, 2, 3, 5: gold (eq, sub, sub, eq) against (eq, sub, eq, UNALIGNED)
    po, pe = 2 / 4, 2 / 4 * 2 / 4 + 2 / 4 * 1 / 4
    assert m["relation_kappa_shared"] == pytest.approx((po - pe) / (1 - pe))
    assert m["relation_kappa_shared"] == pytest.approx(0.2)
    # the overall kappa still sees every unit, including the recoded ones
    assert m["relation_kappa"] != m["relation_kappa_shared"]


def test_shared_units_are_exactly_those_both_codebooks_code_alike():
    shared = [u.unit_id for u in scored(True).units if S._codebooks_agree(u)]
    assert shared == [UNITS[0], UNITS[1], UNITS[2], UNITS[4]]


def test_v1_codings_of_recoded_units_do_not_change_the_shared_kappa():
    """The same predictions, once with the recoded units coded the v1 way (category name
    omitted, transliterated) and once the v2 way: the shared kappa is identical, only the
    overall relation_kappa (descriptive) moves."""
    v2_way = {UNITS[3]: Relation.GENERALISED, UNITS[7]: EQ}
    links = tuple(Link(k.ref_id, k.wit_ids, v2_way.get(str(k.ref_id), k.relation)) for k in SHARED_PRED.links)
    v1 = scored(True).metrics()
    v2 = G.score(Alignment("claude:consensus", REF, WIT, links), SHARED_GOLD, ref_kinds=SHARED_KINDS,
                 conventions=True).metrics()
    assert v1["relation_kappa_shared"] == v2["relation_kappa_shared"]
    assert v1["relation_kappa"] < v2["relation_kappa"]
    gold = G.score(SHARED_GOLD.alignment, SHARED_GOLD, ref_kinds=SHARED_KINDS, conventions=True).metrics()
    assert gold["relation_kappa_shared"] == gold["relation_kappa"] == 1.0


def test_shared_kappa_is_computed_for_every_source_alike():
    """A v1 consensus, a baseline or the placebo: the gold side decides the units, so the
    metric never depends on whether the source produces conventions."""
    assert scored(False).metrics()["relation_kappa_shared"] == scored(True).metrics()["relation_kappa_shared"]
    assert scored(False).metrics()["relation_kappa_shared"] == pytest.approx(0.2)


def test_gold_without_annotated_conventions_has_no_shared_units():
    m = G.score(PRED, GOLD, ref_kinds=KINDS, conventions=False).metrics()
    assert m["relation_kappa_shared"] is None
    assert m["relation_kappa"] is not None


def test_shared_kappa_is_registered_and_gets_window_cluster_intervals():
    assert S.METRICS["relation_kappa_shared"] is S.relation_kappa_shared is G.relation_kappa_shared
    assert "relation_kappa_shared" in S.COLLATE_V2_FAMILIES
    s = scored(True)
    est = G.interval_estimates(s, n_boot=200, seed=1, names=["relation_kappa_shared"])
    shared = est["relation_kappa_shared"]
    assert shared.point == pytest.approx(0.2)
    assert shared.lo is not None and shared.hi is not None and shared.lo <= shared.point <= shared.hi
    # one window alone: w01 holds (eq, eq), (sub, sub), (sub, eq); w02 holds (eq, UNALIGNED)
    assert s.on_windows(["w02"]).metrics()["relation_kappa_shared"] == 0.0
    assert set(G.interval_estimates(s, n_boot=20, seed=1)) == set(s.metrics())


def test_scores_docstring_names_the_adoption_caliber():
    doc = S.__doc__ or ""
    assert "relation_kappa_shared" in doc and "adoption rule" in doc
    assert "descriptive" in doc


def test_without_each_window_scores_the_other_windows():
    s = scored(True)
    left_out = G.without_each_window(s)
    assert set(left_out) == set(s.windows()) == {"w01", "w02"}
    assert left_out["w01"] == s.on_windows(["w02"]).metrics()
    assert left_out["w02"]["relation_kappa_shared"] == s.on_windows(["w01"]).metrics()["relation_kappa_shared"]
    assert G.without_each_window(s.on_windows(["w01"])) == {}

