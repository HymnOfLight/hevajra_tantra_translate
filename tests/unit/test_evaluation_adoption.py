"""The collate.v1/v2 adoption rule (evaluation.adoption, docs-zh/09 section 7): the constants
fixed on 2026-10-13, the note markers of deferred gold conventions and how a ``?`` is read, the
dev regions the gold covers, the scopes (outside the tuning window I.1, without the in-sample
units), the D_any level, and every branch of the decision, including the symmetric
replicate-kappa floor and "not ready"."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hevajra_matrix.config import ConfigError
from hevajra_matrix.core.types import Alignment, Link, Relation
from hevajra_matrix.evaluation import adoption as A
from hevajra_matrix.evaluation import gold as G
from hevajra_matrix.pipeline.adoption import load_note_markers

from test_pipeline_support import REPO

KONGZHI = "\u7a7a\u667a"     # the Hevajra glossary marker: blocks
HALF = "\u534a\u97f3"        # Q7, part-transcribed name: transcribed_name
EXPLAINED = "\u91ca\u8bcd"   # Q8, explained word: none
CLASS_NOUN = "\u7c7b\u540d"  # Q3, class noun: none
OUT_OF_WINDOW = "\u8d8a\u7a97"   # a marker of another kind (an out-of-window counterpart)
TN, EQ, PARA = "transcribed_name", Relation.EQUIVALENT, Relation.PARAPHRASE
REF, WIT = "bo_ref", "zh_wit"
MARKERS = load_note_markers(REPO / "data")


def row(unit: str, relation: str = "equivalent", convention: str = "none", note: str = "", window: str = "II.3",
        wit: tuple[str, ...] | None = None) -> G.GoldRow:
    wit_ids = () if relation == "no_counterpart" else wit or (f"T:{unit}",)
    return G.GoldRow(set_name="dev", window_id=window, row_type="ref", unit_id=unit, fingerprint="f", wit_ids=wit_ids,
                     relation=relation, convention=convention, note=note, annotator="ab", date="2026-10-13")


def gold(*rows: G.GoldRow) -> G.GoldSet:
    windows = tuple(dict.fromkeys(r.window_id for r in rows))
    return G.GoldSet("dev", WIT, rows, tuple(G.TestWindow(w, "a", "b", 1, w) for w in windows), REF)


# --------------------------------------------------------------------------- constants and note markers
def test_the_constants_are_those_fixed_on_2026_10_13() -> None:
    assert (A.TOLERANCE, A.MIN_CONVENTION_PRECISION, A.TUNING_WINDOWS, A.REPLICATES) == (0.05, 0.8, ("I.1",), 3)
    assert A.IN_SAMPLE_UNITS == ("D417:8b.5.2", "D417:8b.5.3", "D417:8b.5.4")


def test_the_shipped_markers_read_as_fixed_on_2026_10_13() -> None:
    assert [(m.marker, m.reads) for m in MARKERS] == [(KONGZHI, None), (HALF, TN), (EXPLAINED, "none"),
                                                      (CLASS_NOUN, "none")]


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(extra=1), r"unknown key\(s\) \['extra'\]"),
    (lambda d: d["markers"][0].update(default="none"), r"markers\[0\]: unknown key\(s\) \['default'\]"),
    (lambda d: d["markers"][1].update(reads="transliterated"), r"markers\[1\]: reads must be null or one of"),
    (lambda d: d["markers"][1].pop("reads"), r"markers\[1\]: reads must be null"),
    (lambda d: d["markers"][2].update(marker=HALF), r"markers\[2\]: marker must be a non-blank string listed once"),
    (lambda d: d.update(markers=[]), "markers must be a non-empty list"),
    (lambda d: d.update(schema_version=2), "schema_version 1 required"),
])
def test_a_malformed_markers_file_is_refused(change, message) -> None:
    doc = {"schema_version": 1, "markers": [{"marker": m.marker, "reads": m.reads, "question": m.question}
                                            for m in MARKERS]}
    change(doc)
    with pytest.raises(ConfigError, match=message):
        A.note_markers(doc, "note_markers.yaml")


# --------------------------------------------------------------------------- reading a "?"
def read(*rows: G.GoldRow) -> tuple[dict[str, str], dict[str, int], tuple[str, ...]]:
    found, counts, blocked = A.read_deferred(gold(*rows), MARKERS)
    return {r.unit_id: r.convention for r in found.rows}, counts, blocked


def test_each_marker_reads_as_its_default_and_the_glossary_marker_blocks() -> None:
    conventions, counts, blocked = read(row("u1", convention="", note=HALF),
                                        row("u2", "transliterated", "", note=EXPLAINED),
                                        row("u3", "transliterated", "", note=f"{CLASS_NOUN} dakini"),
                                        row("u4", convention="", note=KONGZHI))
    assert conventions == {"u1": TN, "u2": "none", "u3": "none", "u4": ""}
    assert counts == {KONGZHI: 1, HALF: 1, EXPLAINED: 1, CLASS_NOUN: 1} and blocked == ("u4",)


def test_the_glossary_marker_wins_then_the_part_transcribed_name() -> None:
    conventions, counts, blocked = read(row("u1", convention="", note=f"{HALF} {KONGZHI}"),
                                        row("u2", convention="", note=f"{CLASS_NOUN};{HALF}"),
                                        row("u3", convention="", note=f"{CLASS_NOUN} {EXPLAINED}"))
    assert conventions == {"u1": "", "u2": TN, "u3": "none"} and blocked == ("u1",)
    assert counts == {KONGZHI: 1, HALF: 1, EXPLAINED: 1, CLASS_NOUN: 0}, "each row counted under its deciding marker"


def test_rows_without_a_known_marker_stay_not_annotated() -> None:
    conventions, counts, blocked = read(row("u1", convention="", note=f"{OUT_OF_WINDOW} T:9.1"),
                                        row("u2", convention=""))
    assert conventions == {"u1": "", "u2": ""} and set(counts.values()) == {0} and blocked == ()
    unmarked = gold(row("u1", convention="", note=OUT_OF_WINDOW), row("u2", "no_counterpart", ""))
    scored = A.score_run(Alignment("claude:consensus", REF, WIT, ()), unmarked, MARKERS,
                         ref_kinds={"u1": "verse_line", "u2": "verse_line"})
    assert scored.unmarked == ("u1",) and scored.figures()["unmarked"] == ["u1"], "an unlinked row takes none"


def test_decided_cells_and_unlinked_rows_are_never_changed() -> None:
    conventions, counts, blocked = read(row("u1", convention="none", note=HALF),
                                        row("u2", convention=TN, note=KONGZHI),
                                        row("u3", "no_counterpart", "", note=HALF),
                                        row("u4", "lacuna", "", note=HALF))
    assert conventions == {"u1": "none", "u2": TN, "u3": "", "u4": ""}
    assert set(counts.values()) == {0} and blocked == (), "a decided Hevajra cell does not block"


def test_the_gold_gaps_are_the_dev_regions_not_imported_yet() -> None:
    found = gold(row("a1", window="I.1"), row("a2", window="I.1"), row("b1", window="I.7"))
    regions = {"I.1": ("a1", "a2"), "I.7": ("b1", "b2"), "II.3": ("c1",)}
    assert A.gold_gaps(found, regions) == {"I.7": 1, "II.3": 1}
    assert A.gold_gaps(found, {"I.1": ("a1", "a2")}) == {}
    assert A.gold_gaps(found, {"I.7": ("a1",)}) == {"I.7": 1}, "a unit counts in its own region's window only"


# --------------------------------------------------------------------------- scopes
IN1, IN2, IN3 = A.IN_SAMPLE_UNITS
GOLD = gold(
    row("D417:1b.1.1", window="I.1"), row("D417:1b.1.2", "paraphrase", window="I.1"),
    row(IN1, window="I.7"), row(IN2, window="I.7"), row(IN3, window="I.7"),
    row("D417:8b.5.5", convention=TN, window="I.7"),
    row("D418:17b.6.1"), row("D418:17b.6.2", "paraphrase"), row("D418:17b.6.3", convention=TN),
    row("D418:17b.6.4", convention="", note=HALF),        # a deferred "?" read as transcribed_name
    row("D418:17b.6.5", convention=""),                   # not annotated, no marker
    row("D418:17b.6.6", convention=TN))
KINDS = {r.unit_id: "verse_line" for r in GOLD.rows}
CLAIMED = {"D417:1b.1.1", IN1, "D417:8b.5.5", "D418:17b.6.3", "D418:17b.6.4", "D418:17b.6.6"}   # v2's transcribed names
V2 = Alignment("claude:consensus", REF, WIT, tuple(
    Link(r.unit_id, r.wit_ids, Relation(r.relation), flags=frozenset({TN}) if r.unit_id in CLAIMED else frozenset())
    for r in GOLD.rows))
V1 = Alignment("claude:consensus", REF, WIT, tuple(Link(r.unit_id, r.wit_ids, EQ) for r in GOLD.rows))


def test_the_rule_reads_outside_the_tuning_window_and_reports_precision_without_the_in_sample_units() -> None:
    v2 = A.score_run(V2, GOLD, MARKERS, ref_kinds=KINDS, conventions=True)
    assert (v2.deferred[HALF], v2.unmarked, v2.blocked) == (1, ("D418:17b.6.5",), ())
    default, excluded = (v2.figures()["readings"][r] for r in (A.DEFAULT, A.EXCLUDED))
    assert default[A.ALL]["units"] == 12 and default[A.ALL]["windows"] == ["I.1", "I.7", "II.3"]
    assert default[A.OUTSIDE]["units"] == 10 and default[A.OUTSIDE]["windows"] == ["I.7", "II.3"]
    assert default[A.OUTSIDE_NO_IN_SAMPLE]["units"] == 7
    # claims: 1b.1.1 wrong (I.1), IN1 wrong (in-sample), 8b.5.5, 17b.6.3, 17b.6.6 and the deferred 17b.6.4 right
    assert default[A.ALL]["convention_precision"] == pytest.approx(4 / 6)
    assert default[A.OUTSIDE]["convention_precision"] == 0.8
    assert default[A.OUTSIDE_NO_IN_SAMPLE]["convention_precision"] == 1.0
    # left out as stored, the "?" is not annotated: the claim on 17b.6.4 no longer counts
    assert excluded[A.OUTSIDE]["convention_precision"] == 0.75
    assert default[A.OUTSIDE]["convention_annotated"] == excluded[A.OUTSIDE]["convention_annotated"] + 1
    # shared units outside I.1: the three in-sample units, 17b.6.1 and 17b.6.2, all relations right
    assert default[A.OUTSIDE]["shared_units"] == 5 and default[A.OUTSIDE]["relation_kappa_shared"] == 1.0
    v1 = A.score_run(V1, GOLD, MARKERS, ref_kinds=KINDS, conventions=False).figures()
    assert v1["readings"][A.DEFAULT][A.OUTSIDE]["relation_kappa_shared"] == 0.0, "v1 never paraphrases"
    assert v1["readings"][A.DEFAULT][A.OUTSIDE]["convention_precision"] is None
    unmarked = A.decide(v1, v2.figures(), 0.85, 0.81, 0.8)
    assert unmarked["decision"] == A.NOT_READY and "D418:17b.6.5" in unmarked["not_ready"][0], "every '?' handled"
    # the same "?" given a marker (Q8, read as none): the rule decides
    marked = replace(GOLD, rows=tuple(replace(r, note=EXPLAINED) if r.unit_id == "D418:17b.6.5" else r
                                      for r in GOLD.rows))
    v1, v2 = (A.score_run(a, marked, MARKERS, ref_kinds=KINDS, conventions=c).figures()
              for a, c in ((V1, False), (V2, True)))
    decision = A.decide(v1, v2, 0.85, 0.81, 0.8)
    assert decision["decision"] == A.ADOPT and decision["sensitivity"]["decision"] == A.KEEP, "0.75 < 0.8"
    assert decision["conditions"]["convention_precision"]["v2"] == 0.8


def test_the_dany_level_of_each_run_is_compared_with_one_gold_level_on_every_shared_unit() -> None:
    rows = (row("u1"), row("u2", "abridged"), row("u3", "substitution"), row("u4"), row("u5", convention=TN))
    found = gold(*rows)
    kinds = {r.unit_id: "verse_line" for r in rows}

    def level(*links: Link) -> dict:
        return A.dany_level(G.score(Alignment("claude:consensus", REF, WIT, links), found, ref_kinds=kinds))
    # shared: u1-u4 (u5 has a convention); gold deviations u2, u3
    one = level(Link("u1", ("T:u1",), Relation.SUBSTITUTION), Link("u2", ("T:u2",), Relation.ABRIDGED),
                Link("u3", ("T:u3",), Relation.SUBSTITUTION), Link("u5", ("T:u5",), Relation.ABRIDGED))
    assert (one["units"], one["unresolved"], one["gold"], one["predicted"]) == (4, 1, 0.5, 0.75)
    assert one["difference"] == pytest.approx(0.25)
    two = level(Link("u1", ("T:u1",), EQ), Link("u2", ("T:u2",), EQ), Link("u4", ("T:u4",), EQ))
    assert (two["units"], two["unresolved"], two["gold"], two["predicted"], two["difference"]) == (4, 1, 0.5, 0.0, -0.5)


def test_the_paired_bootstrap_of_the_kappa_difference_leaves_out_the_tuning_window() -> None:
    v1 = A.score_run(V1, GOLD, MARKERS, ref_kinds=KINDS).readings[A.DEFAULT]
    v2 = A.score_run(V2, GOLD, MARKERS, ref_kinds=KINDS, conventions=True).readings[A.DEFAULT]
    diff = A.kappa_difference(v1, v2, n_boot=200, seed=3)
    assert diff["windows"] == ["I.7", "II.3"] and diff["point"] == 1.0
    assert diff["lo"] is not None and diff["lo"] <= diff["point"] <= diff["hi"]


# --------------------------------------------------------------------------- the decision
def figures(kappa: float | None, precision: float | None = None, *, excluded: tuple | None = None,
            full: float | None = None, blocked: tuple[str, ...] = (), unmarked: tuple[str, ...] = ()) -> dict:
    """What ``decide`` reads of ``RunScores.figures``: the kappa and precision outside I.1 (and
    on every window: ``full``), per reading (``excluded``: (kappa, precision) of that reading)."""
    def reading(k, p):
        outside = {"relation_kappa_shared": k, "convention_precision": p}
        return {A.ALL: {**outside, "relation_kappa_shared": k if full is None else full}, A.OUTSIDE: outside,
                A.OUTSIDE_NO_IN_SAMPLE: {"convention_precision": None if p is None else p + 0.01}}
    return {"blocked": list(blocked), "unmarked": list(unmarked), "readings": {A.DEFAULT: reading(kappa, precision),
                                                   A.EXCLUDED: reading(*(excluded or (kappa, precision)))}}


def passed(decision: dict) -> dict[str, bool | None]:
    return {name: c["passed"] for name, c in decision["conditions"].items()}


def test_all_three_conditions_adopt_collate_v2() -> None:
    d = A.decide(figures(0.70, full=0.72), figures(0.66, 0.85, full=0.69), 0.85, 0.82, 0.8)
    assert d["decision"] == A.ADOPT and d["not_ready"] == [] and d["note"] is None
    assert set(passed(d).values()) == {True}
    k = d["conditions"]["relation_kappa_shared"]
    assert (k["v1"], k["v2"], k["threshold"]) == (0.70, 0.66, pytest.approx(0.65))
    assert d["conditions"]["convention_precision"]["v2_without_in_sample"] == pytest.approx(0.86), "reported only"
    assert d["full_dev"] == {"relation_kappa_shared": {"v1": 0.72, "v2": 0.69}, "convention_precision": {"v2": 0.85}}
    assert d["rule"]["tolerance"] == 0.05 and d["rule"]["tuning_windows"] == ["I.1"] and d["rule"]["replicates"] == 3
    assert d["rule"]["in_sample_units"] == ["D417:8b.5.2", "D417:8b.5.3", "D417:8b.5.4"]


def test_exactly_at_the_tolerance_is_not_worse() -> None:
    assert 0.35 < 0.40 - A.TOLERANCE, "float noise the rule must not read as a failure"
    assert A.decide(figures(0.40), figures(0.35, 0.8), 0.9, 0.9, 0.8)["decision"] == A.ADOPT
    assert A.decide(figures(0.40), figures(0.349, 0.8), 0.9, 0.9, 0.8)["decision"] == A.KEEP


def test_a_worse_relation_agreement_or_a_low_precision_keeps_collate_v1() -> None:
    worse = A.decide(figures(0.70), figures(0.64, 0.95), 0.85, 0.85, 0.8)
    assert worse["decision"] == A.KEEP and passed(worse)["relation_kappa_shared"] is False
    imprecise = A.decide(figures(0.70), figures(0.75, 0.79), 0.85, 0.85, 0.8)
    assert imprecise["decision"] == A.KEEP and passed(imprecise)["convention_precision"] is False


def test_the_floor_binds_v2_only_when_v1_reaches_it() -> None:
    bound = A.decide(figures(0.70), figures(0.70, 0.9), 0.80, 0.79, 0.8)
    assert bound["decision"] == A.KEEP and passed(bound)["replicate_kappa"] is False
    assert bound["conditions"]["replicate_kappa"]["applies"] is True and bound["note"] is None
    free = A.decide(figures(0.70), figures(0.70, 0.9), 0.78, 0.70, 0.8)
    assert free["decision"] == A.ADOPT and free["conditions"]["replicate_kappa"] == {
        "v1": 0.78, "v2": 0.70, "threshold": 0.8, "applies": False, "passed": True}
    assert free["note"].startswith("collate.v2 has replicate kappa 0.700, below the floor 0.8: tune it")
    assert "tuning collate.v1" not in free["note"]
    at = A.decide(figures(0.70), figures(0.70, 0.9), 0.85, 0.8, 0.8)
    assert at["decision"] == A.ADOPT and passed(at)["replicate_kappa"] is True, "exactly at the floor reaches it"


def test_collate_v1_kept_below_the_floor_is_to_be_tuned_too() -> None:
    d = A.decide(figures(0.70), figures(0.50, 0.9), 0.78, 0.85, 0.8)
    assert d["decision"] == A.KEEP and d["note"].startswith("collate.v1 has replicate kappa 0.780")
    assert d["note"].endswith("(G2 is at risk; tuning collate.v1 makes a new version)")
    assert A.decide(figures(0.70), figures(0.70, 0.9), 0.78, 0.85, 0.8)["note"] is None, "v2 chosen, above"


@pytest.mark.parametrize("v1, v2, kappas, reason", [
    (None, figures(0.7, 0.9), (0.9, 0.9), "the v1 run has no consensus"),
    (figures(0.7), None, (0.9, 0.9), "the v2 run has no consensus"),
    (figures(0.7, blocked=("u9",)), figures(0.7, 0.9, blocked=("u9",)), (0.9, 0.9), "wait for a deferred decision"),
    (figures(None), figures(0.7, 0.9), (0.9, 0.9), "relation_kappa_shared: v1 undefined"),
    (figures(0.7), figures(0.7, None), (0.9, 0.9), "convention_precision: v2 undefined"),
    (figures(0.7), figures(0.7, 0.9), (0.9, None), "replicate_kappa: v2 undefined"),
    (figures(0.7), figures(0.7, 0.9), (None, 0.9), "replicate_kappa: v1 undefined"),
    (figures(0.7, unmarked=("u8",)), figures(0.7, 0.9, unmarked=("u8",)), (0.9, 0.9), "no known marker"),
    (figures(0.7), figures(0.7, 0.9), (0.9, 0.9), "does not cover every dev region yet: I.7 (3 unit(s) missing)"),
])
def test_not_ready(v1, v2, kappas, reason) -> None:
    d = A.decide(v1, v2, *kappas, 0.8, {"I.7": 3} if "dev region" in reason else None)
    assert d["decision"] == A.NOT_READY and any(reason in r for r in d["not_ready"]), d["not_ready"]
    assert d["note"] is None


def test_the_excluded_reading_is_a_sensitivity_result_only() -> None:
    d = A.decide(figures(0.70), figures(0.70, 0.9, excluded=(0.60, 0.9)), 0.85, 0.85, 0.8)
    assert d["decision"] == A.ADOPT and d["reading"] == A.DEFAULT
    assert d["sensitivity"]["decision"] == A.KEEP and d["sensitivity"]["reading"] == A.EXCLUDED
    assert d["sensitivity"]["conditions"]["relation_kappa_shared"]["v2"] == 0.60
