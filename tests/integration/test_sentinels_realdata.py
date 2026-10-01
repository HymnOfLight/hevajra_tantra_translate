"""Real-data checks of the evaluation stage (WS7 acceptance).

Run with HEVAJRA_RAW_DIR pointing at T18n0892.xml and derge_rgyud_bum_nga.txt;
``pytest -s`` prints the numbers. Checks:
    * the ingest-stage sentinels (S5 notes, S6 colophon paratext, S7 adjacency) pass;
    * every sentinel locus resolves (proposal and final checks run on a DP baseline without
      an "unresolved locus"; whether they PASS on a baseline is irrelevant);
    * the preregistered dev regions (plus the critique A5 regions) resolve, and 12 test
      windows of 25 units can be drawn from the frame outside them;
    * the DP baselines and the shuffled placebo score against a tiny synthetic gold without
      error, with bootstrap intervals and a paired comparison (no API key involved).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

from hevajra_matrix import registry
from hevajra_matrix.align.dp import SOURCE_ANCHOR, SOURCE_ZERO, DPParams, align_groups
from hevajra_matrix.align.anchors import load_anchor_lexicon
from hevajra_matrix.align.placebo import shuffle_alignment
from hevajra_matrix.align.similarity import AnchorSimilarity, ZeroSimilarity
from hevajra_matrix.config import load_settings
from hevajra_matrix.core.types import Relation
from hevajra_matrix.evaluation import gold as G
from hevajra_matrix.evaluation import sentinels as S
from hevajra_matrix.ingest import CONTENT_KINDS, cbeta, derge

pytestmark = pytest.mark.realdata

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ZH, BO = "zh_T0892_song", "bo_derge_D417_418"


@lru_cache(maxsize=1)
def texts():
    raw = Path(os.environ["HEVAJRA_RAW_DIR"])
    zh = cbeta.parse(raw / "T18n0892.xml", ZH, DATA)
    bo = derge.parse(raw / "derge_rgyud_bum_nga.txt", ["D417", "D418"], BO, DATA)
    return zh, bo


@lru_cache(maxsize=1)
def aligned():
    zh, bo = texts()
    conc = registry.load_concordance(DATA / "registry" / "concordance.yaml")
    ref = [s for s in conc.assign_reference_chapters(bo.segments, BO) if s.kind in CONTENT_KINDS]
    wit = [s for s in zh.segments if s.kind in CONTENT_KINDS]
    groups = [(conc.refs_for_local(ZH, local), [local]) for local in conc.local_order[ZH]]
    params = DPParams.from_config(load_settings(ROOT).run)
    zero = align_groups(ref, wit, groups, ZeroSimilarity(), params, source=SOURCE_ZERO, reference=BO, witness=ZH)
    sim = AnchorSimilarity.fit(load_anchor_lexicon(DATA), [*ref, *wit])
    anchor = align_groups(ref, wit, groups, sim, params, source=SOURCE_ANCHOR, reference=BO, witness=ZH)
    return ref, zero, anchor


def test_ingest_stage_sentinels_pass():
    zh, bo = texts()
    results = S.check(S.load(DATA / "sentinels" / "sentinels.yaml"), "ingest", [*zh.segments, *bo.segments])
    for r in results:
        print(f"\n{r.sentinel_id:40s} {r.status:9s} passed={r.passed} {r.detail}", end="")
    assert {r.sentinel_id.split("_")[0] for r in results} == {"S5a", "S5b", "S5c", "S5d", "S6", "S7"}
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_every_sentinel_locus_resolves_at_later_stages():
    zh, bo = texts()
    _, _, anchor = aligned()
    sentinels = S.load(DATA / "sentinels" / "sentinels.yaml")
    segments = [*zh.segments, *bo.segments]
    results = S.check(sentinels, "proposal", segments, alignment=anchor) + S.check(sentinels, "final", segments,
                                                                                   cells=[])
    unresolved = [(r.sentinel_id, r.detail) for r in results if "unresolved locus" in r.detail]
    assert not unresolved
    passed = sum(r.passed for r in results if r.stage == "proposal")
    print(f"\nB0 as a stand-in proposal: {passed} of {sum(r.stage == 'proposal' for r in results)} sentinels pass")


def test_dev_regions_and_test_windows():
    ref, _, _ = aligned()
    settings = load_settings(ROOT)
    regions = list(settings.prereg["gold"]["dev_regions"]) + [
        {"chapter": "I.1"}, {"chapter": "II.11", "last_units": 20, "id": "II.11-12"},
        {"chapter": "II.12", "id": "II.11-12"}]
    dev = G.dev_region_units(ref, regions)
    sizes = {k: len(v) for k, v in dev.items()}
    print(f"\ndev regions: {sizes}; total {sum(sizes.values())} units")
    assert sizes["I.7"] == sum(1 for s in ref if s.chapter == "I.7")
    assert sizes["II.3-around-D418-17b.6"] == 41 and sizes["II.9-first56"] == 56
    units_by_chapter: dict[str, list[str]] = {}
    for s in ref:
        units_by_chapter.setdefault(s.chapter, []).append(s.id)
    spec = settings.prereg["gold"]["test_windows"]
    excluded = {u for units in dev.values() for u in units}
    windows = G.draw_test_windows(units_by_chapter, spec["n_windows"], spec["width"], excluded, spec["seed"])
    print("test windows: " + ", ".join(f"{w.window_id} {w.chapter} {w.first_unit}" for w in windows))
    covered = [u for w in windows for u in G.window_units(w, units_by_chapter)]
    assert len(covered) == len(set(covered)) == 300 and not set(covered) & excluded


def test_baselines_score_against_a_tiny_synthetic_gold():
    ref, zero, anchor = aligned()
    by_ref = anchor.by_ref()
    chapter_of = {s.id: s.chapter for s in ref}
    units = [s.id for s in ref if s.chapter == "I.8"][:30]
    rows = []
    for n, uid in enumerate(units):      # synthetic "gold": B0's links, with every 7th unit declared NULL
        window = f"w{n // 10 + 1:02d}"
        link = by_ref[uid]
        null = n % 7 == 3 or link.relation is Relation.NO_COUNTERPART
        rows.append(G.GoldRow("dev", window, "ref", uid, "f", () if null else link.wit_ids,
                              "no_counterpart" if null else "equivalent"))
    gold = G.GoldSet("dev", ZH, tuple(rows), reference=BO)
    kinds = {s.id: s.kind for s in ref}
    placebo = shuffle_alignment(anchor, chapter_of, seed=1)
    scores = {a.source: G.score(a, gold, ref_kinds=kinds) for a in (zero, anchor, placebo)}
    for source, s in scores.items():
        m = s.metrics()
        est = G.interval_estimates(s, n_boot=200, seed=1, names=("link_f1",))["link_f1"]
        print(f"\n{source:18s} link F1 {m['link_f1']:.3f} [{est.lo:.3f}, {est.hi:.3f}] "
              f"status kappa {m['status_kappa']} NULL recall {m['null_recall']}", end="")
        assert s.counts()["units"] == 30 and s.windows() == ("w01", "w02", "w03")
    assert scores[SOURCE_ANCHOR].metrics()["link_recall"] == 1.0      # the gold pairs are its own links
    point, lo, hi = G.paired_difference(lambda s, ws: G.link_f1(s.on_windows(ws)), scores[SOURCE_ANCHOR],
                                        scores[SOURCE_ZERO], ("w01", "w02", "w03"), 200, 1)
    assert point is not None and lo <= point <= hi
