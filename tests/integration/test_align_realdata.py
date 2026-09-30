"""Real-data checks of the alignment baselines (review repro check #14, WS3 acceptance).

Run with HEVAJRA_RAW_DIR pointing at the fetched T18n0892.xml and
derge_rgyud_bum_nga.txt; ``pytest -s`` shows the tables. Segments come from the ingest
API (``ingest.cbeta.parse`` / ``ingest.derge.parse``).

Chapter grouping used here (a fixed table, not the concordance, which the pipeline
reads through the registry): the Derge reference chapter is "I.<n>" for local chapter
"D417:<n>" and "II.<n>" for "D418:<n>"; the Chinese chapters ("pin<n>") follow in order,
with three merged groups (synthesis 0.2 and sentinels S3, S9): pin 11 = I.11 + II.1,
pin 18 = II.8 + II.9, pin 20 = II.11 + II.12.
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from hevajra_matrix.align.anchors import extract, load_anchor_lexicon
from hevajra_matrix.align.dp import SOURCE_ANCHOR, SOURCE_ZERO, DPParams, align_groups
from hevajra_matrix.align.similarity import AnchorSimilarity, ZeroSimilarity
from hevajra_matrix.config import load_settings
from hevajra_matrix.core.types import Alignment, Segment

pytestmark = pytest.mark.realdata

ROOT = Path(__file__).resolve().parents[2]
REFERENCE, WITNESS = "bo_derge_D417_418", "zh_T0892_song"
ALIGNABLE = frozenset({"prose", "verse_line", "verse", "mantra"})   # content kinds of the ingesters
PART = {"D417": "I", "D418": "II"}

GROUPS: list[tuple[set[str], set[str]]] = (
    [({f"I.{n}"}, {f"pin{n}"}) for n in range(1, 11)]
    + [({"I.11", "II.1"}, {"pin11"})]
    + [({f"II.{n}"}, {f"pin{n + 10}"}) for n in range(2, 8)]
    + [({"II.8", "II.9"}, {"pin18"}), ({"II.10"}, {"pin19"}), ({"II.11", "II.12"}, {"pin20"})]
)

# Repro check #14 as observed with the v0.2 code at 1f9474c: segments containing the anchor,
# (Chinese, Tibetan). v0.2 matched Tibetan forms as substrings and had no traditional
# Chinese forms, the "cold forest" term or exclusion contexts.
V02_ASYMMETRY = {
    "term:māṃsa": (1, 218), "num:1000": (0, 19), "term:nāḍī": (6, 43), "term:abhiṣeka": (20, 72),
    "term:kapāla": (0, 18), "term:śmaśāna": (0, 15), "term:vidyā": (102, 5), "term:sūrya": (35, 16),
}


@lru_cache(maxsize=1)
def _segments() -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """(Tibetan reference units, Chinese witness segments), alignable kinds only."""
    from hevajra_matrix.ingest import cbeta, derge

    raw, data = Path(os.environ["HEVAJRA_RAW_DIR"]), ROOT / "data"
    zh = cbeta.parse(raw / "T18n0892.xml", WITNESS, data).segments
    bo = derge.parse(raw / "derge_rgyud_bum_nga.txt", ["D417", "D418"], REFERENCE, data).segments
    ref = tuple(_with_reference_chapter(s) for s in bo if s.kind in ALIGNABLE)
    wit = tuple(s for s in zh if s.kind in ALIGNABLE)
    return ref, wit


def _with_reference_chapter(s: Segment) -> Segment:
    if s.chapter is not None or s.local_chapter is None:
        return s
    toh, _, n = s.local_chapter.partition(":")
    return dataclasses.replace(s, chapter=f"{PART[toh]}.{n}")


@lru_cache(maxsize=1)
def _baselines() -> tuple[Alignment, Alignment, AnchorSimilarity, float]:
    ref, wit = _segments()
    params = DPParams.from_config(load_settings(ROOT).run)
    start = time.perf_counter()
    zero = align_groups(ref, wit, GROUPS, ZeroSimilarity(), params, source=SOURCE_ZERO,
                        reference=REFERENCE, witness=WITNESS)
    sim = AnchorSimilarity.fit(load_anchor_lexicon(ROOT / "data"), ref + wit)
    anchor = align_groups(ref, wit, GROUPS, sim, params, source=SOURCE_ANCHOR,
                          reference=REFERENCE, witness=WITNESS)
    return zero, anchor, sim, time.perf_counter() - start


def test_every_group_key_occurs_in_the_ingested_texts():
    ref, wit = _segments()
    ref_keys = {s.chapter for s in ref}
    wit_keys = {s.local_chapter for s in wit}
    for ref_group, wit_group in GROUPS:
        assert ref_group <= ref_keys, f"missing reference chapters {ref_group - ref_keys}"
        assert wit_group <= wit_keys, f"missing witness chapters {wit_group - wit_keys}"
    assert len(ref_keys - {None}) == 23 and len(wit_keys - {None}) == 20


def test_dp_baselines_cover_all_chapters_in_under_60_seconds():
    ref, wit = _segments()
    zero, anchor, _, seconds = _baselines()
    print(f"\nP1 + B0 over {len(ref)} reference units and {len(wit)} witness segments: {seconds:.1f} s")
    assert seconds < 60
    grouped_wit = {s.id for s in wit if any(s.local_chapter in w for _, w in GROUPS)}
    for alignment in (zero, anchor):
        by_ref = alignment.by_ref()
        assert len(by_ref) == len(ref) == sum(1 for link in alignment.links if link.ref_id)
        assert {w for link in alignment.links for w in link.wit_ids} == grouped_wit
        assert {link.source for link in alignment.links} == {alignment.source}
        relations = Counter(str(link.relation) for link in alignment.links)
        print(f"  {alignment.source:10} {dict(relations)}")
    shared = zero.pairs() & anchor.pairs()
    print(f"  link pairs: P1 {len(zero.pairs())}, B0 {len(anchor.pairs())}, "
          f"Jaccard {len(shared) / len(zero.pairs() | anchor.pairs()):.3f}")


def test_anchor_dp_links_more_units_with_agreeing_anchors_than_length_only():
    ref, wit = _segments()
    zero, anchor, sim, _ = _baselines()
    segs = {s.id: s for s in (*ref, *wit)}

    def agreeing(alignment: Alignment) -> int:
        return sum(1 for link in alignment.links if link.ref_id and link.wit_ids
                   and sim.score([segs[link.ref_id]], [segs[w] for w in link.wit_ids]) > 0)

    p1, b0 = agreeing(zero), agreeing(anchor)
    print(f"\nlinked units sharing an anchor with their counterpart: P1 {p1}, B0 {b0}")
    assert b0 > 2 * p1


def test_anchor_dp_links_the_i7_place_names():
    """Each I.7 place name occurs about once per witness; B0 should link most of them."""
    ref, wit = _segments()
    _, anchor, sim, _ = _baselines()
    places = sorted(k for k in sim.lexicon.keys() if k.startswith("place:"))
    segs = {s.id: s for s in wit}
    linked = [k for k in places
              if any(k in sim.bag(r) and any(k in sim.bag(segs[w]) for w in anchor.by_ref()[r.id].wit_ids)
                     for r in ref)]
    print(f"\nI.7 place names linked by B0: {len(linked)} of {len(places)}; missed: "
          f"{sorted(set(places) - set(linked))}")
    assert len(linked) >= 10


def test_anchor_asymmetry_is_far_below_v02():
    """Repro check #14: segments per anchor in each witness, now against v0.2."""
    ref, wit = _segments()
    lexicon = load_anchor_lexicon(ROOT / "data")
    zh = Counter(k for s in wit for k in extract(s.text, "zh", lexicon))
    bo = Counter(k for s in ref for k in extract(s.text, "bo", lexicon))
    print(f"\n  {'anchor':16} {'v0.2 zh':>8} {'v0.2 bo':>8} {'zh':>5} {'bo':>5}")
    for key, (old_zh, old_bo) in V02_ASYMMETRY.items():
        print(f"  {key:16} {old_zh:8d} {old_bo:8d} {zh[key]:5d} {bo[key]:5d}")
    for key in V02_ASYMMETRY:
        if key.startswith("num:") or key in lexicon.keys():
            assert max(zh[key], bo[key]) <= 3 * max(1, min(zh[key], bo[key])), key
        else:  # removed from the lexicon as one-sided
            assert zh[key] == bo[key] == 0, key
    assert bo["term:māṃsa"] == 0
    worst = max(sorted(lexicon.keys()), key=lambda k: max(zh[k], bo[k]) / max(1, min(zh[k], bo[k])))
    print(f"  most asymmetric lexicon anchor now: {worst} zh={zh[worst]} bo={bo[worst]}")
