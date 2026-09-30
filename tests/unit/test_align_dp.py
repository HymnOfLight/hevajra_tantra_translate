"""align.dp: parameters, the bead DP, conversion to Alignment, and chapter groups.

Length-only cases use English "words" (the length of an "en" segment is its word count),
so their lengths are explicit; the anchor case is the toy chapter in
data/fixtures/align_dp.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.align.anchors import load_anchor_lexicon
from hevajra_matrix.align.dp import (
    BEADS,
    SOURCE_ANCHOR,
    SOURCE_ZERO,
    Bead,
    DPParams,
    align,
    align_groups,
    beads,
    links,
)
from hevajra_matrix.align.similarity import AnchorSimilarity, ZeroSimilarity
from hevajra_matrix.config import ConfigError, load_settings
from hevajra_matrix.core.types import Relation, Segment, WitnessOnlyKind

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = yaml.safe_load((ROOT / "data" / "fixtures" / "align_dp.yaml").read_text(encoding="utf-8"))
DEFAULT = DPParams()
ZERO = ZeroSimilarity()


def words(sid: str, n: int, chapter: str | None = None, local: str | None = None, word: str = "w") -> Segment:
    return Segment(id=sid, witness="x", lang="en", text=" ".join([word] * n), start="1", end="1",
                   kind="verse", chapter=chapter, local_chapter=local)


def refs(*lengths: int, chapter: str | None = None) -> list[Segment]:
    return [words(f"R:1a.{i + 1}.1", n, chapter=chapter) for i, n in enumerate(lengths)]


def wits(*lengths: int, local: str | None = None) -> list[Segment]:
    return [words(f"W:1a.{i + 1}.1", n, local=local) for i, n in enumerate(lengths)]


def shape_of(bead_list: list[Bead]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [(b.ref, b.wit) for b in bead_list]


# --------------------------------------------------------------------------- parameters
def test_params_from_the_real_config_match_the_documented_defaults():
    assert DPParams.from_config(load_settings(ROOT).run) == DEFAULT


def test_params_reject_unknown_keys_and_bad_values():
    with pytest.raises(ConfigError, match="unknown key"):
        DPParams.from_config({"align": {"prior_11": 0.0, "prior_31": 1.0}})
    with pytest.raises(ConfigError, match="length_var"):
        DPParams(length_var=0.0)
    with pytest.raises(ConfigError, match="non-negative"):
        DPParams(anchor_weight=-1.0)
    with pytest.raises(ConfigError, match="finite"):
        DPParams(prior_null=float("nan"))
    with pytest.raises(ConfigError, match="finite"):
        DPParams(prior_null="3")


def test_missing_section_gives_defaults():
    assert DPParams.from_config({}) == DEFAULT


def test_every_bead_shape_has_its_prior():
    p = DPParams(prior_11=1, prior_null=2, prior_12=3, prior_22=4, prior_13=5)
    assert {s: p.prior(s) for s in BEADS} == {
        (1, 1): 1, (1, 0): 2, (0, 1): 2, (1, 2): 3, (2, 1): 3, (2, 2): 4, (1, 3): 5, (3, 1): 5}


# --------------------------------------------------------------------------- length only (P1)
def test_equal_lengths_align_one_to_one():
    assert shape_of(beads(refs(8, 8, 8), wits(6, 6, 6), ZERO, DEFAULT)) == [
        ((0,), (0,)), ((1,), (1,)), ((2,), (2,))]


def test_two_short_units_share_one_long_segment():
    # 2:1 costs its prior 2.4 and no length penalty; any split costs more (see module docstring).
    assert shape_of(beads(refs(4, 4, 8), wits(8, 8), ZERO, DEFAULT)) == [((0, 1), (0,)), ((2,), (1,))]


def test_zero_similarity_alignment_depends_on_lengths_only():
    a = align(refs(3, 9, 4, 4, 12), wits(10, 2, 7, 11), ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
    other_ref = [words(s.id, len(s.text.split()), word="z") for s in refs(3, 9, 4, 4, 12)]
    other_wit = [words(s.id, len(s.text.split()), word="q") for s in wits(10, 2, 7, 11)]
    b = align(other_ref, other_wit, ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
    assert a == b


def test_a_unit_without_counterpart_becomes_no_counterpart_when_merging_is_priced_out():
    no_merge = DPParams(prior_12=50.0, prior_22=50.0, prior_13=50.0)
    a = align(refs(10, 3, 10, 10), wits(10, 10, 10), ZERO, no_merge, source=SOURCE_ZERO, reference="r", witness="w")
    by_ref = a.by_ref()
    assert by_ref["R:1a.2.1"].relation is Relation.NO_COUNTERPART
    assert by_ref["R:1a.2.1"].wit_ids == ()
    assert [by_ref[f"R:1a.{i}.1"].wit_ids for i in (1, 3, 4)] == [("W:1a.1.1",), ("W:1a.2.1",), ("W:1a.3.1",)]


def test_empty_sides():
    only_ref = align(refs(3, 4), [], ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
    assert [(l.ref_id, l.relation) for l in only_ref.links] == [
        ("R:1a.1.1", Relation.NO_COUNTERPART), ("R:1a.2.1", Relation.NO_COUNTERPART)]
    only_wit = align([], wits(3), ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
    assert [(l.ref_id, l.wit_ids, l.relation) for l in only_wit.links] == [
        (None, ("W:1a.1.1",), WitnessOnlyKind.ADDITION)]
    assert align([], [], ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w").links == ()


def test_beads_cover_both_sides_in_order():
    bead_list = beads(refs(5, 1, 9, 2, 2, 7, 3), wits(4, 12, 1, 6, 6), ZERO, DEFAULT)
    assert [i for b in bead_list for i in b.ref] == list(range(7))
    assert [j for b in bead_list for j in b.wit] == list(range(5))
    assert all((len(b.ref), len(b.wit)) in BEADS for b in bead_list)


# --------------------------------------------------------------------------- similarity terms
@dataclass(frozen=True)
class ConflictOnDiagonal:
    """No shared evidence anywhere; a conflict for the 1:1 beads (0,0) and (1,1) only."""

    ref: tuple[str, ...]
    wit: tuple[str, ...]

    def score(self, ref, wit) -> float:
        return 0.0

    def conflict(self, ref, wit) -> bool:
        return len(ref) == len(wit) == 1 and self.ref.index(ref[0].id) == self.wit.index(wit[0].id)


def test_conflict_penalty_changes_the_bead_choice():
    r, w = refs(5, 5), wits(5, 5)
    sim = ConflictOnDiagonal(tuple(s.id for s in r), tuple(s.id for s in w))
    # two 1:1 beads cost 2 x 2.0 = 4.0 < 4.5 for one 2:2 bead ...
    assert shape_of(beads(r, w, sim, DEFAULT)) == [((0,), (0,)), ((1,), (1,))]
    # ... but 2 x 3.0 = 6.0 > 4.5
    assert shape_of(beads(r, w, sim, DPParams(anchor_conflict=3.0))) == [((0, 1), (0, 1))]
    assert shape_of(beads(r, w, sim, DPParams(anchor_conflict=0.0))) == [((0,), (0,)), ((1,), (1,))]


def test_repro6_anchors_on_both_sides_without_overlap_are_a_conflict():
    case = FIXTURE["conflict_case"]
    ref = [Segment(id="D417:1a.1.1", witness="bo", lang="bo", text=case["reference"], start="", end="", kind="prose")]
    wit = [Segment(id="T0892:0587c11.1", witness="zh", lang="zh", text=case["witness"], start="", end="", kind="prose")]
    sim = AnchorSimilarity.fit(load_anchor_lexicon(ROOT / "data"), ref + wit)
    assert sim.side(ref) and sim.side(wit)
    assert sim.score(ref, wit) == 0.0
    assert sim.conflict(ref, wit)
    # One segment per side: the length term is 0 (the ratio comes from this pair), so the
    # 1:1 bead costs exactly anchor_conflict = 2.0 against 2 * prior_null for two NULL beads.
    assert sorted(shape_of(beads(ref, wit, sim, DPParams(prior_null=0.9)))) == [((), (0,)), ((0,), ())]
    assert shape_of(beads(ref, wit, sim, DPParams(prior_null=1.1))) == [((0,), (0,))]
    assert shape_of(beads(ref, wit, ZERO, DPParams(prior_null=0.9))) == [((0,), (0,))]


def _fixture_segments() -> tuple[list[Segment], list[Segment]]:
    ref = [Segment(id=r["id"], witness="bo", lang="bo", text=r["text"], start="", end="", kind="verse_line")
           for r in FIXTURE["reference"]]
    wit = [Segment(id=w["id"], witness="zh", lang="zh", text=w["text"], start="", end="", kind="prose")
           for w in FIXTURE["witness"]]
    return ref, wit


def test_anchor_dp_links_by_content_and_leaves_the_unmatched_unit_without_counterpart():
    ref, wit = _fixture_segments()
    sim = AnchorSimilarity.fit(load_anchor_lexicon(ROOT / "data"), ref + wit)
    a = align(ref, wit, sim, DEFAULT, source=SOURCE_ANCHOR, reference="bo", witness="zh")
    got = {rid: list(link.wit_ids) for rid, link in a.by_ref().items()}
    assert got == FIXTURE["expected_anchor"]
    assert a.by_ref()["D417:8a.3.2"].relation is Relation.NO_COUNTERPART
    assert {link.source for link in a.links} == {SOURCE_ANCHOR}


# --------------------------------------------------------------------------- to Alignment
def test_n_to_m_beads_give_every_reference_unit_the_same_witness_ids():
    r, w = refs(1, 1, 1, 1), wits(1, 1, 1, 1)
    bead_list = [Bead((0, 1), (0,)), Bead((2,), (1, 2)), Bead((3,), ()), Bead((), (3,))]
    out = list(links(r, w, bead_list, "dp:test"))
    assert [(l.ref_id, l.wit_ids, l.relation) for l in out] == [
        ("R:1a.1.1", ("W:1a.1.1",), Relation.EQUIVALENT),
        ("R:1a.2.1", ("W:1a.1.1",), Relation.EQUIVALENT),
        ("R:1a.3.1", ("W:1a.2.1", "W:1a.3.1"), Relation.EQUIVALENT),
        ("R:1a.4.1", (), Relation.NO_COUNTERPART),
        (None, ("W:1a.4.1",), WitnessOnlyKind.ADDITION),
    ]
    assert all(l.source == "dp:test" and not l.polarity_flip and not l.quotes for l in out)


def test_alignment_carries_source_reference_and_witness():
    a = align(refs(2), wits(2), ZERO, DEFAULT, source=SOURCE_ZERO, reference="bo_ref", witness="zh_wit")
    assert (a.source, a.reference, a.witness) == (SOURCE_ZERO, "bo_ref", "zh_wit")
    assert a.links[0].source == SOURCE_ZERO


# --------------------------------------------------------------------------- chapter groups
def _two_chapters() -> tuple[list[Segment], list[Segment]]:
    ref = [words("R:1a.1.1", 5, chapter="I.1"), words("R:1a.2.1", 5, chapter="I.2"),
           words("R:1a.3.1", 5, chapter="I.1"), words("R:1a.4.1", 5, chapter="I.3"),
           words("R:1a.5.1", 5, chapter=None)]
    wit = [words("W:1a.1.1", 5, local="p1"), words("W:1a.2.1", 5, local="p2"),
           words("W:1a.3.1", 5, local="p1"), words("W:1a.4.1", 5, local="p9")]
    return ref, wit


def test_groups_are_aligned_independently_and_concatenated_in_group_order():
    ref, wit = _two_chapters()
    groups = [({"I.2", "I.3"}, {"p2"}), ({"I.1"}, {"p1"})]
    a = align_groups(ref, wit, groups, ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
    group_one = align(ref[1:2] + ref[3:4], wit[1:2], ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
    group_two = align([ref[0], ref[2]], [wit[0], wit[2]], ZERO, DEFAULT, source=SOURCE_ZERO, reference="r",
                      witness="w")
    assert a.links == group_one.links + group_two.links
    assert a.by_ref()["R:1a.3.1"].wit_ids == ("W:1a.3.1",)


def test_segments_outside_every_group_get_no_link():
    ref, wit = _two_chapters()
    a = align_groups(ref, wit, [({"I.1"}, {"p1"})], ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
    assert set(a.by_ref()) == {"R:1a.1.1", "R:1a.3.1"}
    assert {w for link in a.links for w in link.wit_ids} == {"W:1a.1.1", "W:1a.3.1"}


def test_a_chapter_key_may_occur_in_one_group_only():
    ref, wit = _two_chapters()
    with pytest.raises(ValueError, match="reference chapter key"):
        align_groups(ref, wit, [({"I.1"}, {"p1"}), ({"I.1", "I.2"}, {"p2"})], ZERO, DEFAULT,
                     source=SOURCE_ZERO, reference="r", witness="w")
    with pytest.raises(ValueError, match="witness chapter key"):
        align_groups(ref, wit, [({"I.1"}, {"p1"}), ({"I.2"}, {"p1"})], ZERO, DEFAULT,
                     source=SOURCE_ZERO, reference="r", witness="w")


def test_a_bare_string_is_not_a_key_collection():
    ref, wit = _two_chapters()
    with pytest.raises(TypeError, match="collection"):
        align_groups(ref, wit, [("I.1", {"p1"})], ZERO, DEFAULT, source=SOURCE_ZERO, reference="r", witness="w")
