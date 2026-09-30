"""align.placebo: the shuffled control P2."""

from __future__ import annotations

from collections import Counter

import pytest

from hevajra_matrix.align.placebo import SOURCE_SHUFFLED, shuffle_alignment
from hevajra_matrix.core.types import Alignment, Link, Quote, Relation, WitnessOnlyKind

RELATIONS = [Relation.EQUIVALENT, Relation.PARAPHRASE, Relation.NO_COUNTERPART, Relation.SUBSTITUTION,
             Relation.ABRIDGED, Relation.REVERSAL]


def _alignment() -> tuple[Alignment, dict[str, str]]:
    """Two chapters of 30 units each with varied tuples, plus two witness-only links."""
    out, chapter_of = [], {}
    for c, chapter in enumerate(("I.1", "I.2")):
        for i in range(30):
            rid = f"R:{c + 1}a.{i + 1}.1"
            rel = RELATIONS[(i + c) % len(RELATIONS)]
            wit = () if rel is Relation.NO_COUNTERPART else (f"W:{c + 1}a.{i + 1}.1",)
            out.append(Link(ref_id=rid, wit_ids=wit, relation=rel, polarity_flip=rel is Relation.REVERSAL,
                            confidence="high", quotes=(Quote("ref", "x"),), flags=frozenset({"f"}), source="claude"))
            chapter_of[rid] = chapter
        out.append(Link(ref_id=None, wit_ids=(f"W:{c + 1}b.1.1",), relation=WitnessOnlyKind.ADDITION,
                        flags=frozenset({"kept"}), source="claude"))
    return Alignment(source="claude", reference="bo", witness="zh", links=tuple(out)), chapter_of


def _tuples(a: Alignment, chapter_of: dict[str, str], chapter: str) -> Counter:
    return Counter((l.wit_ids, l.relation, l.polarity_flip) for l in a.links
                   if l.ref_id is not None and chapter_of[l.ref_id] == chapter)


def test_per_chapter_multisets_are_preserved():
    a, chapter_of = _alignment()
    s = shuffle_alignment(a, chapter_of, seed=7)
    for chapter in ("I.1", "I.2"):
        assert _tuples(s, chapter_of, chapter) == _tuples(a, chapter_of, chapter)
    assert Counter(l.relation for l in s.links) == Counter(l.relation for l in a.links)


def test_assignments_change_and_units_stay_in_place():
    a, chapter_of = _alignment()
    s = shuffle_alignment(a, chapter_of, seed=7)
    assert [l.ref_id for l in s.links] == [l.ref_id for l in a.links]
    moved = sum(1 for x, y in zip(a.links, s.links) if x.ref_id and (x.wit_ids, x.relation) != (y.wit_ids, y.relation))
    assert moved >= 40  # of 60 units; a fixed point is possible, a mostly unchanged shuffle is not
    assert a.pairs() != s.pairs()


def test_nothing_crosses_a_chapter_boundary():
    a, chapter_of = _alignment()
    s = shuffle_alignment(a, chapter_of, seed=3)
    for link in s.links:
        if link.ref_id and link.wit_ids:
            assert link.ref_id.split(":")[1][0] == link.wit_ids[0].split(":")[1][0]


def test_seed_makes_it_reproducible_and_chapters_are_seeded_independently():
    a, chapter_of = _alignment()
    assert shuffle_alignment(a, chapter_of, seed=1) == shuffle_alignment(a, chapter_of, seed=1)
    assert shuffle_alignment(a, chapter_of, seed=1) != shuffle_alignment(a, chapter_of, seed=2)
    only_first = Alignment(a.source, a.reference, a.witness,
                           tuple(l for l in a.links if l.ref_id is None or chapter_of[l.ref_id] == "I.1"))
    first_alone = [l for l in shuffle_alignment(only_first, chapter_of, seed=1).links if l.ref_id]
    first_in_full = [l for l in shuffle_alignment(a, chapter_of, seed=1).links
                     if l.ref_id and chapter_of[l.ref_id] == "I.1"]
    assert first_alone == first_in_full


def test_source_is_set_and_unit_specific_fields_are_cleared():
    a, chapter_of = _alignment()
    s = shuffle_alignment(a, chapter_of, seed=7)
    assert s.source == SOURCE_SHUFFLED and (s.reference, s.witness) == ("bo", "zh")
    for link in s.links:
        assert link.source == SOURCE_SHUFFLED
        if link.ref_id is not None:
            assert link.quotes == () and link.flags == frozenset() and link.confidence is None


def test_witness_only_links_are_kept():
    a, chapter_of = _alignment()
    s = shuffle_alignment(a, chapter_of, seed=7)
    assert [(l.wit_ids, l.relation, l.flags) for l in s.witness_only()] == [
        (l.wit_ids, l.relation, l.flags) for l in a.witness_only()]


def test_units_without_a_chapter_are_an_error():
    a, chapter_of = _alignment()
    del chapter_of["R:1a.1.1"]
    with pytest.raises(ValueError, match="R:1a.1.1"):
        shuffle_alignment(a, chapter_of, seed=1)
