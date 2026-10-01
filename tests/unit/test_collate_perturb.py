"""collate.perturb: perturbation builders return known truths; the scores use them."""

from __future__ import annotations

import pytest

from hevajra_matrix.collate.perturb import (
    DeletionTruth,
    NegationTruth,
    Rate,
    WrongWindowTruth,
    deletion_recall,
    delete_segments,
    false_link_rate,
    negation_recall,
    remove_negators,
    strip_negator,
    wrong_window,
)
from hevajra_matrix.core.textnorm import TSHEG, find_terms
from hevajra_matrix.core.types import Alignment, Link, Relation

from test_collate_support import gold, lexicon, window

U_FISH, Z_FISH = "BT:1a.2.1", "ZT:0001a04.1"
U_SAID, Z_SAID = "BT:1a.1.1", "ZT:0001a03.1"


def alignment(*links: Link) -> Alignment:
    return Alignment("x", "bo_test", "zh_test", links)


# --------------------------------------------------------------------------- wrong window
def test_wrong_window_moves_the_core_and_keeps_the_text() -> None:
    w = window()
    perturbed, truth = wrong_window(w, ["pin3"])
    assert perturbed.core_locals == frozenset({"pin3"})
    assert perturbed.text == w.text and perturbed.units == w.units
    assert truth == WrongWindowTruth(frozenset(s.id for s in w.text if s.local_chapter == "pin3"))
    assert perturbed.key != w.key


@pytest.mark.parametrize("others", [[], ["pin1"], ["pin1", "pin3"], ["pin9"]])
def test_wrong_window_must_be_disjoint_existing_chapters(others: list[str]) -> None:
    with pytest.raises(ValueError):
        wrong_window(window(), others)


def test_false_link_rate_counts_links_into_the_wrong_window() -> None:
    truth = WrongWindowTruth(frozenset({"ZT:0001c02.1", "ZT:0001c03.1"}))
    result = alignment(Link("u1", ("ZT:0001c02.1",), Relation.EQUIVALENT),
                       Link("u2", (Z_SAID,), Relation.EQUIVALENT),
                       Link("u3", (), Relation.NO_COUNTERPART),
                       Link(None, ("ZT:0001c03.1",), Relation.EQUIVALENT))
    rate = false_link_rate(truth, result)
    assert rate == Rate(1, 3) and rate.value == pytest.approx(1 / 3)
    assert Rate(0, 0).value is None


# --------------------------------------------------------------------------- deletion
def test_delete_segments_removes_core_content_with_its_notes() -> None:
    w = window()
    content = [s.id for s in w.text if s.id in w.core and s.kind in {"prose", "verse_line", "mantra"}]
    for seed in range(10):
        perturbed, truth = delete_segments(w, 0.3, seed)
        deleted = set(truth.deleted)
        chosen_content = deleted & set(content)
        assert len(chosen_content) == round(0.3 * len(content))
        assert all(s.id not in deleted for s in perturbed.text)
        assert len(perturbed.text) == len(w.text) - len(deleted)
        if "ZT:0001a05.1" in deleted:
            assert "ZT:0001a05.n1" in deleted, "a deleted segment takes its notes along"
        assert list(perturbed.wit_handles) == [f"z{i:04d}" for i in range(1, len(perturbed.text) + 1)], \
            "handles are renumbered, so the gap leaves no trace"
    assert delete_segments(w, 0.3, 4) == delete_segments(w, 0.3, 4), "seeded"


def test_delete_segments_deletes_at_least_one_and_validates() -> None:
    w = window()
    _, truth = delete_segments(w, 0.01, 0)
    assert len(truth.deleted) >= 1
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            delete_segments(w, bad, 0)


def test_deletion_recall_scores_only_fully_deleted_counterparts() -> None:
    truth = DeletionTruth(("z1", "z2"))
    expected = alignment(Link("u1", ("z1",), Relation.EQUIVALENT), Link("u2", ("z2", "z3"), Relation.EQUIVALENT),
                         Link("u3", ("z2",), Relation.EQUIVALENT), Link("u4", (), Relation.NO_COUNTERPART))
    result = alignment(Link("u1", (), Relation.NO_COUNTERPART), Link("u3", ("z9",), Relation.EQUIVALENT))
    assert deletion_recall(truth, expected, result) == Rate(1, 2)
    abridged = alignment(Link("u1", ("z5",), Relation.ABRIDGED), Link("u3", (), Relation.NO_COUNTERPART))
    assert deletion_recall(truth, expected, abridged) == Rate(2, 2)


def test_deletion_recall_scores_only_the_units_of_the_perturbed_window() -> None:
    # A deleted core segment can be the counterpart of a unit in another chunk of the
    # chapter or in a neighbouring chapter; this window's answer never assesses it, so it
    # must not count as a miss (on the real texts it cut the recall from 25/25 to 25/172).
    w = window()
    _, truth = delete_segments(w, 0.3, 0)
    assert truth.units == tuple(u.id for u in w.units)
    inside, outside = truth.units[0], "BT:9z.9.9"
    expected = alignment(Link(inside, (truth.deleted[0],), Relation.EQUIVALENT),
                         Link(outside, (truth.deleted[0],), Relation.EQUIVALENT))
    result = alignment(Link(inside, (), Relation.NO_COUNTERPART))
    assert deletion_recall(truth, expected, result) == Rate(1, 1)


def test_units_shared_by_consecutive_chunks_are_scored_once() -> None:
    # The last overlap_next units of a chunk open the next chunk too; both perturbed chunks
    # used to score them, so deletion recall counted them twice.
    from dataclasses import replace
    w = replace(window(), overlap_next=1)
    _, truth = delete_segments(w, 0.3, 0)
    assert truth.units == tuple(u.id for u in w.units[:-1]) and w.units[-1].id not in truth.units


def test_deletion_recall_counts_unresolved_as_a_miss() -> None:
    truth = DeletionTruth(("z1",))
    expected = alignment(Link("u1", ("z1",), Relation.EQUIVALENT))
    assert deletion_recall(truth, expected, alignment()) == Rate(0, 1)


# --------------------------------------------------------------------------- negation
def test_remove_negators_strips_one_negator_from_gold_pairs() -> None:
    w = window()
    g = gold()
    pairs = [(uid, g[uid][0][0]) for uid in (U_SAID, U_FISH)]      # U_SAID's segment has no negator
    perturbed, truth = remove_negators(w, pairs, lexicon().negators, n=5)
    assert truth == NegationTruth(((U_FISH, Z_FISH),))
    before, after = w.segment[Z_FISH].text, perturbed.segment[Z_FISH].text
    negators = lexicon().negators
    assert find_terms(before, "zh", negators.for_lang("zh")) and not find_terms(after, "zh", negators.for_lang("zh"))
    assert len(after) == len(before) - 1
    assert perturbed.segment[Z_SAID].text == w.segment[Z_SAID].text


def test_remove_negators_respects_n_and_skips_pairs_outside_the_window() -> None:
    w = window()
    pairs = [("BT:1b.1.1", "ZT:0001b02.1"), (U_FISH, Z_FISH), ("BT:1a.3.2", "ZT:0001a07.1")]
    _, truth = remove_negators(w, pairs, lexicon().negators, n=1)
    assert truth.pairs == ((U_FISH, Z_FISH),)
    with pytest.raises(ValueError):
        remove_negators(w, pairs, lexicon().negators, n=0)


def test_strip_negator_on_tibetan_removes_a_whole_syllable_and_its_tsheg() -> None:
    w = window()
    negators = lexicon().negators
    text = w.segment[U_FISH].text
    stripped = strip_negator(text, "bo", negators)
    assert stripped is not None and not find_terms(stripped, "bo", negators.for_lang("bo"))
    assert TSHEG * 2 not in stripped                               # no doubled tsheg left behind
    assert strip_negator(w.segment[U_SAID].text, "bo", negators) is None


def test_strip_negator_ignores_exclusion_contexts() -> None:
    negators = lexicon().negators
    exclusion = negators.exclusions_for("bo")[0]
    assert strip_negator(exclusion, "bo", negators) is None


def test_negation_recall() -> None:
    truth = NegationTruth((("u1", "z1"), ("u2", "z2"), ("u3", "z3")))
    result = alignment(Link("u1", ("z1",), Relation.REVERSAL, polarity_flip=True),
                       Link("u2", ("z2",), Relation.SUBSTITUTION, polarity_flip=True),
                       Link("u3", ("z3",), Relation.EQUIVALENT))
    assert negation_recall(truth, result) == Rate(2, 3)
