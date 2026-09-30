"""collate.consensus: replicate vote, tie-break, medoid links, grades and agreement."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from hevajra_matrix.collate.consensus import (
    SOURCE_CONSENSUS,
    agreement,
    consensus,
    fleiss_kappa,
    has_majority,
    jaccard,
)
from hevajra_matrix.collate.verify import Collation
from hevajra_matrix.core.types import Alignment, Grade, Link, Relation, WitnessOnlyKind

EQ, SUB, REV, GEN, ABR, NONE = (Relation.EQUIVALENT, Relation.SUBSTITUTION, Relation.REVERSAL,
                                Relation.GENERALISED, Relation.ABRIDGED, Relation.NO_COUNTERPART)
KINDS = {"u1": "prose", "u2": "prose", "u3": "mantra"}


def L(ref, wit, relation=EQ, flip=False, flags=(), rep="r1"):
    return Link(ref, tuple(wit), relation, polarity_flip=flip, flags=frozenset(flags), source=f"src:{rep}")


def rep(*links, unresolved=None, name="r1"):
    return Collation(Alignment(f"src:{name}", "bo", "zh", tuple(links)), MappingProxyType(dict(unresolved or {})))


def run(*replicates, kinds=None):
    return consensus(replicates, kinds or {"u1": "prose"})


# --------------------------------------------------------------------------- vote and grades
def test_unanimous_unflagged_is_grade_b() -> None:
    a, grades, reasons = run(rep(L("u1", ["z1"])), rep(L("u1", ["z1"])), rep(L("u1", ["z1"])))
    (link,) = a.links
    assert (link.ref_id, link.wit_ids, link.relation, link.source) == ("u1", ("z1",), EQ, SOURCE_CONSENSUS)
    assert grades == {"u1": Grade.B} and reasons == {}


def test_unanimous_with_only_corroborated_is_still_b() -> None:
    reps = [rep(L("u1", ["z1"], REV, True, {"corroborated"})) for _ in range(3)]
    _, grades, _ = consensus(reps, {"u1": "prose"})
    assert grades["u1"] is Grade.B


def test_unanimous_class_but_any_other_flag_is_c() -> None:
    reps = [rep(L("u1", ["z1"])), rep(L("u1", ["z1"], flags={"relocation"})), rep(L("u1", ["z1"]))]
    a, grades, _ = consensus(reps, {"u1": "prose"})
    assert grades["u1"] is Grade.C
    assert "relocation" in a.links[0].flags, "flags of every replicate are kept"


def test_unanimous_class_with_different_relations_is_b() -> None:
    """The vote is over outcome class: equivalent and paraphrase are both NONDEV."""
    reps = [rep(L("u1", ["z1"])), rep(L("u1", ["z1"], Relation.PARAPHRASE)), rep(L("u1", ["z1"]))]
    a, grades, _ = consensus(reps, {"u1": "prose"})
    assert grades["u1"] is Grade.B and a.links[0].relation is EQ


def test_two_of_three_is_grade_c_with_the_majority_class() -> None:
    a, grades, _ = run(rep(L("u1", [], NONE)), rep(L("u1", ["z1"])), rep(L("u1", [], NONE)))
    assert a.links[0].relation is NONE and a.links[0].wit_ids == ()
    assert grades["u1"] is Grade.C


def test_no_majority_is_unaligned_grade_x() -> None:
    a, grades, reasons = run(rep(L("u1", [], NONE)), rep(L("u1", ["z1"])), rep(L("u1", ["z1"], ABR)))
    assert a.by_ref() == {}
    assert reasons == {"u1": "no_majority"} and grades == {"u1": Grade.X}


def test_one_resolved_and_two_unresolved_keeps_the_unresolved_reason() -> None:
    reps = [rep(unresolved={"u1": "refused:bio"}), rep(L("u1", ["z1"])), rep(unresolved={"u1": "refused:bio"})]
    a, grades, reasons = consensus(reps, {"u1": "prose"})
    assert a.by_ref() == {} and reasons == {"u1": "refused:bio"} and grades["u1"] is Grade.X


def test_substituted_replicate_does_not_count_but_majority_still_decides() -> None:
    reps = [rep(L("u1", ["z1"])), rep(unresolved={"u1": "substituted_model"}), rep(L("u1", ["z1"]))]
    a, grades, _ = consensus(reps, {"u1": "prose"})
    assert a.links[0].wit_ids == ("z1",) and grades["u1"] is Grade.C


def test_unit_mentioned_by_no_replicate_is_unassessed() -> None:
    a, grades, reasons = consensus([rep(), rep(), rep()], {"u1": "prose"})
    assert reasons == {"u1": "unassessed"} and grades["u1"] is Grade.X


def test_single_replicate() -> None:
    _, grades, _ = consensus([rep(L("u1", ["z1"]))], {"u1": "prose"})
    assert grades["u1"] is Grade.B


def test_reference_kind_enters_the_outcome_class() -> None:
    """transliterated is NONDEV for a mantra unit and DEV_PRESENT otherwise."""
    tr = Relation.TRANSLITERATED
    reps = [rep(L("u3", ["z1"], tr)), rep(L("u3", ["z1"], EQ)), rep(L("u3", ["z1"], tr))]
    _, grades, _ = consensus(reps, {"u3": "mantra"})
    assert grades["u3"] is Grade.B
    reps = [rep(L("u1", ["z1"], tr)), rep(L("u1", ["z1"], EQ)), rep(L("u1", ["z1"], tr))]
    a, grades, _ = consensus(reps, {"u1": "prose"})
    assert grades["u1"] is Grade.C and a.links[0].relation is tr


# --------------------------------------------------------------------------- relation choice
def test_relation_tie_is_broken_by_priority() -> None:
    reps = [rep(L("u1", ["z1"], GEN)), rep(L("u1", ["z1"], REV, True)), rep(L("u1", ["z1"], SUB))]
    a, grades, _ = run(*reps)
    assert a.links[0].relation is REV and a.links[0].polarity_flip is True
    assert grades["u1"] is Grade.B


def test_most_frequent_relation_beats_priority() -> None:
    reps = [rep(L("u1", ["z1"], SUB)), rep(L("u1", ["z1"], REV, True)), rep(L("u1", ["z1"], SUB))]
    assert run(*reps)[0].links[0].relation is SUB


# --------------------------------------------------------------------------- medoid links
def test_links_come_from_the_medoid_replicate() -> None:
    reps = [rep(L("u1", ["z1", "z2"])), rep(L("u1", ["z1"])), rep(L("u1", ["z1", "z2", "z3"]))]
    assert run(*reps)[0].links[0].wit_ids == ("z1", "z2")


def test_medoid_is_chosen_among_replicates_with_the_chosen_relation() -> None:
    reps = [rep(L("u1", ["z1", "z2"], GEN)), rep(L("u1", ["z1"], SUB)), rep(L("u1", ["z1"], SUB))]
    link = run(*reps)[0].links[0]
    assert link.relation is SUB and link.wit_ids == ("z1",)


def test_medoid_tie_takes_the_earliest_replicate() -> None:
    reps = [rep(L("u1", ["z1"], flags={"crossing"})), rep(L("u1", ["z2"])), rep(L("u1", [], NONE))]
    # z1 and z2 readings are both equivalent (2/3); each has Jaccard 0 to the other and 0 to the empty set
    link = run(*reps)[0].links[0]
    assert link.wit_ids == ("z1",)


def test_jaccard_and_majority_helpers() -> None:
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({"a"}), frozenset()) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)
    assert [has_majority(v, 3) for v in (1, 2, 3)] == [False, True, True]
    assert has_majority(1, 1) and has_majority(2, 2) and not has_majority(1, 2)


# --------------------------------------------------------------------------- witness-only
def W(sids, kind, flags=()):
    return Link(None, tuple(sids), kind, flags=frozenset(flags))


def test_witness_only_by_segment_majority() -> None:
    add, note = WitnessOnlyKind.ADDITION, WitnessOnlyKind.TRANSLATOR_NOTE
    reps = [rep(L("u1", ["T:1a1.1"]), W(["T:1a2.1", "T:1a3.1"], add)),
            rep(L("u1", ["T:1a1.1"]), W(["T:1a2.1"], add), W(["T:1a3.n1"], note)),
            rep(L("u1", ["T:1a1.1"]), W(["T:1a2.1"], add))]
    a, grades, _ = run(*reps)
    wo = a.witness_only()
    assert [(x.wit_ids, x.relation) for x in wo] == [(("T:1a2.1",), add)]
    assert grades["+T:1a2.1"] is Grade.B
    assert "+T:1a3.1" not in grades and "+T:1a3.n1" not in grades


def test_witness_only_two_of_three_and_kind_ties() -> None:
    add, para = WitnessOnlyKind.ADDITION, WitnessOnlyKind.PARATEXT
    reps = [rep(W(["T:1a2.1"], para)), rep(W(["T:1a2.1"], add)), rep()]
    a, grades, _ = consensus(reps, {})
    (x,) = a.witness_only()
    assert x.relation is add, "ties follow the WitnessOnlyKind order"
    assert grades["+T:1a2.1"] is Grade.C


def test_segment_linked_by_the_consensus_is_never_witness_only() -> None:
    add = WitnessOnlyKind.ADDITION
    reps = [rep(L("u1", ["T:1a1.1"])), rep(L("u1", ["T:1a1.1"]), W(["T:1a1.1"], add)),
            rep(W(["T:1a1.1"], add), unresolved={"u1": "truncated"})]
    a, _, _ = run(*reps)
    assert a.witness_only() == ()


# --------------------------------------------------------------------------- input checks
def test_consensus_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        consensus([], {"u1": "prose"})
    with pytest.raises(ValueError):
        consensus([rep(L("u9", ["z1"]))], {"u1": "prose"})
    other = Collation(Alignment("s", "sa", "zh", ()))
    with pytest.raises(ValueError):
        consensus([rep(), other], {"u1": "prose"})


def test_output_order_follows_ref_kinds() -> None:
    reps = [rep(L("u2", ["z2"]), L("u1", ["z1"])) for _ in range(3)]
    a, _, _ = consensus(reps, {"u1": "prose", "u2": "prose"})
    assert [link.ref_id for link in a.links] == ["u1", "u2"]


# --------------------------------------------------------------------------- agreement
def test_fleiss_kappa_hand_computed() -> None:
    ratings = [["a", "a", "a"], ["a", "a", "b"], ["b", "b", "b"], ["a", "b", "b"]]
    assert fleiss_kappa(ratings) == pytest.approx(1 / 3)
    assert fleiss_kappa([["a", "a"], ["b", "b"]]) == pytest.approx(1.0)
    assert fleiss_kappa([["a", "a"], ["a", "a"]]) is None
    assert fleiss_kappa([]) is None
    with pytest.raises(ValueError):
        fleiss_kappa([["a"]])


def test_agreement_statistics() -> None:
    kinds = {"u1": "prose", "u2": "prose", "u3": "prose"}
    reps = [
        rep(L("u1", ["z1"]), L("u2", [], NONE), L("u3", ["z3", "z4"])),
        rep(L("u1", ["z1"]), L("u2", [], NONE), L("u3", ["z3"])),
        rep(L("u1", ["z1"]), L("u2", ["z2"]), unresolved={"u3": "truncated"}),
    ]
    stats = agreement(reps, kinds)
    assert stats.replicates == 3 and stats.units == 2
    # items: u1 [nondev x3], u2 [absent, absent, nondev] -> P = (1 + 1/3) / 2; p_e = (4/6)^2 + (2/6)^2
    p_bar, p_e = (1 + 1 / 3) / 2, (4 / 6) ** 2 + (2 / 6) ** 2
    assert stats.fleiss_kappa == pytest.approx((p_bar - p_e) / (1 - p_e))
    # link Jaccard: u1 three pairs of 1.0; u2 pairs (1.0, 0, 0); u3 one pair of 0.5
    assert stats.mean_link_jaccard == pytest.approx((3 + 1 + 0.5) / 7)
    assert stats.class_jaccard["absent"] == pytest.approx((1 + 0 + 0) / 3)
    assert stats.class_jaccard["partial"] is None


def test_agreement_with_one_replicate_has_no_kappa() -> None:
    assert agreement([rep(L("u1", ["z1"]))], {"u1": "prose"}).fleiss_kappa is None
