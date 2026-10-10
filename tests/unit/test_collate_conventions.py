"""collate.v2 conventions in the code checks and the consensus: V12 (verify), the
convention vote and grades (consensus), and V10 (merge) staying blind to conventions.

Tibetan and Chinese forms come from data/codebook/collate.v2/; the few extra letters
are \\u escapes (tests/unit/test_language_policy.py).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType

import pytest
import yaml

from hevajra_matrix.collate.consensus import BENIGN_FLAGS, agreement, consensus, convention_vote
from hevajra_matrix.collate.merge import compare_overlaps, merge_chapter
from hevajra_matrix.collate.verify import FLAG_OVERLAP_DISAGREEMENT, CheckLexicon, Collation, verify
from hevajra_matrix.core.lexicon import StandingEntry, StandingRenderings
from hevajra_matrix.core.translit import transliteration_charset
from hevajra_matrix.core.types import (
    CONVENTION_FLAGS,
    FLAG_CONVENTION_UNSUPPORTED,
    Alignment,
    Convention,
    Grade,
    Link,
    Relation,
)

from test_collate_collator import _example_window
from test_collate_support import DATA, SOURCE, gold_answer, lexicon, raw, windows

V2_DIR = DATA / "codebook" / "collate.v2"
SHAD, FULL_STOP = "\u0f0d", "\u3002"
PARTICLES = {"s": "\u0f66", "r": "\u0f62", "i": "\u0f60\u0f72"}
NOT_A_PARTICLE = "\u0f51"                       # da: not a fused particle
SR, TN = Convention.STANDING_RENDERING.value, Convention.TRANSCRIBED_NAME.value


@lru_cache(maxsize=1)
def _glossary_doc() -> dict:
    return yaml.safe_load((V2_DIR / "standing_renderings.yaml").read_text(encoding="utf-8"))


def hevajra() -> StandingEntry:
    e = _glossary_doc()["entries"][0]
    return StandingEntry(e["id"], tuple(e["reference_forms"]), tuple(e["witness_renderings"]))


def dakini() -> StandingEntry:
    e = next(x for x in _glossary_doc()["considered_not_listed"] if x["id"] == "dakini")
    return StandingEntry(e["id"], tuple(e["reference_forms"]), tuple(e["witness_renderings_seen"]))


GLOSSARY = StandingRenderings((hevajra(), dakini()))


def check(ref_text: str, wit_text: str, ref_quote: str, wit_quote: str, convention: str = SR,
          relation: str = "equivalent", glossary: StandingRenderings = GLOSSARY) -> Collation:
    """Verify a one-unit answer over a one-line reference and a one-line witness."""
    w = _example_window(1, {"reference": [{"kind": "prose", "text": ref_text}],
                            "witness": [{"kind": "prose", "text": wit_text}]})
    answer = {"units": [{"ref": "r001", "wit": ["z0001"], "relation": relation, "polarity_flip": False,
                         "confidence": "high", "ref_quote": ref_quote, "wit_quote": wit_quote,
                         "convention": convention}], "witness_only": []}
    lex = CheckLexicon(lexicon().negators, lexicon().variants, transliteration_charset(w.text), glossary)
    return verify(raw(w, answer), w, lex, SOURCE)


def flags(result: Collation) -> frozenset[str]:
    assert dict(result.unresolved) == {}, "V12 is never fatal"
    (link,) = result.alignment.links
    return link.flags


# --------------------------------------------------------------------------- V12 standing renderings
def test_v12_listed_reference_form_and_rendering_of_the_same_entry() -> None:
    form, rendering = hevajra().reference_forms[0], hevajra().witness_renderings[0]
    result = check(form + SHAD, rendering + FULL_STOP, form, rendering)
    assert flags(result) == {SR}


@pytest.mark.parametrize("particle", sorted(PARTICLES))
def test_v12_accepts_a_fused_particle_on_the_last_syllable(particle: str) -> None:
    form, rendering = hevajra().reference_forms[2], hevajra().witness_renderings[1]
    fused = form + PARTICLES[particle]
    assert flags(check(fused + SHAD, rendering, fused, rendering)) == {SR}


def test_v12_rejects_other_continuations_and_mid_syllable_starts() -> None:
    form, rendering = hevajra().reference_forms[2], hevajra().witness_renderings[1]
    longer = form + NOT_A_PARTICLE
    assert flags(check(longer + SHAD, rendering, longer, rendering)) == {SR, FLAG_CONVENTION_UNSUPPORTED}
    inside = PARTICLES["s"] + form
    assert flags(check(inside + SHAD, rendering, inside, rendering)) == {SR, FLAG_CONVENTION_UNSUPPORTED}


def test_v12_needs_the_rendering_of_the_same_entry() -> None:
    ref, other_ref = hevajra().reference_forms[0], dakini().reference_forms[0]
    rendering, other = hevajra().witness_renderings[0], dakini().witness_renderings[0]
    unsupported = {SR, FLAG_CONVENTION_UNSUPPORTED}
    assert flags(check(ref, other, ref, other)) == unsupported               # rendering of another entry
    assert flags(check(other_ref, rendering, other_ref, rendering)) == unsupported
    assert flags(check(other_ref, other, other_ref, other)) == {SR}          # the other entry, consistently
    only_hevajra = StandingRenderings((hevajra(),))
    assert flags(check(ref, other, ref, other, glossary=only_hevajra)) == unsupported   # not listed at all


def test_v12_with_no_verified_entry_flags_every_standing_rendering() -> None:
    form, rendering = hevajra().reference_forms[0], hevajra().witness_renderings[0]
    result = check(form, rendering, form, rendering, glossary=StandingRenderings())
    assert flags(result) == {SR, FLAG_CONVENTION_UNSUPPORTED}


def test_v12_folds_variant_forms() -> None:
    form, rendering = hevajra().reference_forms[0], hevajra().witness_renderings[1]
    simplified = rendering.replace("\u525b", "\u521a")        # the simplified form of the last character
    assert simplified != rendering
    assert flags(check(form, simplified, form, simplified)) == {SR}


def test_v12_missing_quote_is_unsupported_but_never_fatal() -> None:
    form, rendering = hevajra().reference_forms[0], hevajra().witness_renderings[0]
    result = check(form, rendering, "", rendering)          # equivalent: V5 lets the ref_quote be empty
    assert flags(result) == {SR, FLAG_CONVENTION_UNSUPPORTED}
    assert result.alignment.links[0].relation is Relation.EQUIVALENT


# --------------------------------------------------------------------------- V12 transcribed names
@lru_cache(maxsize=1)
def _example3() -> dict:
    return yaml.safe_load((V2_DIR / "examples.yaml").read_text(encoding="utf-8"))["examples"][2]


def check_example3(**changes) -> frozenset[str]:
    ex = _example3()
    w = _example_window(3, ex)
    first = {**ex["answer"]["units"][0], **changes}
    answer = {**ex["answer"], "units": [first, *ex["answer"]["units"][1:]]}
    lex = CheckLexicon(lexicon().negators, lexicon().variants, transliteration_charset(w.text))
    result = verify(raw(w, answer), w, lex, SOURCE)
    assert dict(result.unresolved) == {}
    return result.alignment.by_ref()[w.ref_handles["r001"]].flags


def test_v12_transcribed_name_needs_a_transcription_quote() -> None:
    assert check_example3() == {TN}
    meaning = _example3()["witness"][1]["text"][0]       # the verb before the name, rendered by meaning
    assert check_example3(wit_quote=meaning) == {TN, FLAG_CONVENTION_UNSUPPORTED}


def test_v12_transcribed_name_needs_a_reference_quote() -> None:
    assert check_example3(ref_quote="") == {TN, FLAG_CONVENTION_UNSUPPORTED}


def test_v12_no_convention_no_flag() -> None:
    assert check_example3(convention="none") == frozenset()
    w = windows()[0]
    result = verify(raw(w, gold_answer(w)), w, lexicon(), SOURCE)
    assert not any(link.flags & (CONVENTION_FLAGS | {FLAG_CONVENTION_UNSUPPORTED}) for link in result.alignment.links)


# --------------------------------------------------------------------------- consensus
EQ, SUB = Relation.EQUIVALENT, Relation.SUBSTITUTION


def L(wit=("z1",), relation=EQ, flags=(), ref="u1"):
    return Link(ref, tuple(wit), relation, flags=frozenset(flags), source="src")


def rep(*links, name="r1"):
    return Collation(Alignment(f"src:{name}", "bo", "zh", tuple(links)), MappingProxyType({}))


def run(*links):
    a, grades, _ = consensus([rep(x) for x in links], {"u1": "prose"})
    return a.links[0], grades["u1"]


def test_conventions_never_lower_a_grade() -> None:
    assert CONVENTION_FLAGS <= BENIGN_FLAGS and FLAG_CONVENTION_UNSUPPORTED not in BENIGN_FLAGS
    for conv in (SR, TN):
        link, grade = run(L(flags={conv}), L(flags={conv}), L(flags={conv}))
        assert grade is Grade.B and link.flags == {conv}


def test_a_convention_split_keeps_grade_b() -> None:
    link, grade = run(L(flags={SR}), L(flags={SR}), L())
    assert grade is Grade.B and link.flags == {SR}
    link, grade = run(L(flags={SR}), L(), L())
    assert grade is Grade.B and link.flags == frozenset(), "one of three is no majority"


def test_the_convention_is_voted_never_united() -> None:
    link, _ = run(L(flags={SR}), L(flags={TN}), L())
    assert link.flags == frozenset(), "three different readings: none"
    link, _ = run(L(flags={TN}), L(flags={SR}), L(flags={TN}))
    assert link.flags == {TN}, "exactly one convention flag"
    assert convention_vote([L(flags={SR}), None, L(flags={SR})], 3) is Convention.STANDING_RENDERING
    assert convention_vote([L(flags={SR}), None, None], 3) is Convention.NONE


def test_unsupported_counts_only_when_raised_by_the_winning_convention() -> None:
    link, grade = run(L(flags={SR, FLAG_CONVENTION_UNSUPPORTED}), L(flags={SR}), L(flags={SR}))
    assert grade is Grade.C and link.flags == {SR, FLAG_CONVENTION_UNSUPPORTED}
    link, grade = run(L(flags={TN, FLAG_CONVENTION_UNSUPPORTED}), L(), L())
    assert grade is Grade.B and link.flags == frozenset(), "an outvoted claim is dropped"


def test_the_medoid_carries_the_voted_convention() -> None:
    """The most central reading ({z1, z2}) lacks the voted convention, so the medoid is
    taken among the two readings that have it (the earlier on a tie)."""
    link, _ = run(L(("z1",), flags={TN}), L(("z2",), flags={TN}), L(("z1", "z2")))
    assert link.wit_ids == ("z1",) and link.flags == {TN}
    link, _ = run(L(("z1",)), L(("z2",)), L(("z1", "z2"), flags={TN}))
    assert link.wit_ids == ("z1",) and link.flags == frozenset(), "NONE voted: the medoid among the NONE readings"


def test_a_medoid_that_never_claimed_the_voted_convention_is_unsupported() -> None:
    """r1 paraphrase without convention, r2 equivalent and r3 substitution with a standing
    rendering: the class (PRESENT, r1 and r2) and the relation (paraphrase, by priority)
    come from r1, the convention (r2 and r3) does not, so r1's quotes never claimed it."""
    PARA = Relation.PARAPHRASE
    link, grade = run(L(("z1",), relation=PARA), L(("z2",), flags={SR}), L(("z3",), relation=SUB, flags={SR}))
    assert link.relation is PARA and link.wit_ids == ("z1",)
    assert link.flags == {SR, FLAG_CONVENTION_UNSUPPORTED} and grade is Grade.C
    # all three in one class: still a finding, so the grade drops from B to C
    link, grade = run(L(relation=PARA), L(flags={SR}), L(relation=Relation.EXPANDED, flags={SR}))
    assert link.relation is PARA and link.flags == {SR, FLAG_CONVENTION_UNSUPPORTED} and grade is Grade.C
    # a majority reading with the voted relation holds the convention: no finding
    link, grade = run(L(relation=PARA), L(relation=PARA, flags={SR}), L(relation=Relation.EXPANDED, flags={SR}))
    assert link.relation is PARA and link.flags == {SR} and grade is Grade.B
    # NONE voted: nothing to support
    link, grade = run(L(relation=PARA), L(relation=PARA), L(flags={SR}))
    assert link.flags == frozenset() and grade is Grade.B


def test_the_class_vote_ignores_conventions() -> None:
    link, grade = run(L(flags={SR}), L(relation=SUB), L(relation=SUB, flags={SR}))
    assert link.relation is SUB and grade is Grade.C


def test_v1_consensus_is_unchanged_without_conventions() -> None:
    link, grade = run(L(flags={"relocation"}), L(), L(flags={"crossing"}))
    assert link.flags == {"relocation", "crossing"} and grade is Grade.C


def test_convention_kappa_is_descriptive() -> None:
    kinds = {"u1": "prose", "u2": "prose"}
    same = [rep(L(), L(ref="u2", flags={TN})) for _ in range(3)]
    stats = agreement(same, kinds)
    assert stats.convention_kappa == 1.0
    plain = [rep(L(), L(ref="u2")) for _ in range(3)]
    assert agreement(plain, kinds).convention_kappa is None, "a single convention throughout is undefined"


# --------------------------------------------------------------------------- V10
def test_v10_ignores_a_convention_only_difference() -> None:
    ws = [w for w in windows(max_ref_units=4, overlap=2) if w.chapter == "I.1"]
    results = [verify(raw(w, gold_answer(w)), w, lexicon(), SOURCE) for w in ws]
    uid = ws[1].units[0].id
    first = results[1].alignment
    links = tuple(replace(x, flags=x.flags | {TN}) if x.ref_id == uid else x for x in first.links)
    results[1] = replace(results[1], alignment=replace(first, links=links))
    report = compare_overlaps(ws, results)
    assert report.agreed == report.compared == 2
    assert FLAG_OVERLAP_DISAGREEMENT not in merge_chapter(ws, results).alignment.by_ref()[uid].flags
