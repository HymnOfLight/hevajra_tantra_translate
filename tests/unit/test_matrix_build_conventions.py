"""matrix.build: a human convention (gold or final verdict) becomes the grade-A cell's
convention flag; a human cell replaces a machine convention; legacy "" leaves none."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hevajra_matrix.core.types import Grade, Relation
from hevajra_matrix.matrix.status import cell_deviates, has_convention

from test_matrix_build import REFS, WITS, build, link, verdict


def test_human_conventions_reach_the_cells_with_grade_a():
    u0, u1, u2, u3 = REFS
    links = [link(u.id, (WITS[i].id,), Relation.EQUIVALENT, flags={"transcribed_name"}) for i, u in enumerate(REFS)]
    gold = [replace(verdict(u0, "equivalent", (WITS[0].id,), task="gold", batch="test_w01", final=False),
                    blind_convention="standing_rendering")]
    verdicts = [replace(verdict(u1, "abridged", (WITS[1].id,)), blind_convention="transcribed_name",
                        final_convention="none"),
                replace(verdict(u2, "equivalent", (WITS[2].id,)), final_convention="transcribed_name")]
    cells, _, _ = build(links, gold=gold, verdicts=verdicts)
    c0, c1, c2, c3 = cells
    assert (c0.grade, c0.flags) == (Grade.A, {"standing_rendering"})
    assert (c1.grade, c1.flags) == (Grade.A, frozenset()), "a human none removes the machine's convention"
    assert (c2.grade, c2.flags) == (Grade.A, {"transcribed_name"})
    assert (c3.grade, c3.flags) == (Grade.B, {"transcribed_name"}), "the machine cell keeps its flag"
    assert [has_convention(c.flags) for c in cells] == [True, False, True, True]
    assert cell_deviates(c0, "any") is False and cell_deviates(c0, "any_conv") is True


def test_legacy_and_unrevealed_conventions_add_no_flag():
    u0, u1 = REFS[:2]
    verdicts = [verdict(u0, "equivalent", (WITS[0].id,)),                                 # "" (not annotated)
                replace(verdict(u1, "equivalent", (WITS[1].id,), final=False), blind_convention="transcribed_name")]
    cells, _, _ = build([link(u1.id, (WITS[1].id,), Relation.EQUIVALENT)], verdicts=verdicts)
    assert (cells[0].grade, cells[0].flags) == (Grade.A, frozenset())
    assert (cells[1].grade, cells[1].flags) == (Grade.B, frozenset()), "a blind-only convention is not applied"


def test_an_unknown_convention_names_the_verdict():
    bad = replace(verdict(REFS[0], "equivalent", (WITS[0].id,)), final_convention="name")
    with pytest.raises(ValueError, match="verify verdict verify:D417:1a.1.1 has an unknown convention 'name'"):
        build(verdicts=[bad])
