"""Witness distance and UPGMA over the Cell type."""

from __future__ import annotations

import pytest

from hevajra_matrix.core.types import Cell, Grade, Status
from hevajra_matrix.stats import distance as ds


def c(unit, witness, relation="equivalent", status=Status.PRESENT, **kw) -> Cell:
    return Cell(unit_id=unit, witness=witness, status=status, grade=Grade.B, relation=relation, **kw)


CELLS = [
    c("u1", "A"), c("u2", "A"), c("u3", "A", "no_counterpart", Status.ABSENT),
    c("u1", "B"), c("u2", "B"), c("u3", "B", "no_counterpart", Status.ABSENT),
    c("u1", "C", "substitution"), c("u2", "C"), c("u3", "C"),
    c("u4", "A"), c("u4", "B", None, Status.UNALIGNED), c("+x", "A", None, Status.NA),
]


def test_witness_distance():
    assert ds.witness_distance(CELLS, "A", "B") == 0.0
    assert ds.witness_distance(CELLS, "A", "C") == pytest.approx(2 / 3)
    assert ds.witness_distance(CELLS, "A", "Z") is None


def test_dims_and_weights():
    cells = [c("u", "A", d_len=0.5, d_ord=None), c("u", "B", d_len=0.1, d_ord=3.0)]
    assert ds.witness_distance(cells, "A", "B") == pytest.approx(0.4)
    assert ds.witness_distance(cells, "A", "B", weights={"d_len": 2.0}) == pytest.approx(0.8)


def test_transliterated_mantra_agrees_with_equivalent():
    cells = [c("m", "A", "transliterated"), c("m", "B")]
    assert ds.witness_distance(cells, "A", "B", ref_kinds={"m": "mantra"}) == 0.0
    assert ds.witness_distance(cells, "A", "B") == 1.0


def test_distance_matrix_needs_three_witnesses():
    with pytest.raises(ValueError, match=">= 3"):
        ds.distance_matrix(CELLS, ["A", "B"])
    labels, rows = ds.distance_matrix(CELLS, ["A", "B", "C"])
    assert labels == ["A", "B", "C"]
    assert rows[0][2] == rows[2][0] == pytest.approx(2 / 3) and rows[1][1] == 0.0


def test_upgma_joins_closest_pair_first():
    newick = ds.average_linkage_newick(["A", "B", "C"], [[0, 0.1, 0.8], [0.1, 0, 0.6], [0.8, 0.6, 0]])
    assert newick == "(C:0.3500,(A:0.0500,B:0.0500):0.3000);"   # d(AB, C) = (0.8 + 0.6) / 2
    assert ds.average_linkage_newick(["A", "B", "C"], [[0, None, 1], [None, 0, 1], [1, 1, 0]]).count("(") == 2
