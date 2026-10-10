"""D_any_conv (collate.v2 conventions) in the statistics: E1 only, plugged in, never primary.

E1_any_conv uses the final-column draws of E1_any and adds every unit whose built cell carries
a convention. Every consumer that cannot know conventions (E3, E4, the MDE, distances) refuses
``any_conv`` instead of computing D_any under that name.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hevajra_matrix.core.types import FLAG_CONVENTION_UNSUPPORTED, Cell, Grade, OutcomeClass, Status, Verdict
from hevajra_matrix.matrix.status import PRIMARY_OUTCOMES
from hevajra_matrix.pipeline import e3 as e3_stage
from hevajra_matrix.pipeline import e4 as e4_stage
from hevajra_matrix.stats import contrast, distance
from hevajra_matrix.stats import twophase as tp

TN, SR = "transcribed_name", "standing_rendering"


def cell(unit: str, relation: str | None = "equivalent", grade: Grade = Grade.B, status: Status | None = None,
         flags: tuple[str, ...] = (), **kw) -> Cell:
    if status is None:
        status = {"no_counterpart": Status.ABSENT, "abridged": Status.PARTIAL}.get(relation or "", Status.PRESENT)
    return Cell(unit_id=unit, witness="T0892", status=status, grade=grade, relation=relation,
                flags=frozenset(flags), **kw)


def verdict(unit: str, final: str, task: str = "audit") -> Verdict:
    return Verdict(batch_id="b1", item_id=unit, task=task, unit_id=unit, fingerprint="f", final_relation=final)


# --------------------------------------------------------------------------- indicator and guards
def test_indicator_any_conv_needs_the_convention():
    assert tp.indicator(OutcomeClass.NONDEV, "any_conv", True) is True
    assert tp.indicator(OutcomeClass.NONDEV, "any_conv", False) is False
    assert tp.indicator(OutcomeClass.ABSENT, "any_conv", False) is True
    assert tp.indicator(OutcomeClass.NONDEV, "any", True) is False, "a convention never enters D_any"
    assert tp.indicator(OutcomeClass.NONDEV, "cov", True) is False
    with pytest.raises(ValueError, match="convention"):
        tp.indicator(OutcomeClass.NONDEV, "any_conv")
    with pytest.raises(ValueError):
        tp.indicator(OutcomeClass.NONDEV, "conv")


def test_require_primary_accepts_only_the_preregistrable_outcomes():
    assert PRIMARY_OUTCOMES == ("any", "cov")
    assert tp.require_primary("any", "E4") == "any" and tp.require_primary("cov", "E4") == "cov"
    for outcome in ("any_conv", "absent", ""):
        with pytest.raises(ValueError, match="E4 uses the primary outcome"):
            tp.require_primary(outcome, "E4")


def test_convention_flags_do_not_move_a_unit_between_strata():
    for flags, grade in (((), Grade.B), ((TN,), Grade.B), ((SR,), Grade.B), ((TN, FLAG_CONVENTION_UNSUPPORTED), Grade.C)):
        c = cell("u", "paraphrase", grade, flags=flags)
        assert tp.stratum_of(c, "other") == f"neg:{grade}:other"
    assert tp.stratum_of(cell("u", "no_counterpart", flags=(TN,)), "other") == "pos:absent:other"


# --------------------------------------------------------------------------- E1_any_conv
def _matrix():
    """Two verified units (one human convention, one human none over a machine convention),
    and four unverified negatives, two of them with a machine-majority convention."""
    built = [cell("v1", "paraphrase", Grade.A, flags=(TN,), source="verdict:b1"),
             cell("v2", "equivalent", Grade.A, source="verdict:b1"),
             cell("n0", "paraphrase", flags=(SR,)), cell("n1", "equivalent", Grade.C, flags=(TN, FLAG_CONVENTION_UNSUPPORTED)),
             cell("n2"), cell("n3")]
    strata = {c.unit_id: "neg:B:other" for c in built}
    verdicts = [verdict("v1", "paraphrase"), verdict("v2", "equivalent")]
    return built, strata, verdicts


def test_unit_conventions_read_the_built_matrix():
    built, _, _ = _matrix()
    assert tp.unit_conventions(built) == {"v1": True, "v2": False, "n0": True, "n1": True, "n2": False, "n3": False}


def test_e1_any_conv_adds_the_plugged_in_conventions_to_the_same_draws():
    built, strata, verdicts = _matrix()
    draws = tp.draws(built, verdicts, strata, 200, seed=7)
    conv = tp.unit_conventions(built)
    any_ = tp.prevalence(draws, "any", conventions=conv)
    any_conv = tp.prevalence(draws, "any_conv", name="E1_any_conv", conventions=conv)
    expected = [sum(cls is not OutcomeClass.NONDEV or conv[u] for u, cls in d.items()) / len(d) for d in draws]
    assert any_conv.point == pytest.approx(sum(expected) / len(expected))
    # per draw, D_any_conv - D_any is the share of convention units drawn NONDEV; the verified
    # v1 (human convention, NONDEV) is one of them in every draw, so the gap is at least 1/6
    gaps = [e - sum(cls is not OutcomeClass.NONDEV for cls in d.values()) / len(d) for e, d in zip(expected, draws)]
    assert min(gaps) >= 1 / 6 - 1e-12 and max(gaps) <= 3 / 6 + 1e-12
    assert any_conv.point > any_.point
    assert any_conv.name == "E1_any_conv" and any_conv.n == any_.n == 6


def test_e1_any_conv_equals_e1_any_without_conventions():
    # a collate.v1 run: no cell carries a convention flag
    built, strata, verdicts = _matrix()
    v1 = [Cell(c.unit_id, c.witness, c.status, c.grade, relation=c.relation, source=c.source) for c in built]
    draws = tp.draws(v1, verdicts, strata, 100, seed=3)
    conv = tp.unit_conventions(v1)
    a = tp.prevalence(draws, "any", name="E1", conventions=conv)
    b = tp.prevalence(draws, "any_conv", name="E1", conventions=conv)
    assert (a.point, a.lo, a.hi, a.n, a.sources) == (b.point, b.lo, b.hi, b.n, b.sources)


def test_e1_any_conv_refuses_missing_conventions():
    draws = [{"a": OutcomeClass.NONDEV}]
    with pytest.raises(ValueError, match="convention"):
        tp.prevalence(draws, "any_conv")
    with pytest.raises(KeyError):
        tp.prevalence(draws, "any_conv", conventions={})


def test_manski_any_conv_reads_the_cell_flags():
    cells = [cell("a", "no_counterpart"), cell("b", flags=(TN,)), cell("c"), cell("d", "paraphrase", flags=(SR,)),
             cell("u", None, Grade.X, Status.UNALIGNED), cell("l", None, Grade.B, Status.LACUNA, flags=(TN,))]
    assert tp.manski(cells, "any") == (pytest.approx(1 / 5), pytest.approx(2 / 5))
    assert tp.manski(cells, "any_conv") == (pytest.approx(3 / 5), pytest.approx(4 / 5))
    assert tp.manski(cells, "cov") == tp.manski([cell(c.unit_id, c.relation, c.grade, c.status) for c in cells], "cov")


# --------------------------------------------------------------------------- E2-E4, distances: primary only
def test_e4_functions_refuse_any_conv():
    draws = [{"a": OutcomeClass.NONDEV, "b": OutcomeClass.ABSENT}]
    with pytest.raises(ValueError, match="primary outcome"):
        tp.unit_means(draws, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):
        contrast.delta(draws, {"a": True, "b": False}, {"a": "s", "b": "s"}, {"a": "c", "b": "c"}, 0, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):     # refused even with nothing to compute
        contrast.delta([], {}, {}, {}, 0, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):
        contrast.misclassification_table([], {}, {}, outcome="any_conv")
    assert tp.unit_means(draws, "cov") == {"a": 0.0, "b": 1.0}


def test_pipeline_e3_e4_refuse_any_conv():
    with pytest.raises(ValueError, match="E3 uses the primary outcome"):
        e3_stage.unit_evidence(None, [], None, "any_conv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="primary outcome"):
        e4_stage.machine_indicator([cell("a", flags=(TN,))], {}, "any_conv")
    with pytest.raises(ValueError, match="primary outcome"):
        e4_stage.mde_on_labels(SimpleNamespace(), [], {}, "any_conv", seed=0)  # type: ignore[arg-type]
    assert e4_stage.machine_indicator([cell("a", flags=(TN,)), cell("b", "abridged")], {}, "any") == {"a": 0.0, "b": 1.0}


def test_distance_ignores_conventions_and_refuses_unknown_weights():
    cells = [Cell("u1", w, Status.PRESENT, Grade.B, relation="equivalent", flags=frozenset(f))
             for w, f in (("w1", ()), ("w2", (TN,)))]
    assert distance.witness_distance(cells, "w1", "w2") == 0.0, "conventions are not a distance dimension"
    with pytest.raises(ValueError, match="unknown distance weight"):
        distance.witness_distance(cells, "w1", "w2", weights={"convention": 1.0})
    assert distance.witness_distance(cells, "w1", "w2", weights={"outcome": 2.0, "d_len": 0.5}) == 0.0
