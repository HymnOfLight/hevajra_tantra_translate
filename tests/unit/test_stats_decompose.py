"""E3 decomposition: ordered exhaustive classes, translator flag, excess statistics, the G4 gate."""

from __future__ import annotations

import random

import pytest

from hevajra_matrix.core.types import Estimate
from hevajra_matrix.stats import decompose as dc
from hevajra_matrix.stats.decompose import UnitEvidence


def ev(uid="u", dev=True, notes=(), mss=None, cow=False, chapter="I.1") -> UnitEvidence:
    mss = {"C": "present", "K": "present"} if mss is None else mss
    return UnitEvidence(uid, dev, frozenset(notes), mss, cow, chapter)


def test_class_order_rules():
    assert dc.classify(ev(notes=["vorlage_statement"], mss={"C": "absent"})) == "attested_vorlage"
    assert dc.classify(ev(mss={"C": "variant"})) == "vorlage_ms"
    assert dc.classify(ev(mss={"C": "absent", "K": "illegible"})) == "vorlage_ms"   # one lacking ms suffices
    assert dc.classify(ev(mss={"C": "present", "K": "not_collated"})) == "insufficient"
    assert dc.classify(ev(cow=None)) == "insufficient"
    assert dc.classify(ev(cow=True)) == "shared"
    assert dc.classify(ev(cow=False)) == "residual"
    assert dc.classify(ev(mss={"C": "present"}), m_min=1) == "residual"
    with pytest.raises(ValueError):
        dc.classify(ev(dev=False))


def test_translator_note_flags_but_never_reduces_residual():
    units = [ev("a", notes=["substitution_instruction"]), ev("b"), ev("c", notes=["substitution_instruction"], cow=True)]
    d = dc.decompose(units, cowitness_revised=False)
    assert d.counts["residual"] == 2
    assert d.translator_flagged == ("a",)
    assert d.counts["shared"] == 1


def test_revised_cowitness_label():
    d = dc.decompose([ev(cow=True)])
    assert "shared" not in d.counts and d.counts["shared_revised"] == 1
    assert list(d.counts) == ["attested_vorlage", "vorlage_ms", "shared_revised", "residual", "insufficient"]


def test_decomposition_property_loop():
    rng = random.Random(20260930)
    readings = ["present", "absent", "variant", "illegible", "not_collated"]
    notes = ["vorlage_statement", "substitution_instruction", "phonetic", "fascicle"]
    for _ in range(300):
        units = []
        for i in range(rng.randint(0, 40)):
            mss = {f"m{k}": rng.choice(readings) for k in range(rng.randint(0, 4))}
            units.append(ev(f"u{i}", rng.random() < 0.4, rng.sample(notes, rng.randint(0, 2)), mss,
                            rng.choice([True, False, None])))
        m_min = rng.randint(1, 3)
        d = dc.decompose(units, m_min, cowitness_revised=rng.random() < 0.5)
        n_dev = sum(u.deviates for u in units)
        assert sum(d.counts.values()) == d.n_deviating == n_dev == len(d.by_unit)
        assert all(d.by_unit[u] == "residual" for u in d.translator_flagged)
        assert set(d.by_unit) == {u.unit_id for u in units if u.deviates}


def test_excess_by_hand():
    dev = {"a": True, "b": True, "c": False, "d": False}
    ind = {"a": True, "b": False, "c": True, "d": False}
    ex = dc.excess(dev, ind)
    assert (ex.n, ex.observed) == (4, 1)
    assert ex.expected == pytest.approx(4 * 0.5 * 0.5)
    assert ex.excess == pytest.approx(0.0)
    assert (ex.case_rate, ex.base_rate) == (0.5, 0.5)
    assert ex.phi == pytest.approx(0.0)
    chapters = {"a": "x", "b": "y", "c": "x", "d": "y"}
    assert dc.excess(dev, ind, chapters).expected == pytest.approx(2 * 0.5 * 1.0 + 2 * 0.5 * 0.0)


def test_excess_fraction():
    assert dc.excess_fraction(0.6, 0.2) == pytest.approx(0.5)
    assert dc.excess_fraction(0.3, 1.0) is None
    assert dc.excess({"a": True}, {"a": True}).phi is None       # no controls


def test_not_estimable_under_current_reference():
    result = dc.e3([ev()], manuscripts=[], reference="D418", cowitness="D418")
    assert isinstance(result, Estimate)
    assert result.not_estimable == "G4: no manuscript column; reference is the co-witness"
    assert result.name == "E3" and result.point is None
    assert dc.not_estimable_reason(["C"], "D418", None) == "G4: no co-witness"
    assert dc.not_estimable_reason(["C"], "SA", "D418") is None


def test_e3_on_synthetic_data_with_an_independent_cowitness():
    rng = random.Random(3)
    units = []
    for i in range(4000):
        lacks = rng.random() < 0.1
        dev = rng.random() < (0.8 if lacks else 0.2)
        mss = {"C": "absent" if lacks else "present", "K": "present"}
        units.append(ev(f"u{i}", dev, mss=mss, cow=rng.random() < 0.3, chapter=f"c{i % 5}"))
    result = dc.e3(units, manuscripts=["C", "K"], reference="SA", cowitness="D418", cowitness_revised=False)
    assert not isinstance(result, Estimate)
    assert result.vorlage.phi == pytest.approx((0.8 * 0.1 / (0.8 * 0.1 + 0.2 * 0.9) - 0.1 * 0.2 / (0.1 * 0.2 + 0.9 * 0.8))
                                               / (1 - 0.1 * 0.2 / (0.1 * 0.2 + 0.9 * 0.8)), abs=0.05)
    assert result.shared.phi == pytest.approx(0.0, abs=0.06)
    assert abs(result.shared.excess) < 3 * result.shared.expected ** 0.5
    phi_v, phi_s = result.estimates()
    assert phi_v.name == "E3.phi_V" and phi_s.point == result.shared.phi


def test_bound_unverified_shared():
    units = [ev("a", cow=True), ev("b", cow=True), ev("c", dev=False, cow=True)]
    bounded = dc.bound_unverified_shared(units, verified={"a"})
    assert [u.cowitness_deviates for u in bounded] == [True, False, True]
    assert dc.decompose(bounded, cowitness_revised=False).counts["shared"] == 1
