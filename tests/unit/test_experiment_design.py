"""Over-attribution experiment: item bank, evidence lines and the trial schedule."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from hevajra_matrix.config import ConfigError, load_settings
from hevajra_matrix.core.io import read_csv
from hevajra_matrix.experiments.overattribution.analysis import HUMAN_COLUMNS
from hevajra_matrix.experiments.overattribution.design import (
    CONDITIONS,
    ITEM_COLUMNS,
    DesignError,
    ExperimentParams,
    Item,
    bank_shortfalls,
    double_score_ids,
    ew_versions,
    items_in_phase,
    load_evidence,
    load_items,
    trials,
    validate_items,
)
from hevajra_matrix.experiments.overattribution.score import SCORER_CLASSES

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "data" / "experiments" / "overattribution"


def make_items(n_pairs: int, phase: str = "main") -> list[Item]:
    items = []
    for k in range(n_pairs):
        for arm in ("sensitive", "neutral"):
            items.append(Item(item_id=f"{arm[0]}{k:03d}", arm=arm, unit_ids=(f"D418:{k}a.1.1",), chapter="I.7",
                              pair_id=f"p{k:03d}", omission_origin="real" if k % 3 else "constructed",
                              zh_context_from=f"T0892:0590a{k:02d}.1", zh_context_to=f"T0892:0590a{k:02d}.2",
                              phase=phase))
    return items


def write_items(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ITEM_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in ITEM_COLUMNS})
    return path


def row(item_id: str, arm: str, pair: str, **extra: str) -> dict[str, str]:
    base = {"item_id": item_id, "phase": "main", "arm": arm, "pair_id": pair, "chapter": "II.3",
            "unit_ids": "D418:17b.6.1; D418:17b.6.2", "omission_origin": "real",
            "zh_context_from": "T0892:0596b25.1", "zh_context_to": "T0892:0596b25.2", "synthetic_facts": "true"}
    return {**base, **extra}


# --------------------------------------------------------------------------- committed files
def test_committed_item_bank_and_human_codes_are_header_only():
    assert load_items(EXP_DIR / "items.csv") == []
    with (EXP_DIR / "items.csv").open(encoding="utf-8") as fh:
        assert tuple(next(csv.reader(fh))) == ITEM_COLUMNS
    with (EXP_DIR / "human_codes.csv").open(encoding="utf-8") as fh:
        assert tuple(next(csv.reader(fh))) == HUMAN_COLUMNS
    assert read_csv(EXP_DIR / "human_codes.csv") == []


def test_evidence_lines_are_loaded_and_length_matched():
    evidence = load_evidence(EXP_DIR / "evidence.yaml")
    assert set(evidence) == {"EP", "EW-V", "EW-S"}
    assert evidence["EP"].implies == frozenset()
    for key in ("EW-V", "EW-S"):
        assert evidence[key].implies and evidence[key].implies <= set(SCORER_CLASSES)
    ew_mean = (len(evidence["EW-V"].text) + len(evidence["EW-S"].text)) / 2
    assert abs(len(evidence["EP"].text) - ew_mean) <= 0.25 * ew_mean
    assert evidence["EP"].text.startswith("In the printed edition used here")


def test_evidence_file_must_have_exactly_the_three_lines(tmp_path):
    path = tmp_path / "evidence.yaml"
    path.write_text("lines:\n  EP: {text: x, implies: []}\n", encoding="utf-8")
    with pytest.raises(DesignError, match="exactly"):
        load_evidence(path)


def test_prereg_experiment_section_parses():
    params = ExperimentParams.from_prereg(load_settings(ROOT).prereg)
    assert params.conditions == CONDITIONS and params.replicates == 3 and params.items_per_arm == 60
    with pytest.raises(ConfigError):
        ExperimentParams.from_prereg({"experiment": {"replicate": 3}})


# --------------------------------------------------------------------------- items
def test_load_items_parses_rows(tmp_path):
    path = write_items(tmp_path / "items.csv", [row("s1", "sensitive", "p1"),
                                                row("n1", "neutral", "p1", omission_origin="constructed",
                                                    synthetic_facts="false", phase="main")])
    s1, n1 = load_items(path)
    assert s1.unit_ids == ("D418:17b.6.1", "D418:17b.6.2")
    assert s1.synthetic_facts is True and n1.synthetic_facts is False
    assert n1.omission_origin == "constructed" and s1.arm == "sensitive"


@pytest.mark.parametrize("bad, message", [
    ({"arm": "risky"}, "arm must be one of"),
    ({"omission_origin": "guess"}, "omission_origin"),
    ({"phase": "later"}, "phase"),
    ({"unit_ids": ""}, "unit_ids is empty"),
    ({"zh_context_to": ""}, "zh_context_to is empty"),
    ({"zh_context_to": "T0892:0596b25.1"}, "equals"),
    ({"synthetic_facts": "maybe"}, "synthetic_facts"),
])
def test_load_items_rejects_invalid_rows(tmp_path, bad, message):
    path = write_items(tmp_path / "items.csv", [{**row("s1", "sensitive", "p1"), **bad}, row("n1", "neutral", "p1")])
    with pytest.raises(DesignError, match=message):
        load_items(path)


def test_pairs_need_one_item_per_arm_and_unique_ids():
    items = make_items(2)
    assert validate_items(items) == []
    unpaired = [items[0], items[2], items[3]]
    assert any("pair 'p000'" in e for e in validate_items(unpaired))
    assert any("duplicate item_id" in e for e in validate_items(items + [items[0]]))
    two_sensitive = [items[0], Item(**{**items[1].__dict__, "arm": "sensitive"})]
    assert any("one sensitive and one neutral" in e for e in validate_items(two_sensitive))


def test_items_in_phase_and_bank_shortfalls():
    items = make_items(3) + make_items(1, phase="pilot")
    items = [Item(**{**it.__dict__, "item_id": f"x{i}", "pair_id": f"{it.pair_id}{it.phase}"})
             for i, it in enumerate(items)]
    assert len(items_in_phase(items, "pilot")) == 2
    short = bank_shortfalls(items, ExperimentParams(items_per_arm=3, pilot_items_per_arm=2))
    assert short == ["pilot/sensitive: 1 of 2 items", "pilot/neutral: 1 of 2 items"]


# --------------------------------------------------------------------------- schedule
def test_ew_versions_are_balanced_within_arm_and_seeded():
    items = make_items(21)
    versions = ew_versions(items, seed=7)
    for arm in ("sensitive", "neutral"):
        counts = Counter(v for i, v in versions.items() if i.startswith(arm[0]))
        assert abs(counts["EW-V"] - counts["EW-S"]) <= 1 and sum(counts.values()) == 21
    assert ew_versions(items, seed=7) == versions
    assert ew_versions(list(reversed(items)), seed=7) == versions
    assert ew_versions(items, seed=8) != versions


def test_trials_cover_every_cell_and_are_reproducible():
    items = make_items(10)
    schedule = trials(items, CONDITIONS, replicates=3, seed=20261002)
    assert len(schedule) == 20 * 3 * 3
    assert sorted(t.order for t in schedule) == list(range(len(schedule)))
    cells = Counter((t.item_id, t.condition) for t in schedule)
    assert set(cells.values()) == {3}
    assert len({t.trial_id for t in schedule}) == len(schedule)
    versions = ew_versions(items, 20261002)
    for t in schedule:
        expected = {"E0": "none", "EP": "EP", "EW": versions[t.item_id]}[t.condition]
        assert t.evidence == expected
        assert t.arm == t.item.arm and t.trial_id == f"{t.item_id}:{t.condition}:{t.replicate}"
    assert trials(items, CONDITIONS, 3, 20261002) == schedule
    other = trials(items, CONDITIONS, 3, 1)
    assert [t.trial_id for t in other] != [t.trial_id for t in schedule]


def test_condition_order_is_randomised_not_blocked():
    schedule = sorted(trials(make_items(10), CONDITIONS, 3, 5), key=lambda t: t.order)
    first_third = Counter(t.condition for t in schedule[:len(schedule) // 3])
    assert set(first_third) == set(CONDITIONS)


@pytest.mark.parametrize("conditions", [("E0", "E9"), ("E0", "E0"), ()])
def test_trials_reject_bad_conditions(conditions):
    with pytest.raises(DesignError):
        trials(make_items(1), conditions, 1, 0)


def test_trials_reject_invalid_items_and_replicates():
    with pytest.raises(DesignError):
        trials(make_items(1)[:1], CONDITIONS, 1, 0)
    with pytest.raises(DesignError):
        trials(make_items(1), CONDITIONS, 0, 0)


def test_double_score_selection():
    ids = [f"t{i}" for i in range(100)]
    chosen = double_score_ids(ids, 0.2, seed=3)
    assert len(chosen) == 20 and chosen <= set(ids)
    assert double_score_ids(reversed(ids), 0.2, seed=3) == chosen
    with pytest.raises(DesignError):
        double_score_ids(ids, 1.5, seed=3)
