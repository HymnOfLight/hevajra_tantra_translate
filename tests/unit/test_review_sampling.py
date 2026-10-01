"""review.sampling: strata shared with estimation, census/capped verification, audit, queue, coverage."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from hevajra_matrix.config import ConfigError, load_settings
from hevajra_matrix.core.types import Cell, Grade, Status, Verdict
from hevajra_matrix.review.sampling import (
    AuditSpec,
    ReviewItem,
    ReviewParams,
    VerificationSpec,
    audit_strata,
    coverage,
    draw_audit,
    draw_verification,
    format_coverage,
    largest_remainder,
    machine_strata,
    make_item,
    note_rows,
    stratum_class,
    queue,
    read_plan,
    write_plan,
)
from hevajra_matrix.stats.twophase import stratum_of

W = "zh_wit"


def unit(n: int) -> str:
    return f"D417:{n // 10 + 1}a.{n % 10 + 1}.1"


def cell(n: int, relation: str | None, grade: Grade = Grade.B, status: Status | None = None) -> Cell:
    if status is None:
        status = {"abridged": Status.PARTIAL, "no_counterpart": Status.ABSENT}.get(relation or "", Status.PRESENT)
    return Cell(unit(n), W, status, grade, relation=relation, reason="refused:bio" if status is Status.UNALIGNED else None)


def population(n_absent=5, n_reversal=3, n_abridged=40, n_generalised=20, n_neg_b=200, n_neg_c=60, n_x=4):
    cells, n = [], 0
    for count, rel, grade in ((n_absent, "no_counterpart", Grade.B), (n_reversal, "reversal", Grade.C),
                              (n_abridged, "abridged", Grade.B), (n_generalised, "generalised", Grade.C),
                              (n_neg_b, "equivalent", Grade.B), (n_neg_c, "paraphrase", Grade.C)):
        for _ in range(count):
            c = cell(n, rel, grade)
            cells.append(c if rel != "reversal" else Cell(c.unit_id, W, c.status, c.grade, relation=rel,
                                                            polarity_flip=True))
            n += 1
    for _ in range(n_x):
        cells.append(Cell(unit(n), W, Status.UNALIGNED, Grade.X, reason="no_majority"))
        n += 1
    cells.append(Cell("+T0892:0601c01.2", W, Status.NA, Grade.B, relation="addition", wit_ids=("T0892:0601c01.2",)))
    return cells


def topics_for(cells, every: int = 3) -> dict[str, str]:
    return {c.unit_id: ("sensitive" if i % every == 0 else "neutral") for i, c in enumerate(cells)}


SPEC = VerificationSpec(census_classes=("absent", "reversal", "substitution", "category_name_omitted", "relocated",
                                        "witness_only", "grade_x"),
                        sampled_classes=("abridged", "generalised", "transliterated_prose"),
                        cap_units=450, sampled_fraction=0.5, seed=7)
KINDS: dict[str, str] = {}              # every test unit is a verse unless stated otherwise
AUDIT = AuditSpec(total=100, min_per_stratum=10, grade_c_oversample=2.0, seed=11)


# --------------------------------------------------------------------------- strata
def test_strata_are_those_of_the_estimator():
    cells = population()
    topics = topics_for(cells)
    strata = machine_strata(cells, topics, KINDS)
    for c in cells:
        if not c.unit_id.startswith("+"):
            assert strata[c.unit_id] == stratum_of(c, topics[c.unit_id])
    assert strata["+T0892:0601c01.2"] == "pos:witness_only"


def test_transliterated_mantra_is_negative_but_prose_is_a_positive():
    mantra, prose = cell(0, "transliterated"), cell(1, "transliterated")
    strata = machine_strata([mantra, prose], {}, {mantra.unit_id: "mantra", prose.unit_id: "prose"})
    assert strata == {mantra.unit_id: "neg:B:other", prose.unit_id: "pos:transliterated_prose:other"}


def test_grade_a_cells_are_refused():
    with pytest.raises(ValueError, match="machine cells"):
        machine_strata([cell(0, "equivalent", Grade.A)], {}, {})


# --------------------------------------------------------------------------- verification
def test_everything_is_taken_when_it_fits_under_the_cap():
    cells = population()
    items = draw_verification(cells, topics_for(cells), SPEC, ref_kinds=KINDS)
    by_task = Counter(i.task for i in items)
    assert by_task == {"verify": 5 + 3 + 40 + 20 + 1, "resolve": 4}
    assert all(i.inclusion_prob == 1.0 for i in items)
    assert {i.stratum for i in items if i.task == "resolve"} == {"unresolved"}


def test_census_classes_are_never_capped():
    cells = population(n_absent=30)
    spec = VerificationSpec(SPEC.census_classes, SPEC.sampled_classes, cap_units=10, sampled_fraction=0.5, seed=1)
    items = draw_verification(cells, topics_for(cells), spec, ref_kinds=KINDS)
    census = [i for i in items if i.stratum.startswith(("pos:absent", "pos:reversal", "pos:witness_only", "unresolved"))]
    assert len(census) == 30 + 3 + 1 + 4                       # 38 > cap 10, all kept
    assert all(i.inclusion_prob == 1.0 for i in census)
    # no room left under the cap, but every sampled stratum keeps its floor min(|U_h|, 20)
    pool = Counter(machine_strata(cells, topics_for(cells), KINDS).values())
    sampled = Counter(i.stratum for i in items if "abridged" in i.stratum or "generalised" in i.stratum)
    assert sampled and all(n == min(pool[s], 20) for s, n in sampled.items())
    assert set(sampled) == {s for s in pool if "abridged" in s or "generalised" in s}


def test_sampled_classes_at_the_preregistered_fraction():
    cells = population()
    spec = VerificationSpec(SPEC.census_classes, SPEC.sampled_classes, cap_units=60, sampled_fraction=0.5, seed=3,
                            min_per_sampled_stratum=0)       # the fraction alone, without the floor
    topics = topics_for(cells)
    items = draw_verification(cells, topics, spec, ref_kinds=KINDS)
    strata = machine_strata(cells, topics, KINDS)
    pool = Counter(strata.values())
    drawn = Counter(i.stratum for i in items if i.inclusion_prob < 1 or "abridged" in i.stratum)
    for stratum, n in drawn.items():
        assert n == -(-pool[stratum] // 2)                      # ceil(0.5 * |U_h|)
        assert all(i.inclusion_prob == pytest.approx(n / pool[stratum]) for i in items if i.stratum == stratum)


def test_fraction_is_shrunk_to_the_room_under_the_cap():
    cells = population()
    spec = VerificationSpec(SPEC.census_classes, SPEC.sampled_classes, cap_units=13 + 10, sampled_fraction=0.5, seed=3,
                            min_per_sampled_stratum=0)       # the room share alone, without the floor
    items = draw_verification(cells, topics_for(cells), spec, ref_kinds=KINDS)
    sampled = [i for i in items if i.priority == 8]
    assert len(sampled) == 10
    assert len(items) == 23
    floored = draw_verification(cells, topics_for(cells), replace(spec, min_per_sampled_stratum=20), ref_kinds=KINDS)
    assert len(floored) > 23, "the room share falls below the floor of 20 per sampled stratum"


def test_draw_is_reproducible_from_the_seed():
    cells = population()
    topics = topics_for(cells)
    spec = VerificationSpec(SPEC.census_classes, SPEC.sampled_classes, cap_units=40, sampled_fraction=0.5, seed=3)
    a = draw_verification(cells, topics, spec, ref_kinds=KINDS)
    assert a == draw_verification(list(reversed(cells)), topics, spec, ref_kinds=KINDS)
    assert a != draw_verification(cells, topics, spec, seed=4, ref_kinds=KINDS)


def test_verified_units_leave_the_pool():
    cells = population()
    topics = topics_for(cells)
    done = {c.unit_id for c in cells if c.relation == "no_counterpart"}
    items = draw_verification(cells, topics, SPEC, ref_kinds=KINDS, verified=done)
    assert not done & {i.unit_id for i in items}


def test_sampled_strata_never_get_zero_items_when_the_census_floods_the_cap():
    # ~510 orphan rows filled the census and left the sampled classes no room at all:
    # their strata had no phase-2 verdict and E1 was not estimable.
    cells = population(n_absent=500)
    items = draw_verification(cells, topics_for(cells), SPEC, ref_kinds=KINDS)
    sampled = Counter(i.stratum for i in items if i.priority == 8)
    pool = Counter(machine_strata(cells, topics_for(cells), KINDS).values())
    assert {s for s in pool if stratum_class(s) in SPEC.sampled_classes} == set(sampled)
    assert all(n == min(pool[s], SPEC.min_per_sampled_stratum) for s, n in sampled.items())
    assert all(i.inclusion_prob == pytest.approx(sampled[i.stratum] / pool[i.stratum]) for i in items if i.priority == 8)


def test_witness_only_claims_on_ingest_notes_are_note_rows():
    note = Cell("+T0892:0601c01.n1", W, Status.NA, Grade.B, relation="translator_note", wit_ids=("T0892:0601c01.n1",))
    content = Cell("+T0892:0601c01.2", W, Status.NA, Grade.B, relation="addition", wit_ids=("T0892:0601c01.2",))
    kinds = {"T0892:0601c01.n1": "note", "T0892:0601c01.2": "prose"}
    assert note_rows([note, content, cell(0, "abridged")], kinds) == {note.unit_id}


def test_unknown_positive_class_is_an_error():
    spec = VerificationSpec(("absent",), ("abridged",), 450, 0.5, 1)
    cells = population()
    with pytest.raises(ValueError, match="neither a census nor a sampled class"):
        draw_verification(cells, topics_for(cells), spec, ref_kinds=KINDS)


# --------------------------------------------------------------------------- audit
def test_audit_allocation():
    cells = population()
    topics = topics_for(cells)
    items = draw_audit(cells, topics, AUDIT, ref_kinds=KINDS)
    counts = Counter(i.stratum for i in items)
    assert set(counts) == set(audit_strata())
    assert sum(counts.values()) == AUDIT.total
    assert all(n >= AUDIT.min_per_stratum for n in counts.values())
    pool = Counter(machine_strata(cells, topics, KINDS).values())
    # Grade C is oversampled: its sampling fraction exceeds grade B's within the same topic level.
    for topic in ("sensitive", "other"):
        frac_c = counts[f"neg:C:{topic}"] / pool[f"neg:C:{topic}"]
        frac_b = counts[f"neg:B:{topic}"] / pool[f"neg:B:{topic}"]
        assert frac_c > frac_b
    for i in items:
        assert i.task == "audit" and i.inclusion_prob == pytest.approx(counts[i.stratum] / pool[i.stratum])
    assert items == draw_audit(cells, topics, AUDIT, ref_kinds=KINDS)


def test_small_audit_strata_are_taken_whole():
    cells = population(n_neg_c=6)
    items = draw_audit(cells, topics_for(cells), AUDIT, ref_kinds=KINDS)
    c_items = [i for i in items if i.stratum.startswith("neg:C:")]
    assert len(c_items) == 6 and all(i.inclusion_prob == 1.0 for i in c_items)
    assert len(items) == AUDIT.total


def test_largest_remainder():
    assert largest_remainder(10, {"a": 1.0, "b": 1.0, "c": 2.0}, {"a": 9, "b": 9, "c": 9}) == {"a": 3, "b": 2, "c": 5}
    assert largest_remainder(10, {"a": 1.0, "b": 9.0}, {"a": 9, "b": 2}) == {"a": 8, "b": 2}
    assert largest_remainder(5, {"a": 1.0}, {"a": 3}) == {"a": 3}
    assert largest_remainder(0, {"a": 1.0}, {"a": 3}) == {"a": 0}


# --------------------------------------------------------------------------- queue and coverage
PARAMS = ReviewParams(minutes={"gold": 2.5, "verify": 2.5, "audit": 2.0, "resolve": 3.0, "topics": 0.15},
                      competence={"gold": ("bo", "zh"), "verify": ("bo", "zh"), "audit": ("bo", "zh"),
                                  "resolve": ("bo", "zh"), "topics": ("bo",)},
                      quote_max_chars={"zh": 30, "bo": 60})


def test_queue_respects_priority_minutes_and_competence():
    items = [make_item("audit", unit(1), "neg:B:other"), make_item("verify", unit(2), "pos:abridged:other", 0.5, 8),
             make_item("topics", unit(3)), make_item("resolve", unit(4), "unresolved"),
             make_item("verify", unit(5), "pos:absent:other", 1.0, 6), make_item("gold", unit(6))]
    q = queue(items, budget_minutes=8.0, competence={"bo", "zh"}, params=PARAMS)
    assert [i.task for i in q] == ["gold", "topics", "resolve"]          # 2.5 + 0.15 + 3.0; verify (2.5) no longer fits
    assert [i.task for i in queue(items, 100, {"bo"}, PARAMS)] == ["topics"]
    full = queue(items, 100, {"bo", "zh", "sa"}, PARAMS, done={items[5].item_id})
    assert [i.task for i in full] == ["topics", "resolve", "verify", "audit", "verify"]
    assert [i.priority for i in full] == sorted(i.priority for i in full)


def test_queue_needs_configured_tasks():
    with pytest.raises(ConfigError, match="minutes"):
        queue([make_item("reveal", unit(1))], 10, {"bo"}, ReviewParams(competence={"reveal": ("bo",)}))


def test_coverage_counts_blind_and_final():
    items = [make_item("verify", unit(n), "pos:reversal:other") for n in range(3)] + [make_item("audit", unit(9), "neg:C:other")]
    verdicts = [Verdict("b", items[0].item_id, "verify", unit(0), "f", blind_relation="reversal", final_relation="reversal"),
                Verdict("b", items[1].item_id, "verify", unit(1), "f", blind_relation="reversal")]
    cov = coverage(items, verdicts)
    assert cov == {"neg:C:other": {"planned": 1, "blind": 0, "final": 0},
                   "pos:reversal:other": {"planned": 3, "blind": 2, "final": 1}}
    assert format_coverage(cov)[1] == "pos:reversal:other final 1/3 (blind 2)"


def test_plan_round_trip(tmp_path: Path):
    items = [make_item("verify", unit(1), "pos:abridged:other", 0.25, 8), make_item("resolve", unit(2), "unresolved")]
    write_plan(items, tmp_path / "plan_b.csv")
    assert read_plan(tmp_path / "plan_b.csv") == items
    assert isinstance(items[0], ReviewItem) and items[0].item_id == f"verify:{unit(1)}"


# --------------------------------------------------------------------------- configuration
def test_parameters_from_the_repository_config(repo_root: Path):
    s = load_settings(repo_root)
    params = ReviewParams.from_config(s.run)
    assert params.minutes_for("audit") == 2.0 and params.competence_for("topics_second") == {"bo"}
    assert params.quote_max_chars["zh"] == 30
    spec = VerificationSpec.from_config(s.prereg)
    assert "grade_x" in spec.census_classes and spec.cap_units == 450
    assert AuditSpec.from_config(s.prereg).min_per_stratum == 40


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        ReviewParams.from_config({"review": {"minutes": {}, "hours": 3}})
    with pytest.raises(ConfigError, match="both census and sampled"):
        VerificationSpec.from_config({"verification": dict(census_classes=["a"], sampled_classes=["a"], cap_units=1,
                                                           sampled_fraction=0.5, seed=1)})
