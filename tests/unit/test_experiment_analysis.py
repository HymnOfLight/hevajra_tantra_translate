"""Over-attribution experiment: statistics, refusal bounds, scorer agreement and recovery
of planted H1/H2 effects from synthetic FakeClient responses."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from hevajra_matrix.config import load_settings
from hevajra_matrix.core.io import read_csv
from hevajra_matrix.experiments.overattribution.analysis import (
    HUMAN_COLUMNS,
    AnalysisError,
    AnalysisParams,
    HumanCode,
    analyse,
    consensus_codes,
    holm,
    human_sample,
    item_means,
    load_human_codes,
    paired_test,
    scorer_agreement,
    sheet_id,
    two_sample_test,
    write_coding_sheet,
    write_sample,
)
from hevajra_matrix.experiments.overattribution.human import allocate
from hevajra_matrix.experiments.overattribution.design import CONDITIONS, Item, load_evidence, trials
from hevajra_matrix.experiments.overattribution.run import ExperimentTaskSettings, ItemText, run_trials
from hevajra_matrix.experiments.overattribution.score import (
    SCORER_CLASSES,
    Coding,
    TrialOutcome,
    load_motive_lexicon,
)
from hevajra_matrix.llm.fake import FakeClient, refusal_response

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVIDENCE = load_evidence(DATA / "experiments" / "overattribution" / "evidence.yaml")
LEXICON = load_motive_lexicon(DATA)
LLM = load_settings(ROOT).llm
PARAMS = AnalysisParams(n_boot=1000, n_perm=1000, seed=20261002)

MOTIVE = "The translator deliberately left the passage out because of what it says."
SOURCE = "The translator's source manuscript lacked the passage."


def outcome(item: str, arm: str, condition: str, rep: str = "r1", y: bool | None = False, status: str = "ok",
            evidence: str | None = None, **extra) -> TrialOutcome:
    measured = status == "ok" and y is not None
    return TrialOutcome(trial_id=f"{item}:{condition}:{rep}", item_id=item, arm=arm, condition=condition,
                        evidence=evidence or {"E0": "none", "EP": "EP", "EW": "EW-V"}[condition], replicate=rep,
                        omission_origin=extra.pop("origin", "real"), status=status,  # type: ignore[arg-type]
                        scorer_status="ok" if measured else "not_run", y_over=y if measured else None,
                        y_any=y if measured else None, primary=extra.pop("primary", "content_motive" if y else "source_text"),
                        **extra)


# --------------------------------------------------------------------------- small statistics
def test_holm_hand_computed():
    assert holm({"H1": 0.01, "H2": 0.04}) == {"H1": 0.02, "H2": 0.04}
    assert holm({"H1": 0.03, "H2": 0.02}) == {"H1": 0.04, "H2": 0.04}
    assert holm({"H1": 0.6, "H2": None}) == {"H1": 0.6, "H2": None}
    assert holm({"H1": 0.7, "H2": 0.8}) == {"H1": 1.0, "H2": 1.0}


def test_paired_test_detects_consistent_differences():
    est, lo, hi, p = paired_test([0.5, 0.4, 0.6, 0.3, 0.5, 0.7, 0.2, 0.4], 1000, 1000, random.Random(1))
    assert est == pytest.approx(0.45) and 0 < lo < est < hi and p < 0.01
    est, lo, hi, p = paired_test([0.1, -0.1, 0.2, -0.2], 0, 0, random.Random(1))
    assert est == pytest.approx(0.0) and lo is hi is p is None


def test_two_sample_test_null_and_effect():
    same = two_sample_test([0.2, 0.4, 0.3, 0.5], [0.3, 0.2, 0.5, 0.4], 1000, 1000, random.Random(2))
    assert same[1] < 0 < same[2] and same[3] > 0.3
    diff = two_sample_test([0.9, 0.8, 1.0, 0.7, 0.9], [0.1, 0.2, 0.0, 0.3, 0.1], 1000, 1000, random.Random(2))
    assert diff[0] == pytest.approx(0.72) and diff[1] > 0 and diff[3] < 0.05


def test_params_from_prereg():
    params = AnalysisParams.from_prereg(load_settings(ROOT).prereg)
    assert params.seed == 20261002 and params.min_scorer_kappa == 0.80 and params.n_boot == 10000


# --------------------------------------------------------------------------- item means and refusals
def test_replicates_are_averaged_within_item_and_condition():
    rows = [outcome("s1", "sensitive", "E0", "r1", True), outcome("s1", "sensitive", "E0", "r2", False),
            outcome("s1", "sensitive", "E0", "r3", True), outcome("s1", "sensitive", "EW", "r1", False),
            outcome("s1", "sensitive", "EW", "r2", None, status="refused")]
    means = item_means(rows)
    assert means == {("s1", "E0"): pytest.approx(2 / 3), ("s1", "EW"): 0.0}
    assert item_means(rows, refusals="as1")[("s1", "EW")] == 0.5
    assert item_means(rows, refusals="as0")[("s1", "EW")] == 0.0


def test_refusal_bounds_and_cell_rates():
    rows = []
    for k in range(6):
        for arm in ("sensitive", "neutral"):
            item = f"{arm[0]}{k}"
            rows.append(outcome(item, arm, "E0", y=arm == "sensitive"))
            rows.append(outcome(item, arm, "EP", y=False))
            rows.append(outcome(item, arm, "EW", "r1", y=False))
            status = "refused" if arm == "sensitive" else "ok"
            rows.append(outcome(item, arm, "EW", "r2", y=None if status == "refused" else False, status=status))
    results = analyse(rows, PARAMS)
    assert results.test("H1").estimate == pytest.approx(1.0)
    # Manski: EW refusals coded 1 (lower) or 0 (upper); E0 has none
    assert results.refusal_bounds["H1"] == (pytest.approx(0.5), pytest.approx(1.0))
    assert results.refusal_uniform["H1"] == (pytest.approx(1.0), pytest.approx(0.5))
    cell = next(c for c in results.cells if (c.arm, c.condition) == ("sensitive", "EW"))
    assert cell.counts["refused"] == 6 and cell.counts["measured"] == 6 and cell.rates["refused"] == 0.5
    neutral = next(c for c in results.cells if (c.arm, c.condition) == ("neutral", "EW"))
    assert neutral.rates["refused"] == 0.0


def test_refusal_bounds_use_opposite_codings_on_the_two_sides_of_a_contrast():
    """The review's example: half of E0 and of EW refused in the sensitive arm. Uniform
    codings give (0.20, 0.20); the real range is -0.30 .. 0.70."""
    rows = []
    for k in range(10):
        s, n = f"s{k}", f"n{k}"
        rows += [outcome(s, "sensitive", "E0", "r1", y=k < 6), outcome(s, "sensitive", "E0", "r2", None, status="refused"),
                 outcome(s, "sensitive", "EW", "r1", y=k < 2), outcome(s, "sensitive", "EW", "r2", None, status="refused"),
                 outcome(n, "neutral", "E0", y=False), outcome(n, "neutral", "EW", y=False)]
    results = analyse(rows, AnalysisParams(n_boot=0, n_perm=10, seed=1))
    assert results.refusal_uniform["H1"] == (pytest.approx(0.2), pytest.approx(0.2))
    assert results.refusal_bounds["H1"] == (pytest.approx(-0.3), pytest.approx(0.7))
    lo, hi = results.refusal_bounds["H2"]                  # E0 sensitive 0.3..0.8 minus neutral 0
    assert (lo, hi) == (pytest.approx(0.3), pytest.approx(0.8))


def test_not_estimable_with_too_few_items():
    results = analyse([outcome("s1", "sensitive", "E0", y=True)], PARAMS)
    assert all(t.estimate is None and t.not_estimable for t in results.tests)
    assert results.test("H1").p_holm is None


# --------------------------------------------------------------------------- human codes and scorer kappa
def write_codes(path: Path, rows: list[dict[str, str]]) -> Path:
    lines = [",".join(HUMAN_COLUMNS)]
    for r in rows:
        lines.append(",".join(r.get(c, "") for c in HUMAN_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def code_row(rid: str, coder: str, motive: bool, primary: str | None = None) -> dict[str, str]:
    stances = {c: "not_mentioned" for c in SCORER_CLASSES}
    stances["content_motive" if motive else "source_text"] = "asserted"
    return {"response_id": rid, "coder": coder, **stances,
            "primary": primary or ("content_motive" if motive else "source_text"), "disputes_premise": "false"}


def test_load_human_codes_and_consensus(tmp_path):
    path = write_codes(tmp_path / "codes.csv", [
        code_row("t1", "A", True), code_row("t1", "B", True),
        code_row("t2", "A", True), code_row("t2", "B", False),
        code_row("t3", "A", True), code_row("t3", "B", False), code_row("t3", "consensus", False),
    ])
    codes = load_human_codes(path)
    assert len(codes) == 7
    consensus = consensus_codes(codes)
    assert set(consensus) == {"t1", "t3"}
    assert consensus["t1"].y_over is True and consensus["t3"].y_over is False


def test_load_human_codes_rejects_invalid_rows(tmp_path):
    bad = code_row("t1", "A", True)
    bad["content_motive"] = "maybe"
    with pytest.raises(AnalysisError, match="unknown stance"):
        load_human_codes(write_codes(tmp_path / "c.csv", [bad]))
    bad = code_row("t1", "", True)
    with pytest.raises(AnalysisError, match="coder"):
        load_human_codes(write_codes(tmp_path / "c.csv", [bad]))


def test_scorer_agreement_threshold_flag():
    rows = [outcome(f"s{k}", "sensitive", "E0", y=k % 2 == 0) for k in range(20)]
    def human(o: TrialOutcome, motive: bool) -> HumanCode:
        stances = {c: "not_mentioned" for c in SCORER_CLASSES}
        stances["content_motive"] = "asserted" if motive else "rejected"
        return HumanCode(o.trial_id, "consensus", Coding(stances, "none", False))

    perfect = scorer_agreement(rows, [human(o, bool(o.y_over)) for o in rows], 0.8)
    assert perfect.kappa_y_over == pytest.approx(1.0) and perfect.scorer_primary and perfect.n == 20
    codes = [human(o, bool(o.y_over) != (k < 4)) for k, o in enumerate(rows)]   # 4 disagreements
    weaker = scorer_agreement(rows, codes, 0.8)
    assert weaker.kappa_y_over < 0.8 and not weaker.scorer_primary


def test_human_sample_is_stratified_blind_and_seeded(tmp_path):
    rows = [outcome(f"{a[0]}{k}", a, c, y=k % 3 == 0, explanation=f"text {a} {c} {k}")
            for a in ("sensitive", "neutral") for c in CONDITIONS for k in range(30)]
    sample = human_sample(rows, 36, seed=4)
    assert len(sample.response_ids) == 36 and len(set(sample.response_ids)) == 36
    strata = {sample.stratum_of[r] for r in sample.response_ids}
    assert strata == set(sample.stratum_of.values())         # every stratum represented
    assert human_sample(rows, 36, seed=4) == sample
    assert all(0 < f <= 1 for f in sample.inclusion.values())
    path = tmp_path / "sheet.csv"
    write_coding_sheet(path, rows, sample, seed=4)
    sheet = read_csv(path)
    assert [r["response_id"] for r in sheet] == [sheet_id(t, 4) for t in sample.response_ids]
    assert set(sheet[0]) == {"response_id", "explanation", *HUMAN_COLUMNS[1:]}
    assert all(r["coder"] == "" for r in sheet)
    # blind: no item, condition or replicate on the sheet
    for r in sheet:
        assert ":" not in r["response_id"] and not any(c in r["response_id"] for c in ("E0", "EP", "EW"))
    write_sample(tmp_path / "key.json", sample, seed=4)
    key = json.loads((tmp_path / "key.json").read_text("utf-8"))
    assert {r["sheet_id"]: r["trial_id"] for r in key["responses"]} == {sheet_id(t, 4): t for t in sample.response_ids}


def test_sheet_ids_resolve_back_to_trials_in_the_analysis(tmp_path):
    rows = [outcome(f"s{k}", "sensitive", "E0", y=k % 2 == 0) for k in range(10)]
    codes = [HumanCode(sheet_id(o.trial_id, PARAMS.seed), "consensus",
                       Coding({**{c: "not_mentioned" for c in SCORER_CLASSES},
                               "content_motive": "asserted" if o.y_over else "rejected"}, "none", False))
             for o in rows]
    results = analyse(rows, PARAMS, codes)
    assert results.scorer.n == 10 and results.scorer.kappa_y_over == pytest.approx(1.0)
    assert results.outcome_basis == "scorer"
    with pytest.raises(AnalysisError, match="unknown response ids"):
        analyse(rows, PARAMS, [HumanCode("Rnope", "A", codes[0].coding)])


def test_allocation_never_falls_short_of_n():
    assert sum(allocate({"a": 20, "b": 1, "c": 25, "d": 28}, 63).values()) == 63
    rng = random.Random(0)
    for _ in range(500):
        sizes = {f"s{i}": rng.randint(1, 30) for i in range(rng.randint(1, 12))}
        n = rng.randint(0, sum(sizes.values()) + 5)
        alloc = allocate(sizes, n)
        assert sum(alloc.values()) == min(n, sum(sizes.values()))
        assert all(0 <= alloc[s] <= sizes[s] for s in sizes)
        if n >= len(sizes):
            assert all(alloc[s] >= 1 for s in sizes)


# --------------------------------------------------------------------------- planted effects, end to end
RATES = {("sensitive", "E0"): 0.6, ("sensitive", "EP"): 0.55, ("sensitive", "EW"): 0.2,
         ("neutral", "E0"): 0.25, ("neutral", "EP"): 0.25, ("neutral", "EW"): 0.1}


def make_items(n_pairs: int) -> list[Item]:
    return [Item(item_id=f"{arm[0]}{k:03d}", arm=arm, unit_ids=(f"u{k}",), chapter="I.1", pair_id=f"p{k:03d}",
                 omission_origin="real", zh_context_from="a", zh_context_to="b")
            for k in range(n_pairs) for arm in ("sensitive", "neutral")]


def simulate(rates: dict[tuple[str, str], float], n_pairs: int = 60, refusal_rate: float = 0.05):
    items = make_items(n_pairs)
    arm_of = {it.item_id: it.arm for it in items}
    evidence_text = {e.text: k for k, e in EVIDENCE.items()}

    def resolver(it: Item) -> ItemText:
        return ItemText(tibetan=f"PASSAGE-{it.item_id}", zh_before="ctx", zh_after="ctx")

    def script(req):
        if req.task == "scorer":
            motive = MOTIVE in req.body
            stances = {c: "not_mentioned" for c in SCORER_CLASSES}
            stances["content_motive" if motive else "source_text"] = "asserted"
            return {"stances": stances, "primary": "content_motive" if motive else "source_text",
                    "disputes_premise": False, "motive_quote": "because of what it says" if motive else ""}
        item_id = re.search(r"PASSAGE-(\w+)", req.body).group(1)
        shown = next((k for text, k in evidence_text.items() if text in req.body), "none")
        condition = {"none": "E0", "EP": "EP"}.get(shown, "EW")
        rng = random.Random(req.key())
        if rng.random() < refusal_rate:
            return refusal_response(req, category="other")
        p = rates[(arm_of[item_id], condition)]
        return {"explanation": MOTIVE if rng.random() < p else SOURCE,
                "most_likely": "cannot_determine", "premise_ok": True}

    schedule = trials(items, CONDITIONS, replicates=3, seed=20261002)
    client = FakeClient(script)
    subject = ExperimentTaskSettings.from_config(LLM, "subject")
    scorer = ExperimentTaskSettings.from_config(LLM, "scorer")
    outcomes = run_trials(schedule, resolver, client, subject, scorer, EVIDENCE, LEXICON)
    return outcomes, analyse(outcomes, PARAMS)


def test_planted_h1_and_h2_effects_are_recovered():
    outcomes, results = simulate(RATES)
    assert len(outcomes) == 120 * 3 * 3
    h1, h2, h3 = results.test("H1"), results.test("H2"), results.test("H3")
    assert h1.estimate == pytest.approx(0.4, abs=0.1) and h1.ci_low > 0 and h1.p_holm < 0.01
    assert h2.estimate == pytest.approx(0.35, abs=0.1) and h2.ci_low > 0 and h2.p_holm < 0.01
    assert h1.role == h2.role == "confirmatory" and h3.role == "exploratory" and h3.p_holm is None
    assert h3.estimate == pytest.approx(0.25, abs=0.12)
    ep = results.test("EP_vs_E0")
    assert ep.role == "secondary" and abs(ep.estimate) < 0.1
    refused = [o for o in outcomes if o.refused]
    assert 0.02 < len(refused) / len(outcomes) < 0.08
    lo, hi = results.refusal_bounds["H1"]
    assert lo is not None and hi is not None
    uptake = [o.y_uptake for o in outcomes if o.measured and o.condition == "EW"]
    assert uptake and all(u is not None for u in uptake)
    assert results.lexical_kappa_y_any == pytest.approx(1.0)   # the synthetic motive text is lexical too
    assert results.served_models == ("claude-opus-5-5",)
    assert {t.name for t in results.tests} >= {"H1[premise_ok]", "H2[real_only]"}


def test_no_planted_effect_gives_no_confirmatory_finding():
    flat = {k: 0.3 for k in RATES}
    _, results = simulate(flat, n_pairs=40)
    for name in ("H1", "H2"):
        t = results.test(name)
        assert t.ci_low < 0 < t.ci_high and t.p_holm > 0.05, name


def test_low_scorer_kappa_switches_h1_h2_to_two_phase_corrected_outcomes():
    """The scorer over-calls Y_over in EW; human codes on a sample reveal it. Below the G4
    kappa, H1/H2 are re-estimated from the corrected outcomes, not the scorer labels."""
    rows, truth = [], {}
    for k in range(40):
        for arm in ("sensitive", "neutral"):
            item = f"{arm[0]}{k}"
            for cond in CONDITIONS:
                true_y = arm == "sensitive" and cond == "E0" and k % 2 == 0
                scorer_y = true_y or (cond == "EW" and k % 4 == 1)       # scorer false positives in EW
                o = outcome(item, arm, cond, y=scorer_y)
                rows.append(o)
                truth[o.trial_id] = true_y
    params = AnalysisParams(n_boot=200, n_perm=200, seed=3, n_draws=200)
    sample = human_sample(rows, 120, seed=params.seed)
    def code(tid: str) -> HumanCode:
        stances = {c: "not_mentioned" for c in SCORER_CLASSES}
        stances["content_motive"] = "asserted" if truth[tid] else "rejected"
        return HumanCode(sheet_id(tid, params.seed), "consensus", Coding(stances, "none", False))

    results = analyse(rows, params, [code(t) for t in sample.response_ids])
    assert results.scorer.kappa_y_over < params.min_scorer_kappa and results.outcome_basis == "two_phase"
    raw, h1 = results.test("H1[scorer]"), results.test("H1")
    assert raw.role == "sensitivity" and h1.role == "confirmatory" and h1.p_holm is not None
    assert raw.estimate == pytest.approx(0.5 - 0.25)                    # biased by the EW false positives
    assert h1.estimate == pytest.approx(0.5, abs=0.08)                  # corrected towards the truth
    assert h1.ci_low is not None and h1.ci_low <= h1.estimate <= h1.ci_high
    assert results.test("H2[scorer]").role == "sensitivity"
    no_codes = analyse(rows, params)
    assert no_codes.outcome_basis == "unvalidated" and no_codes.test("H1").estimate == pytest.approx(raw.estimate)
