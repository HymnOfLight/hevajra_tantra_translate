"""Analysis of the over-attribution experiment (synthesis section 8).

    results = analyse(outcomes, AnalysisParams.from_prereg(settings.prereg), human_codes)

The item is the unit of analysis: replicates are averaged within item x condition first.
Refused trials are excluded from the primary analysis. ``refusal_bounds`` are Manski bounds
per contrast: refusals coded 0 on the minuend side and 1 on the subtrahend side (lower bound),
and the reverse (upper bound). ``refusal_uniform`` codes every refusal 0, then every refusal
1, in all cells (a sensitivity analysis, not a bound for a difference). Substituted-model, truncated,
invalid and scorer-unverified trials never enter an estimate; they are counted per cell.

Hypotheses (one-sided, pre-registered; Holm over H1-H2)
    H1  sensitive arm: P(Y_over | E0) > P(Y_over | EW)
        paired item bootstrap CI, sign-flip permutation p
    H2  under E0: sensitive > neutral; item bootstrap CI (within arm), label permutation p
    H3  difference in differences (E0 - EW, sensitive minus neutral): EXPLORATORY, never in
        the Holm family (promoting it needs >= 100 verified items per arm and a
        pre-registration amendment)
Secondary (two-sided): EP - E0 over all items (demand/length effect); EW-V - EW-S.
Sensitivity: H1 and H2 without trials whose subject disputed the premise
(``premise_ok`` false), and on ``real`` omissions only.

Scorer validity (gate G4): Cohen's kappa of scorer vs human consensus on Y_over. When it
is below ``min_scorer_kappa`` the scorer labels are NOT primary and H1/H2 are two-phase
corrected (``two_phase``; strata condition x arm x scorer Y_over, the strata of the human
sample): human-coded trials keep their consensus Y_over, every other measured trial of
stratum h has Y_over ~ Bernoulli(p_h), p_h ~ Beta(x_h + 1/2, n_h - x_h + 1/2) (the Jeffreys
prior ``stats.twophase`` uses; drawn here with ``random.betavariate`` so the experiment
stays independent of the matrix packages). The corrected H1/H2 become the confirmatory tests (estimate and
permutation p on the posterior-mean outcomes; CI over imputation draws x item bootstrap) and
the scorer-label tests are kept as ``H1[scorer]``/``H2[scorer]`` sensitivity rows.
``outcome_basis`` records which basis H1/H2 use: "scorer" (kappa passed), "two_phase",
"uncalibrated", or "unvalidated" (no human codes yet: the scorer labels are not validated and
E6 is not final). "uncalibrated" is the two-phase case in which a stratum holds measured
trials but no human code: imputing it would report the Beta(1/2, 1/2) prior as data, so (as
``stats.twophase.UncalibratedStrata`` does for the matrix) H1/H2 and their sensitivity rows
are not estimable, the refusal bounds are not computed, and ``uncalibrated`` lists the strata
(stratum -> uncoded trials). The scorer-label rows ``H1[scorer]``/``H2[scorer]`` are kept.

Human codes are read for both phases from one file; codes whose id is no trial of the scored
phase are set aside (``human_codes_set_aside`` counts them).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from ...topics.labels import cohen_kappa
from .human import (  # noqa: F401  (re-exported: the human-coding API lives in .human)
    CONSENSUS,
    HUMAN_COLUMNS,
    AnalysisError,
    HumanCode,
    HumanSample,
    ScorerAgreement,
    consensus_codes,
    human_sample,
    load_human_codes,
    resolve_codes,
    scorer_agreement,
    sheet_id,
    stratum,
    write_coding_sheet,
    write_sample,
)
from .score import TrialOutcome

RefusalCoding = Literal["exclude", "as0", "as1"] | Callable[[TrialOutcome], float]
Keep = Callable[[TrialOutcome], bool]
OutcomeBasis = Literal["scorer", "two_phase", "uncalibrated", "unvalidated"]
SCOPE = ("results are specific to the requested model at the campaign date; responses from a substituted "
         "model are excluded; evidence lines are synthetic and outputs are never philological evidence")


@dataclass(frozen=True)
class AnalysisParams:
    n_boot: int = 10000
    n_perm: int = 10000
    seed: int = 0
    min_scorer_kappa: float = 0.80
    n_draws: int = 2000                # two-phase imputation draws (each with one item bootstrap)

    @classmethod
    def from_prereg(cls, prereg: Mapping[str, Any], n_boot: int = 10000) -> "AnalysisParams":
        """Seed from ``experiment``, permutations from ``stats``, kappa threshold from ``gates.g4``."""
        exp, stats = prereg.get("experiment") or {}, prereg.get("stats") or {}
        g4 = (prereg.get("gates") or {}).get("g4") or {}
        return cls(n_boot=n_boot, n_perm=int(stats.get("n_permutations", 10000)), seed=int(exp.get("seed", 0)),
                   min_scorer_kappa=float(g4.get("min_scorer_kappa", 0.80)))


@dataclass(frozen=True)
class TestResult:
    """One contrast. ``p_value`` is one-sided for H1-H3 and two-sided for secondary tests."""

    name: str
    role: Literal["confirmatory", "exploratory", "secondary", "sensitivity"]
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    n: Mapping[str, int]
    p_holm: float | None = None
    not_estimable: str | None = None


@dataclass(frozen=True)
class CellSummary:
    """Trial-level counts and descriptive rates of one arm x condition cell."""

    arm: str
    condition: str
    n_trials: int
    counts: Mapping[str, int]          # refused, truncated, invalid, substituted, scorer_failed, measured
    rates: Mapping[str, float | None]  # refused, y_over, y_any, y_uptake, disputes_premise, premise_not_ok


@dataclass(frozen=True)
class ExperimentResults:
    tests: tuple[TestResult, ...]
    refusal_bounds: Mapping[str, tuple[float | None, float | None]]   # test -> Manski (lo, hi)
    cells: tuple[CellSummary, ...]
    scorer: ScorerAgreement | None
    lexical_kappa_y_any: float | None
    served_models: tuple[str, ...]
    refusal_uniform: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)  # (as 0, as 1)
    outcome_basis: OutcomeBasis = "unvalidated"   # what H1/H2 are computed on (gate G4)
    uncalibrated: Mapping[str, int] = field(default_factory=dict)   # stratum -> uncoded trials, no human code
    human_codes_set_aside: int = 0     # codes whose id is no trial of the scored phase
    scope: str = SCOPE

    def test(self, name: str) -> TestResult:
        return next(t for t in self.tests if t.name == name)


# --------------------------------------------------------------------------- small statistics
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def _p(null: Sequence[float], observed: float, two_sided: bool) -> float:
    eps = 1e-12
    hits = sum(abs(x) >= abs(observed) - eps if two_sided else x >= observed - eps for x in null)
    return (1 + hits) / (1 + len(null))


def paired_test(d: Sequence[float], n_boot: int, n_perm: int, rng: random.Random,
                two_sided: bool = False) -> tuple[float, float | None, float | None, float | None]:
    """Mean of paired differences, percentile bootstrap 95% CI over items, sign-flip p."""
    est = _mean(d)
    if not n_boot:
        return est, None, None, None
    boots = sorted(_mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(n_boot))
    null = [_mean([x if rng.random() < 0.5 else -x for x in d]) for _ in range(n_perm)]
    return est, _quantile(boots, 0.025), _quantile(boots, 0.975), _p(null, est, two_sided)


def two_sample_test(a: Sequence[float], b: Sequence[float], n_boot: int, n_perm: int, rng: random.Random,
                    two_sided: bool = False) -> tuple[float, float | None, float | None, float | None]:
    """mean(a) - mean(b), bootstrap CI resampling within each group, label-permutation p."""
    est = _mean(a) - _mean(b)
    if not n_boot:
        return est, None, None, None
    boots = sorted(_mean([a[rng.randrange(len(a))] for _ in a]) - _mean([b[rng.randrange(len(b))] for _ in b])
                   for _ in range(n_boot))
    pooled, null = list(a) + list(b), []
    for _ in range(n_perm):
        rng.shuffle(pooled)
        null.append(_mean(pooled[:len(a)]) - _mean(pooled[len(a):]))
    return est, _quantile(boots, 0.025), _quantile(boots, 0.975), _p(null, est, two_sided)


def holm(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values; None (not estimable) stays None and is not counted."""
    ordered = sorted((p, name) for name, p in pvalues.items() if p is not None)
    m, running, out = len(ordered), 0.0, {name: None for name in pvalues}
    for k, (p, name) in enumerate(ordered):
        running = max(running, min(1.0, (m - k) * p))
        out[name] = running
    return out


# --------------------------------------------------------------------------- item means
def item_means(outcomes: Iterable[TrialOutcome], outcome: str = "y_over", refusals: RefusalCoding = "exclude",
               keep: Keep | None = None) -> dict[tuple[str, str], float]:
    """(item, condition) -> mean of ``outcome`` over that cell's replicates."""
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for o in outcomes:
        if keep is not None and not keep(o):
            continue
        if o.refused and refusals != "exclude":
            value: float | None = (refusals(o) if callable(refusals)
                                   else 1.0 if refusals == "as1" else 0.0)
        else:
            y = getattr(o, outcome) if o.measured else None
            value = None if y is None else float(y)
        if value is not None:
            values[(o.item_id, o.condition)].append(value)
    return {k: _mean(v) for k, v in values.items()}


def _contrasts(means: Mapping[tuple[str, str], float], arm_of: Mapping[str, str],
               ew_of: Mapping[str, str]) -> dict[str, tuple[bool, list[float], list[float]]]:
    """name -> (paired?, values, second group) for every contrast of the analysis."""
    def diff(item: str, x: str, y: str) -> float | None:
        return means[(item, x)] - means[(item, y)] if (item, x) in means and (item, y) in means else None

    def arm(name: str) -> list[str]:
        return sorted(i for i, a in arm_of.items() if a == name)

    d_s = [d for i in arm("sensitive") if (d := diff(i, "E0", "EW")) is not None]
    d_n = [d for i in arm("neutral") if (d := diff(i, "E0", "EW")) is not None]
    e0 = {a: [means[(i, "E0")] for i in arm(a) if (i, "E0") in means] for a in ("sensitive", "neutral")}
    ew = {v: [means[(i, "EW")] for i in sorted(arm_of) if ew_of.get(i) == v and (i, "EW") in means]
          for v in ("EW-V", "EW-S")}
    return {
        "H1": (True, d_s, []),
        "H2": (False, e0["sensitive"], e0["neutral"]),
        "H3": (False, d_s, d_n),
        "EP_vs_E0": (True, [d for i in sorted(arm_of) if (d := diff(i, "EP", "E0")) is not None], []),
        "EWV_vs_EWS": (False, ew["EW-V"], ew["EW-S"]),
    }


ROLES = {"H1": "confirmatory", "H2": "confirmatory", "H3": "exploratory",
         "EP_vs_E0": "secondary", "EWV_vs_EWS": "secondary"}
TWO_SIDED = frozenset({"EP_vs_E0", "EWV_vs_EWS"})


def run_tests(outcomes: Sequence[TrialOutcome], params: AnalysisParams, outcome: str = "y_over",
              refusals: RefusalCoding = "exclude", keep: Keep | None = None, names: Sequence[str] = tuple(ROLES),
              n_boot: int | None = None) -> list[TestResult]:
    """Estimate the named contrasts on item means; ``n_boot=0`` gives point estimates only."""
    arm_of = {o.item_id: o.arm for o in outcomes}
    ew_of = {o.item_id: o.evidence for o in outcomes if o.condition == "EW"}
    contrasts = _contrasts(item_means(outcomes, outcome, refusals, keep), arm_of, ew_of)
    boots = params.n_boot if n_boot is None else n_boot
    results = []
    for name in names:
        paired, a, b = contrasts[name]
        n = {"items": len(a)} if paired else {"a": len(a), "b": len(b)}
        if len(a) < 2 or (not paired and len(b) < 2):
            results.append(TestResult(name, ROLES[name], None, None, None, None, n,  # type: ignore[arg-type]
                                      not_estimable=f"fewer than 2 items with data ({n})"))
            continue
        rng = random.Random(f"{params.seed}:{name}:{outcome}")
        if paired:
            est, lo, hi, p = paired_test(a, boots, params.n_perm, rng, name in TWO_SIDED)
        else:
            est, lo, hi, p = two_sample_test(a, b, boots, params.n_perm, rng, name in TWO_SIDED)
        results.append(TestResult(name, ROLES[name], est, lo, hi, p, n))  # type: ignore[arg-type]
    return results


# --------------------------------------------------------------------------- cells
def cell_summaries(outcomes: Sequence[TrialOutcome]) -> tuple[CellSummary, ...]:
    groups: dict[tuple[str, str], list[TrialOutcome]] = defaultdict(list)
    for o in outcomes:
        groups[(o.arm, o.condition)].append(o)

    def rate(xs: Sequence[bool | None]) -> float | None:
        vals = [bool(x) for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else None

    cells = []
    for (arm, condition), group in sorted(groups.items()):
        measured = [o for o in group if o.measured]
        counts = Counter("measured" if o.measured else
                         ("scorer_failed" if o.status == "ok" else o.status) for o in group)
        cells.append(CellSummary(
            arm=arm, condition=condition, n_trials=len(group),
            counts={k: counts.get(k, 0) for k in
                    ("measured", "refused", "truncated", "invalid", "substituted", "scorer_failed")},
            rates={"refused": rate([o.refused for o in group if o.status != "substituted"]),
                   "y_over": rate([o.y_over for o in measured]), "y_any": rate([o.y_any for o in measured]),
                   "y_uptake": rate([o.y_uptake for o in measured]),
                   "disputes_premise": rate([o.disputes_premise for o in measured]),
                   "premise_not_ok": rate([None if o.premise_ok is None else not o.premise_ok
                                           for o in group if o.status == "ok"])}))
    return tuple(cells)


# --------------------------------------------------------------------------- everything
# --------------------------------------------------------------------------- refusal bounds
# contrast -> (minuend side, subtrahend side): a refusal on the minuend side coded 0 and on the
# subtrahend side coded 1 gives the smallest estimate, the reverse the largest.
_SIDES: dict[str, tuple[Keep, Keep]] = {
    "H1": (lambda o: o.condition == "E0", lambda o: o.condition == "EW"),
    "H2": (lambda o: o.arm == "sensitive", lambda o: o.arm == "neutral"),
}


def refusal_bounds(outcomes: Sequence[TrialOutcome], params: AnalysisParams,
                   ) -> tuple[dict[str, tuple[float | None, float | None]], dict[str, tuple[float | None, float | None]]]:
    """(Manski bounds, uniform codings) of H1 and H2 over every coding of the refusals."""
    def estimate(name: str, coding: RefusalCoding) -> float | None:
        return run_tests(outcomes, params, refusals=coding, names=(name,), n_boot=0)[0].estimate

    bounds, uniform = {}, {}
    for name, (minuend, _) in _SIDES.items():
        lo = estimate(name, lambda o, m=minuend: 0.0 if m(o) else 1.0)
        hi = estimate(name, lambda o, m=minuend: 1.0 if m(o) else 0.0)
        bounds[name] = (lo, hi)
        uniform[name] = (estimate(name, "as0"), estimate(name, "as1"))
    return bounds, uniform


# --------------------------------------------------------------------------- two-phase correction
def _with_y(outcomes: Sequence[TrialOutcome], y: Mapping[str, float]) -> list[TrialOutcome]:
    return [replace(o, y_over=y[o.trial_id]) if o.trial_id in y else o for o in outcomes]  # type: ignore[arg-type]


def _phase2(outcomes: Sequence[TrialOutcome], human: Sequence[HumanCode],
            ) -> tuple[dict[str, float], dict[str, list[str]], dict[str, tuple[float, float]]]:
    """(human Y_over per coded trial, uncoded trials per stratum, Beta parameters per stratum)."""
    consensus = consensus_codes(human)
    measured = [o for o in outcomes if o.measured]
    known = {o.trial_id: float(consensus[o.trial_id].y_over) for o in measured if o.trial_id in consensus}
    pools: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])     # stratum -> [Y_over, not Y_over]
    for o in measured:
        if o.trial_id in known:
            counts[stratum(o)][0 if known[o.trial_id] else 1] += 1
        else:
            pools[stratum(o)].append(o.trial_id)
    return known, dict(pools), {h: (counts[h][0] + 0.5, counts[h][1] + 0.5) for h in pools}


def uncalibrated_strata(outcomes: Sequence[TrialOutcome], human: Sequence[HumanCode]) -> dict[str, int]:
    """Stratum -> uncoded measured trials, for strata with no human code (nothing calibrates them)."""
    _, pools, alpha = _phase2(outcomes, human)
    return {h: len(ids) for h, ids in sorted(pools.items()) if alpha[h] == (0.5, 0.5)}


def _posterior_mean_y(outcomes: Sequence[TrialOutcome], human: Sequence[HumanCode]) -> dict[str, float]:
    known, pools, alpha = _phase2(outcomes, human)
    return {**known, **{t: alpha[h][0] / sum(alpha[h]) for h, ids in pools.items() for t in ids}}


def two_phase(outcomes: Sequence[TrialOutcome], human: Sequence[HumanCode], params: AnalysisParams,
              names: Sequence[str] = ("H1", "H2")) -> list[TestResult]:
    """H1/H2 on two-phase-corrected Y_over (see the module docstring); ``human`` keyed by trial id."""
    known, pools, alpha = _phase2(outcomes, human)
    tests = run_tests(_with_y(outcomes, _posterior_mean_y(outcomes, human)), params, names=names)
    if not params.n_boot or not params.n_draws:
        return [replace(t, ci_low=None, ci_high=None) for t in tests]
    rng = random.Random(f"{params.seed}:twophase")
    draws: dict[str, list[float]] = defaultdict(list)
    arm_of = {o.item_id: o.arm for o in outcomes}
    ew_of = {o.item_id: o.evidence for o in outcomes if o.condition == "EW"}
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)       # (item, condition) -> measured trials
    for o in outcomes:
        if o.measured:
            cells[(o.item_id, o.condition)].append(o.trial_id)
    items = sorted(arm_of)
    for _ in range(params.n_draws):
        y = dict(known)
        for h in sorted(pools):
            p = rng.betavariate(*alpha[h])
            y.update({t: float(rng.random() < p) for t in pools[h]})
        means = {key: _mean([y[t] for t in ids]) for key, ids in cells.items()}
        weight = Counter(rng.choice(items) for _ in items)              # one item bootstrap per draw
        boot = {(f"{i}#{k}", c): m for (i, c), m in means.items() for k in range(weight.get(i, 0))}
        b_arm = {f"{i}#{k}": arm_of[i] for i in weight for k in range(weight[i])}
        b_ew = {f"{i}#{k}": ew_of[i] for i in weight if i in ew_of for k in range(weight[i])}
        contrasts = _contrasts(boot, b_arm, b_ew)
        for name in names:
            paired, a, b = contrasts[name]
            if len(a) >= 2 and (paired or len(b) >= 2):
                draws[name].append(_mean(a) - (0.0 if paired else _mean(b)))
    out = []
    for t in tests:
        xs = sorted(draws.get(t.name, ()))
        out.append(replace(t, ci_low=_quantile(xs, 0.025) if xs else None, ci_high=_quantile(xs, 0.975) if xs else None))
    return out


# --------------------------------------------------------------------------- everything
def analyse(outcomes: Sequence[TrialOutcome], params: AnalysisParams,
            human: Sequence[HumanCode] | None = None) -> ExperimentResults:
    """All pre-registered estimates, the refusal bounds, cell summaries and scorer checks.

    ``human`` may use sheet ids or trial ids (``resolve_codes``). Below the G4 kappa the
    confirmatory H1/H2 are the two-phase-corrected ones.
    """
    n_codes = len(human or ())
    human = resolve_codes(human, outcomes, params.seed) if human else None
    set_aside = n_codes - len(human or ())
    human = human or None
    scorer = scorer_agreement(outcomes, human, params.min_scorer_kappa) if human else None
    basis: OutcomeBasis = "unvalidated" if scorer is None else "scorer" if scorer.scorer_primary else "two_phase"
    uncalibrated = uncalibrated_strata(outcomes, human) if basis == "two_phase" and human else {}
    if uncalibrated:
        basis = "uncalibrated"
    tests = run_tests(outcomes, params)
    basis_outcomes: Sequence[TrialOutcome] = outcomes
    reason = "no human code in stratum " + ", ".join(f"{h} ({n} uncoded)" for h, n in uncalibrated.items())
    if basis in ("two_phase", "uncalibrated"):
        assert human is not None
        if basis == "two_phase":
            corrected = {t.name: t for t in two_phase(outcomes, human, params)}
            basis_outcomes = _with_y(outcomes, _posterior_mean_y(outcomes, human))
        else:
            corrected = {t.name: replace(t, estimate=None, ci_low=None, ci_high=None, p_value=None,
                                         not_estimable=reason) for t in tests if t.name in ("H1", "H2")}
        raw = [replace(t, name=f"{t.name}[scorer]", role="sensitivity") for t in tests if t.name in corrected]
        tests = [corrected.get(t.name, t) for t in tests] + raw
    adjusted = holm({t.name: t.p_value for t in tests if t.role == "confirmatory"})
    tests = [replace(t, p_holm=adjusted.get(t.name)) for t in tests]
    sensitivity = {
        "premise_ok": lambda o: o.premise_ok is not False,
        "real_only": lambda o: o.omission_origin == "real",
    }
    for label, keep in sensitivity.items():
        for t in run_tests(basis_outcomes, params, keep=keep, names=("H1", "H2")):
            if uncalibrated:
                t = replace(t, estimate=None, ci_low=None, ci_high=None, p_value=None, not_estimable=reason)
            tests.append(replace(t, name=f"{t.name}[{label}]", role="sensitivity"))
    if uncalibrated:
        bounds = uniform = {name: (None, None) for name in _SIDES}
    else:
        bounds, uniform = refusal_bounds(basis_outcomes, params)
    measured = [o for o in outcomes if o.measured and o.lexical is not None]
    lexical = cohen_kappa([(o.lexical == "asserted", bool(o.y_any)) for o in measured])
    return ExperimentResults(
        tests=tuple(tests), refusal_bounds=bounds, refusal_uniform=uniform, cells=cell_summaries(outcomes),
        scorer=scorer, lexical_kappa_y_any=lexical, outcome_basis=basis, uncalibrated=uncalibrated,
        human_codes_set_aside=set_aside,
        served_models=tuple(sorted({o.served_model for o in outcomes if o.served_model})))
