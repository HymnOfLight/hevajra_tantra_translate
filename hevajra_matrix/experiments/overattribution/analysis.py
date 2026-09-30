"""Analysis of the over-attribution experiment (synthesis section 8).

    results = analyse(outcomes, AnalysisParams.from_prereg(settings.prereg), human_codes)

The item is the unit of analysis: replicates are averaged within item x condition first.
Refused trials are excluded from the primary analysis; ``refusal_bounds`` recomputes the
H1 and H2 estimates with refusals coded 0 and coded 1. Substituted-model, truncated,
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
is below ``min_scorer_kappa`` the scorer labels are NOT primary and the outcomes must be
two-phase corrected (strata condition x arm x scorer label) with ``stats.twophase``; this
module reports the flag and leaves the correction to that module.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from ...core.io import read_csv, write_csv
from ...topics.labels import cohen_kappa
from .score import SCORER_CLASSES, Coding, TrialOutcome, coding_error

RefusalCoding = Literal["exclude", "as0", "as1"]
Keep = Callable[[TrialOutcome], bool]
HUMAN_COLUMNS: tuple[str, ...] = ("response_id", "coder", *SCORER_CLASSES, "primary", "disputes_premise", "date", "note")
CONSENSUS = "consensus"
SCOPE = ("results are specific to the requested model at the campaign date; responses from a substituted "
         "model are excluded; evidence lines are synthetic and outputs are never philological evidence")


class AnalysisError(ValueError):
    """Invalid human codes or analysis parameters."""


@dataclass(frozen=True)
class AnalysisParams:
    n_boot: int = 10000
    n_perm: int = 10000
    seed: int = 0
    min_scorer_kappa: float = 0.80

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
class ScorerAgreement:
    kappa_y_over: float | None
    kappa_primary: float | None
    n: int
    human_human_kappa_y_over: float | None
    n_human_pairs: int
    repeat_kappa_primary: float | None
    n_repeat: int
    scorer_primary: bool               # kappa_y_over >= min_scorer_kappa (gate G4)


@dataclass(frozen=True)
class ExperimentResults:
    tests: tuple[TestResult, ...]
    refusal_bounds: Mapping[str, tuple[float | None, float | None]]   # test -> (as 0, as 1)
    cells: tuple[CellSummary, ...]
    scorer: ScorerAgreement | None
    lexical_kappa_y_any: float | None
    served_models: tuple[str, ...]
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
            value: float | None = 1.0 if refusals == "as1" else 0.0
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


# --------------------------------------------------------------------------- human codes
@dataclass(frozen=True)
class HumanCode:
    response_id: str
    coder: str
    coding: Coding


def load_human_codes(path: Path) -> list[HumanCode]:
    """Read ``human_codes.csv`` (one row per response and coder; coder "consensus" for an
    adjudicated code). Raises ``AnalysisError`` listing invalid rows."""
    codes, errors = [], []
    for line, row in enumerate(read_csv(path), start=2):
        cell = {k: (v or "").strip() for k, v in row.items() if k}
        stances = {c: cell.get(c, "") for c in SCORER_CLASSES}
        problem = coding_error(stances, cell.get("primary", ""))
        flag = cell.get("disputes_premise", "").lower()
        if flag not in ("true", "false", "1", "0"):
            problem = problem or f"disputes_premise must be true or false, not {flag!r}"
        if not cell.get("response_id") or not cell.get("coder"):
            problem = problem or "response_id and coder are required"
        if problem:
            errors.append(f"line {line}: {problem}")
            continue
        codes.append(HumanCode(cell["response_id"], cell["coder"],
                               Coding(stances, cell["primary"], flag in ("true", "1"))))
    if errors:
        raise AnalysisError(f"{path}: " + "; ".join(errors[:20]))
    return codes


def consensus_codes(codes: Iterable[HumanCode]) -> dict[str, Coding]:
    """response id -> the consensus coding: the "consensus" row when present, else the
    coders' coding when all coders agree on Y_over and primary; otherwise unresolved (absent)."""
    by_id: dict[str, list[HumanCode]] = defaultdict(list)
    for c in codes:
        by_id[c.response_id].append(c)
    out = {}
    for rid, rows in by_id.items():
        adjudicated = [r for r in rows if r.coder == CONSENSUS]
        if adjudicated:
            out[rid] = adjudicated[-1].coding
        elif len({(r.coding.y_over, r.coding.primary) for r in rows}) == 1:
            out[rid] = rows[0].coding
    return out


def scorer_agreement(outcomes: Sequence[TrialOutcome], codes: Sequence[HumanCode],
                     min_kappa: float) -> ScorerAgreement:
    """Scorer vs human consensus, human vs human, and scorer vs its own repeat."""
    consensus = consensus_codes(codes)
    pairs = [(o, consensus[o.trial_id]) for o in outcomes if o.measured and o.trial_id in consensus]
    k_over = cohen_kappa([(o.y_over, c.y_over) for o, c in pairs])
    coders: dict[str, dict[str, Coding]] = defaultdict(dict)
    for c in codes:
        if c.coder != CONSENSUS:
            coders[c.response_id].setdefault(c.coder, c.coding)
    hh = [tuple(v.y_over for v in list(by.values())[:2]) for by in coders.values() if len(by) >= 2]
    repeats = [(o.primary, o.repeat_primary) for o in outcomes if o.measured and o.repeat_primary is not None]
    return ScorerAgreement(
        kappa_y_over=k_over, kappa_primary=cohen_kappa([(o.primary, c.primary) for o, c in pairs]), n=len(pairs),
        human_human_kappa_y_over=cohen_kappa(hh), n_human_pairs=len(hh),  # type: ignore[arg-type]
        repeat_kappa_primary=cohen_kappa(repeats), n_repeat=len(repeats),
        scorer_primary=k_over is not None and k_over >= min_kappa)


@dataclass(frozen=True)
class HumanSample:
    response_ids: tuple[str, ...]            # in a seeded random order, blind to condition
    stratum_of: Mapping[str, str]            # every measured response -> stratum
    inclusion: Mapping[str, float]           # stratum -> sampling fraction (for two-phase weights)


def human_sample(outcomes: Sequence[TrialOutcome], n: int, seed: int) -> HumanSample:
    """Responses for the two human coders, stratified by condition x arm x scorer primary.

    One response per non-empty stratum first (when ``n`` allows), the rest proportionally
    (largest remainder), drawn at random within each stratum.
    """
    stratum_of = {o.trial_id: f"{o.condition}|{o.arm}|{o.primary}" for o in outcomes if o.measured}
    strata: dict[str, list[str]] = defaultdict(list)
    for rid, s in sorted(stratum_of.items()):
        strata[s].append(rid)
    n = min(n, len(stratum_of))
    alloc = {s: (1 if n >= len(strata) else 0) for s in strata}
    rest, total = n - sum(alloc.values()), len(stratum_of)
    quotas = {s: rest * len(ids) / total for s, ids in strata.items()}
    for s in strata:
        alloc[s] += int(quotas[s])
    for s in sorted(strata, key=lambda s: (quotas[s] - int(quotas[s]), s), reverse=True)[:n - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(f"{seed}:human")
    chosen = [rid for s in sorted(strata) for rid in rng.sample(strata[s], min(alloc[s], len(strata[s])))]
    rng.shuffle(chosen)
    return HumanSample(tuple(chosen), stratum_of,
                       {s: min(alloc[s], len(ids)) / len(ids) for s, ids in strata.items()})


def write_coding_sheet(path: Path, outcomes: Sequence[TrialOutcome], sample: HumanSample) -> None:
    """Blind sheet for one coder: response id and explanation only, code columns empty.
    It contains model output, so it is written under ``runs/`` and never committed."""
    text = {o.trial_id: o.explanation for o in outcomes}
    columns = ("response_id", "explanation", *HUMAN_COLUMNS[1:])
    write_csv(path, ({"response_id": rid, "explanation": text[rid]} for rid in sample.response_ids), columns, bom=True)


# --------------------------------------------------------------------------- everything
def analyse(outcomes: Sequence[TrialOutcome], params: AnalysisParams,
            human: Sequence[HumanCode] | None = None) -> ExperimentResults:
    """All pre-registered estimates, the refusal bounds, cell summaries and scorer checks."""
    tests = run_tests(outcomes, params)
    adjusted = holm({t.name: t.p_value for t in tests if t.role == "confirmatory"})
    tests = [replace(t, p_holm=adjusted.get(t.name)) for t in tests]
    sensitivity = {
        "premise_ok": lambda o: o.premise_ok is not False,
        "real_only": lambda o: o.omission_origin == "real",
    }
    for label, keep in sensitivity.items():
        for t in run_tests(outcomes, params, keep=keep, names=("H1", "H2")):
            tests.append(replace(t, name=f"{t.name}[{label}]", role="sensitivity"))
    bounds = {}
    for name in ("H1", "H2"):
        lo, hi = (run_tests(outcomes, params, refusals=r, names=(name,), n_boot=0)[0].estimate
                  for r in ("as0", "as1"))
        bounds[name] = (lo, hi)
    measured = [o for o in outcomes if o.measured and o.lexical is not None]
    lexical = cohen_kappa([(o.lexical == "asserted", bool(o.y_any)) for o in measured])
    return ExperimentResults(
        tests=tuple(tests), refusal_bounds=bounds, cells=cell_summaries(outcomes),
        scorer=scorer_agreement(outcomes, human, params.min_scorer_kappa) if human else None,
        lexical_kappa_y_any=lexical,
        served_models=tuple(sorted({o.served_model for o in outcomes if o.served_model})))
