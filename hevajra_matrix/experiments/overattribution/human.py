"""Human coding of the over-attribution experiment: sample, blind sheet, codes, scorer kappa.

    sample = human_sample(outcomes, n=150, seed)        # stratified, seeded
    write_coding_sheet(path, outcomes, sample, seed)    # opaque response ids only
    codes = resolve_codes(load_human_codes(csv), outcomes, seed)   # sheet id -> trial id (this phase)

Blinding. A trial id is ``<item>:<condition>:<replicate>``, so it would show the coder the
condition and the item. The sheet therefore carries an opaque id (``sheet_id``: a seeded
hash of the trial id) and the explanation only; ``resolve_codes`` maps the ids of
``human_codes.csv`` back to trial ids. The private map (sheet id, trial id, stratum) is
written next to the sheet under ``runs/`` by ``write_sample``.

Strata are condition x arm x scorer Y_over label (synthesis 8: the strata of the two-phase
correction), so the sample is a uniform random sample within each correction stratum.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ...core.io import read_csv, write_csv
from ...topics.labels import cohen_kappa
from .score import SCORER_CLASSES, Coding, TrialOutcome, coding_error

HUMAN_COLUMNS: tuple[str, ...] = ("response_id", "coder", *SCORER_CLASSES, "primary", "disputes_premise", "date", "note")
CONSENSUS = "consensus"


class AnalysisError(ValueError):
    """Invalid human codes or analysis parameters."""


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


# --------------------------------------------------------------------------- blinding
def sheet_id(trial_id: str, seed: int) -> str:
    """Opaque, stable response id shown to coders (no item, condition or replicate)."""
    return "R" + hashlib.sha256(f"{seed}:human:{trial_id}".encode("utf-8")).hexdigest()[:12]


def resolve_codes(codes: Sequence[HumanCode], outcomes: Sequence[TrialOutcome], seed: int) -> list[HumanCode]:
    """Codes keyed by trial id: sheet ids are mapped back; ids that already are trial ids
    are kept. A code whose id matches no trial of ``outcomes`` (the phase being scored) is
    set aside: ``human_codes.csv`` holds the codes of both phases (pilot and main), so the
    caller reports ``len(codes) - len(result)`` instead of failing."""
    trial_of = {sheet_id(o.trial_id, seed): o.trial_id for o in outcomes}
    trial_of.update({o.trial_id: o.trial_id for o in outcomes})
    return [HumanCode(trial_of[c.response_id], c.coder, c.coding) for c in codes if c.response_id in trial_of]


# --------------------------------------------------------------------------- scorer agreement
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


def scorer_agreement(outcomes: Sequence[TrialOutcome], codes: Sequence[HumanCode],
                     min_kappa: float) -> ScorerAgreement:
    """Scorer vs human consensus, human vs human, and scorer vs its own repeat (codes keyed by trial id)."""
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


# --------------------------------------------------------------------------- sample
def stratum(o: TrialOutcome) -> str:
    """Two-phase stratum of a measured trial: condition x arm x scorer Y_over."""
    return f"{o.condition}|{o.arm}|y_over={bool(o.y_over)}"


@dataclass(frozen=True)
class HumanSample:
    response_ids: tuple[str, ...]            # trial ids, in a seeded random order
    stratum_of: Mapping[str, str]            # every measured response -> stratum
    inclusion: Mapping[str, float]           # stratum -> sampling fraction (for two-phase weights)


def allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Stratum -> sample size: one per non-empty stratum first (when ``n`` allows), the rest
    by largest remainder, never more than a stratum holds; any excess is redistributed, so
    the total is ``min(n, sum(sizes))``."""
    n = min(n, sum(sizes.values()))
    alloc = {s: (1 if n >= len(sizes) and size else 0) for s, size in sizes.items()}
    while sum(alloc.values()) < n:
        room = {s: sizes[s] - alloc[s] for s in sizes if sizes[s] > alloc[s]}
        rest, total = n - sum(alloc.values()), sum(sizes[s] for s in room)
        quotas = {s: rest * sizes[s] / total for s in room}
        for s in room:
            alloc[s] += min(int(quotas[s]), room[s])
        left = n - sum(alloc.values())
        for s in sorted(room, key=lambda s: (quotas[s] - int(quotas[s]), s), reverse=True):
            if left and alloc[s] < sizes[s]:
                alloc[s] += 1
                left -= 1
    return alloc


def human_sample(outcomes: Sequence[TrialOutcome], n: int, seed: int) -> HumanSample:
    """Responses for the two human coders, stratified by ``stratum`` and drawn at random
    within each stratum (``allocate`` gives the per-stratum sizes)."""
    stratum_of = {o.trial_id: stratum(o) for o in outcomes if o.measured}
    strata: dict[str, list[str]] = defaultdict(list)
    for rid, s in sorted(stratum_of.items()):
        strata[s].append(rid)
    alloc = allocate({s: len(ids) for s, ids in strata.items()}, n)
    rng = random.Random(f"{seed}:human")
    chosen = [rid for s in sorted(strata) for rid in rng.sample(strata[s], alloc[s])]
    rng.shuffle(chosen)
    return HumanSample(tuple(chosen), stratum_of, {s: alloc[s] / len(ids) for s, ids in strata.items()})


def write_coding_sheet(path: Path, outcomes: Sequence[TrialOutcome], sample: HumanSample, seed: int) -> None:
    """Blind sheet for one coder: opaque response id and explanation only, code columns
    empty. It contains model output, so it is written under ``runs/`` and never committed."""
    text = {o.trial_id: o.explanation for o in outcomes}
    columns = ("response_id", "explanation", *HUMAN_COLUMNS[1:])
    write_csv(path, ({"response_id": sheet_id(rid, seed), "explanation": text[rid]} for rid in sample.response_ids),
              columns, bom=True)


def write_sample(path: Path, sample: HumanSample, seed: int) -> None:
    """The private key of the sheet: sheet id -> trial id and stratum, plus inclusion fractions."""
    rows = [{"sheet_id": sheet_id(rid, seed), "trial_id": rid, "stratum": sample.stratum_of[rid]}
            for rid in sample.response_ids]
    payload = {"seed": seed, "responses": rows, "inclusion": dict(sorted(sample.inclusion.items()))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
