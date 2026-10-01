"""Per-unit scores of one aligner against gold, and the metrics computed from them (synthesis 5.2).

``gold.score`` builds an ``AlignmentScores``: one ``UnitScore`` per scored gold unit and one
``WitnessOnlyScore`` per gold witness-only segment. Every metric is a pure function of
those records, so the same function serves the point estimate and each bootstrap
replicate (``AlignmentScores.on_windows`` pools the records of resampled windows).

Conventions: a unit the aligner left unresolved has ``pred_relation`` None and status
"UNALIGNED"; it stays in every denominator (a missed link, a status disagreement). A metric
with an empty denominator is None. Link metrics count (reference unit, witness segment)
pairs; NULL metrics count units with relation ``no_counterpart``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..core.types import REASON_REFUSED, REASON_SUBSTITUTED_MODEL, Estimate, Relation, Status
from .resample import cohen_kappa, f1, percentile_interval, positive_agreement, ratio, resample_windows

STATUS_CLASSES = (Status.PRESENT.value, Status.PARTIAL.value, Status.ABSENT.value)

UNALIGNED = Status.UNALIGNED.value


@dataclass(frozen=True)
class UnitScore:
    """Gold and prediction for one reference unit (``pred_relation`` None: unresolved)."""

    unit_id: str
    window_id: str
    gold_relation: str
    gold_status: str
    gold_dev: bool
    gold_wit: frozenset[str]
    pred_relation: str | None
    pred_status: str
    pred_dev: bool | None
    pred_wit: frozenset[str]
    reason: str | None = None
    quote_failure: bool = False
    invalid_handle: bool = False
    topic_group: str = ""
    pred_confidence: str | None = None    # the instrument's own confidence (T1: high, medium, low)

    @property
    def status_correct(self) -> bool:
        return self.pred_status == self.gold_status


@dataclass(frozen=True)
class WitnessOnlyScore:
    """A gold witness-only segment and how the aligner treated it (``pred_kind`` None: not
    witness-only, i.e. linked to a unit or not mentioned)."""

    row_id: str
    window_id: str
    segment_id: str
    gold_kind: str
    pred_kind: str | None


@dataclass(frozen=True)
class AlignmentScores:
    """Per-unit comparison of one aligner with one gold set; metrics are computed from it."""

    source: str
    gold_set: str
    units: tuple[UnitScore, ...]
    witness_only: tuple[WitnessOnlyScore, ...] = ()
    excluded: tuple[str, ...] = ()

    def windows(self) -> tuple[str, ...]:
        """Window ids (the bootstrap clusters), in first-appearance order."""
        return tuple(dict.fromkeys([u.window_id for u in self.units] + [w.window_id for w in self.witness_only]))

    def unit_ids(self) -> frozenset[str]:
        return frozenset(u.unit_id for u in self.units)

    def on_windows(self, windows: Sequence[str]) -> "AlignmentScores":
        """The records of the given windows, pooled; a window listed twice counts twice."""
        units: dict[str, list[UnitScore]] = {}
        only: dict[str, list[WitnessOnlyScore]] = {}
        for u in self.units:
            units.setdefault(u.window_id, []).append(u)
        for w in self.witness_only:
            only.setdefault(w.window_id, []).append(w)
        return AlignmentScores(self.source, self.gold_set, tuple(x for w in windows for x in units.get(w, ())),
                               tuple(x for w in windows for x in only.get(w, ())), self.excluded)

    def metrics(self) -> dict[str, float | None]:
        """Every scalar metric, plus per-class agreement, per-relation recall and refusal by group."""
        out = {name: fn(self) for name, fn in METRICS.items()}
        pairs = [(u.gold_status, u.pred_status) for u in self.units]
        out.update({f"status_agreement:{c}": positive_agreement(pairs, c) for c in STATUS_CLASSES})
        dev = _dev_pairs(self)
        out.update({f"dany_agreement:{c}": positive_agreement(dev, c) for c in ("dev", "nondev")})
        out.update({f"relation_recall:{r}": v for r, v in relation_recall(self).items()})
        out.update({f"refusal_rate:{g}": v for g, v in refusal_by_group(self).items()})
        return out

    def counts(self) -> dict[str, int]:
        return {"units": len(self.units), "windows": len(self.windows()), "excluded": len(self.excluded),
                "gold_links": sum(len(u.gold_wit) for u in self.units),
                "gold_null": sum(1 for u in self.units if u.gold_relation == Relation.NO_COUNTERPART.value),
                "gold_witness_only": len(self.witness_only),
                "unresolved": sum(1 for u in self.units if u.pred_relation is None)}

    def confusion(self) -> dict[tuple[str, str], int]:
        """(gold status, predicted status) -> number of units."""
        return dict(Counter((u.gold_status, u.pred_status) for u in self.units))


# --------------------------------------------------------------------------- metrics
def link_precision(s: AlignmentScores) -> float | None:
    return ratio(sum(len(u.gold_wit & u.pred_wit) for u in s.units), sum(len(u.pred_wit) for u in s.units))


def link_recall(s: AlignmentScores) -> float | None:
    return ratio(sum(len(u.gold_wit & u.pred_wit) for u in s.units), sum(len(u.gold_wit) for u in s.units))


def link_f1(s: AlignmentScores) -> float | None:
    return f1(link_precision(s), link_recall(s))


_NULL = Relation.NO_COUNTERPART.value


def null_precision(s: AlignmentScores) -> float | None:
    predicted = [u for u in s.units if u.pred_relation == _NULL]
    return ratio(sum(1 for u in predicted if u.gold_relation == _NULL), len(predicted))


def null_recall(s: AlignmentScores) -> float | None:
    gold = [u for u in s.units if u.gold_relation == _NULL]
    return ratio(sum(1 for u in gold if u.pred_relation == _NULL), len(gold))


def witness_only_recall(s: AlignmentScores) -> float | None:
    return ratio(sum(1 for w in s.witness_only if w.pred_kind is not None), len(s.witness_only))


def witness_only_kind_agreement(s: AlignmentScores) -> float | None:
    """Among gold witness-only segments the aligner also found, the share with the gold kind."""
    found = [w for w in s.witness_only if w.pred_kind is not None]
    return ratio(sum(1 for w in found if w.pred_kind == w.gold_kind), len(found))


def status_kappa(s: AlignmentScores) -> float | None:
    """Cohen's kappa over PRESENT / PARTIAL / ABSENT (unresolved units predicted "UNALIGNED")."""
    return cohen_kappa([(u.gold_status, u.pred_status) for u in s.units])


def _dev_pairs(s: AlignmentScores) -> list[tuple[str, str]]:
    def label(dev: bool | None) -> str:
        return "unresolved" if dev is None else ("dev" if dev else "nondev")
    return [(label(u.gold_dev), label(u.pred_dev)) for u in s.units]


def dany_kappa(s: AlignmentScores) -> float | None:
    """Cohen's kappa on D_any (any deviation) per unit."""
    return cohen_kappa(_dev_pairs(s))


def relation_recall(s: AlignmentScores) -> dict[str, float | None]:
    """Per gold relation, the share of its units given exactly that relation."""
    out: dict[str, float | None] = {}
    for rel in sorted({u.gold_relation for u in s.units}):
        units = [u for u in s.units if u.gold_relation == rel]
        out[rel] = ratio(sum(1 for u in units if u.pred_relation == rel), len(units))
    return out


def quote_failure_rate(s: AlignmentScores) -> float | None:
    return ratio(sum(1 for u in s.units if u.quote_failure), len(s.units))


def invalid_handle_rate(s: AlignmentScores) -> float | None:
    return ratio(sum(1 for u in s.units if u.invalid_handle), len(s.units))


def _refused(u: UnitScore) -> bool:
    """A refusal by the requested model. Server-side fallback only fires when the requested
    model declines, so a ``substituted_model`` unit is a requested-model refusal too (its
    answer stays a reviewer hint and is never measured; impl_decisions 4)."""
    head = (u.reason or "").split(":")[0]
    return u.pred_relation is None and head in (REASON_REFUSED, REASON_SUBSTITUTED_MODEL)


def refusal_rate(s: AlignmentScores) -> float | None:
    return ratio(sum(1 for u in s.units if _refused(u)), len(s.units))


def refusal_by_group(s: AlignmentScores) -> dict[str, float | None]:
    groups = sorted({u.topic_group for u in s.units if u.topic_group})
    return {g: ratio(sum(1 for u in s.units if u.topic_group == g and _refused(u)),
                     sum(1 for u in s.units if u.topic_group == g)) for g in groups}


METRICS: Mapping[str, Callable[[AlignmentScores], float | None]] = {
    "link_precision": link_precision, "link_recall": link_recall, "link_f1": link_f1,
    "null_precision": null_precision, "null_recall": null_recall,
    "witness_only_recall": witness_only_recall, "witness_only_kind_agreement": witness_only_kind_agreement,
    "status_kappa": status_kappa, "dany_kappa": dany_kappa,
    "quote_failure_rate": quote_failure_rate, "invalid_handle_rate": invalid_handle_rate,
    "refusal_rate": refusal_rate,
}


BREAKDOWNS = ("status_agreement", "dany_agreement", "relation_recall", "refusal_rate")
# keyed metrics of ``AlignmentScores.metrics``: "<breakdown>:<class, relation or group>"


def interval_estimates(s: AlignmentScores, n_boot: int, seed: int,
                       names: Sequence[str] | None = None) -> dict[str, Estimate]:
    """Each named metric with its 95% window-cluster bootstrap interval (synthesis 5.2).

    ``names`` defaults to every key of ``s.metrics()``: the scalar ``METRICS`` and the keyed
    breakdowns (per-class agreement, per-relation recall, refusal by topic group), all on the
    same resampled windows. A breakdown key absent from a replicate (its relation or group
    not drawn) is undefined there and dropped, like any undefined replicate value.
    """
    full = s.metrics()
    keys = tuple(full) if names is None else tuple(names)
    for name in keys:
        if name not in METRICS and name.split(":", 1)[0] not in BREAKDOWNS:
            raise KeyError(f"unknown metric {name!r}")
    replicates = [s.on_windows(ws).metrics() for ws in resample_windows(s.windows(), n_boot, seed)]
    out = {}
    for name in keys:
        lo, hi = percentile_interval(r.get(name) for r in replicates)
        out[name] = Estimate(name=name, point=full.get(name), lo=lo, hi=hi, n=len(s.units),
                             scope=f"gold:{s.gold_set}", sources=(s.source,))
    return out


# Nominal probability that a unit is right, for each T1 confidence label (fixed before any
# gold is seen; the Brier score compares them with the outcome).
CONFIDENCE_PROBABILITY: Mapping[str, float] = MappingProxyType({"high": 0.9, "medium": 0.7, "low": 0.5})


def confidence_reliability(s: AlignmentScores) -> dict[str, Any] | None:
    """Is the instrument's confidence informative? (synthesis 5.2: confidence is used only as a
    stratum unless its Brier score beats a constant predictor on dev gold.)

    Over the units the aligner resolved with a confidence label, the outcome is "status
    correct" (the unit's status equals gold). ``brier`` scores the nominal probabilities of
    ``CONFIDENCE_PROBABILITY``; ``brier_constant`` the constant predictor that gives every unit
    the observed accuracy (p (1 - p), the best any constant can do on these units). ``table``
    is the reliability table: per label, n, correct units and accuracy. None when no unit
    carries a confidence (every control, gold, the placebo).
    """
    rated = [u for u in s.units if u.pred_confidence in CONFIDENCE_PROBABILITY]
    if not rated:
        return None
    hits = [1.0 if u.status_correct else 0.0 for u in rated]
    accuracy = sum(hits) / len(rated)
    brier = sum((CONFIDENCE_PROBABILITY[str(u.pred_confidence)] - y) ** 2 for u, y in zip(rated, hits)) / len(rated)
    constant = sum((accuracy - y) ** 2 for y in hits) / len(rated)
    table = {}
    for label, nominal in CONFIDENCE_PROBABILITY.items():
        mine = [y for u, y in zip(rated, hits) if u.pred_confidence == label]
        table[label] = {"n": len(mine), "correct": int(sum(mine)), "accuracy": ratio(sum(mine), len(mine)),
                        "nominal": nominal}
    return {"outcome": "status_correct", "n": len(rated), "accuracy": accuracy, "brier": brier,
            "brier_constant": constant, "beats_constant": brier < constant, "table": table}


def status_correct(s: AlignmentScores) -> dict[str, bool]:
    """Unit id -> status correct, the input of ``mcnemar``."""
    return {u.unit_id: u.status_correct for u in s.units}
