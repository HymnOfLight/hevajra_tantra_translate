"""Per-unit scores of one aligner against gold, and the metrics computed from them (synthesis 5.2).

``gold.score`` builds an ``AlignmentScores``: one ``UnitScore`` per scored gold unit and one
``WitnessOnlyScore`` per gold witness-only segment. Every metric is a pure function of
those records, so the same function serves the point estimate and each bootstrap
replicate (``AlignmentScores.on_windows`` pools the records of resampled windows).

Conventions: a unit the aligner left unresolved has ``pred_relation`` None and status
"UNALIGNED"; it stays in every denominator (a missed link, a status disagreement). A metric
with an empty denominator is None. Link metrics count (reference unit, witness segment)
pairs; NULL metrics count units with relation ``no_counterpart``.

collate.v2 conventions (``standing_rendering``, ``transcribed_name``) are scored apart from
the relation (draft section 7), never gated:

    convention_precision   among units the aligner gave a convention, the share whose gold
                           convention is the same
    convention_recall      among gold units with a convention, the share given that very
                           convention (an unresolved unit is a miss)
    convention_f1, convention_unsupported_rate (V12 could not confirm the convention),
    and per convention ``convention_precision:<c>`` / ``convention_recall:<c>``

Only an aligner that produces conventions (``AlignmentScores.conventions``: a collate.v2
consensus) has them; for every other source (baselines, the placebo, external alignments,
a collate.v1 consensus) they are None (n/a), never 0, because a link without a convention
flag cannot say whether its source was asked. Gold rows without an annotated convention
("" in the gold file) are left out of convention precision and recall and of the D_any_conv
agreement.

Two calibers come from one gold (draft Q6): ``dany_kappa`` on D_any (the primary outcome,
conventions not counted) and ``dany_conv_kappa`` on D_any_conv (``matrix.status`` outcome
``any_conv``: a convention also counts as a deviation; for a source without conventions its
own D_any). ``relation_kappa`` is the overall relation agreement, descriptive only: a
collate.v1 source scored on v2 gold loses agreement by construction where v2 changed the
codebook (``generalised`` for merged lists; names and seed syllables judged as translated).
``relation_kappa_shared`` is the relation agreement on the units the two codebooks code
alike: linked gold units (relation not ``no_counterpart``) whose gold convention is annotated
as ``none`` and whose gold relation is neither ``generalised`` nor ``category_name_omitted``.
The section-7 adoption rule ("v2 not worse than v1 on relation agreement": at least v1's value
minus 0.05, a point-estimate tolerance) compares ``relation_kappa_shared``, never
``relation_kappa``.

``COLLATE_V2_FAMILIES`` names the metric families collate.v2 added; a ledger record written
before them lacks these keys and nothing else (``pipeline.measure.already_ledgered``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..core.types import REASON_REFUSED, REASON_SUBSTITUTED_MODEL, Convention, Estimate, Relation, Status
from .resample import cohen_kappa, f1, percentile_interval, positive_agreement, ratio, resample_windows

STATUS_CLASSES = (Status.PRESENT.value, Status.PARTIAL.value, Status.ABSENT.value)

UNALIGNED = Status.UNALIGNED.value
_NONE = Convention.NONE.value
CONVENTION_CLASSES = (Convention.STANDING_RENDERING.value, Convention.TRANSCRIBED_NAME.value)


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
    # collate.v2 conventions (``Convention`` values). gold_convention None: the gold row has
    # none annotated; pred_convention None: unresolved, or a source without conventions.
    gold_convention: str | None = None
    pred_convention: str | None = None
    gold_dev_conv: bool | None = None       # D_any_conv; None when gold_convention is None
    pred_dev_conv: bool | None = None       # None: unresolved
    convention_unsupported: bool = False    # V12 could not confirm the predicted convention

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
    conventions: bool = False      # the source produces conventions (a collate.v2 consensus)

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
                               tuple(x for w in windows for x in only.get(w, ())), self.excluded, self.conventions)

    def metrics(self) -> dict[str, float | None]:
        """Every scalar metric, plus per-class agreement, per-relation recall and refusal by group."""
        out = {name: fn(self) for name, fn in METRICS.items()}
        pairs = [(u.gold_status, u.pred_status) for u in self.units]
        out.update({f"status_agreement:{c}": positive_agreement(pairs, c) for c in STATUS_CLASSES})
        dev = _dev_pairs(self)
        out.update({f"dany_agreement:{c}": positive_agreement(dev, c) for c in ("dev", "nondev")})
        dev_conv = _dev_conv_pairs(self)
        out.update({f"dany_conv_agreement:{c}": positive_agreement(dev_conv, c) for c in ("dev", "nondev")})
        out.update({f"convention_precision:{c}": convention_precision(self, c) for c in CONVENTION_CLASSES})
        out.update({f"convention_recall:{c}": convention_recall(self, c) for c in CONVENTION_CLASSES})
        out.update({f"relation_recall:{r}": v for r, v in relation_recall(self).items()})
        out.update({f"refusal_rate:{g}": v for g, v in refusal_by_group(self).items()})
        return out

    def counts(self) -> dict[str, int | None]:
        """Sizes behind the metrics; ``pred_conventions`` is None for a source without conventions."""
        return {"units": len(self.units), "windows": len(self.windows()), "excluded": len(self.excluded),
                "gold_links": sum(len(u.gold_wit) for u in self.units),
                "gold_null": sum(1 for u in self.units if u.gold_relation == Relation.NO_COUNTERPART.value),
                "gold_witness_only": len(self.witness_only),
                "unresolved": sum(1 for u in self.units if u.pred_relation is None),
                "gold_convention_annotated": sum(1 for u in self.units if u.gold_convention is not None),
                "gold_conventions": sum(1 for u in self.units if _marked(u.gold_convention)),
                "pred_conventions": sum(1 for u in self.units if _marked(u.pred_convention))
                if self.conventions else None}

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


def _dev_conv_pairs(s: AlignmentScores) -> list[tuple[str, str]]:
    """D_any_conv labels on the units whose gold convention is annotated."""
    def label(dev: bool | None) -> str:
        return "unresolved" if dev is None else ("dev" if dev else "nondev")
    return [(label(u.gold_dev_conv), label(u.pred_dev_conv)) for u in s.units if u.gold_dev_conv is not None]


def dany_conv_kappa(s: AlignmentScores) -> float | None:
    """Cohen's kappa on D_any_conv (any deviation, or any transcribed, part-transcribed or
    standing-rendered name in the unit; wider than collate.v1's D_any, docs-zh/09 section 9.2,
    and not directly the caliber of published counts that include transcriptions, the
    comparable one being D_any) over the units whose gold convention is annotated. A source
    without conventions is read by its own D_any."""
    return cohen_kappa(_dev_conv_pairs(s))


def relation_kappa(s: AlignmentScores) -> float | None:
    """Cohen's kappa on the relation per unit (unresolved units predicted "UNALIGNED")."""
    return cohen_kappa([(u.gold_relation, u.pred_relation or UNALIGNED) for u in s.units])


# Gold relations the collate.v2 codebook changed (merged lists): left out of the shared caliber.
_RECODED_RELATIONS = frozenset({Relation.GENERALISED.value, Relation.CATEGORY_NAME_OMITTED.value})


def _codebooks_agree(u: UnitScore) -> bool:
    """A unit the v1 and v2 codebooks code alike: linked in gold, gold convention annotated as
    ``none`` (not a convention, not un-annotated) and a gold relation v2 did not recode."""
    return (u.gold_convention == _NONE and u.gold_relation != _NULL
            and u.gold_relation not in _RECODED_RELATIONS)


def relation_kappa_shared(s: AlignmentScores) -> float | None:
    """Cohen's kappa on the relation over the units the v1 and v2 codebooks code alike
    (``_codebooks_agree``; unresolved units predicted "UNALIGNED"). The relation agreement the
    section-7 adoption rule compares, for every source alike; None when no such unit exists
    (e.g. gold without annotated conventions)."""
    return cohen_kappa([(u.gold_relation, u.pred_relation or UNALIGNED) for u in s.units if _codebooks_agree(u)])


def relation_recall(s: AlignmentScores) -> dict[str, float | None]:
    """Per gold relation, the share of its units given exactly that relation."""
    out: dict[str, float | None] = {}
    for rel in sorted({u.gold_relation for u in s.units}):
        units = [u for u in s.units if u.gold_relation == rel]
        out[rel] = ratio(sum(1 for u in units if u.pred_relation == rel), len(units))
    return out


def _marked(convention: str | None) -> bool:
    return convention is not None and convention != _NONE


def convention_precision(s: AlignmentScores, convention: str | None = None) -> float | None:
    """Among units the source gave a convention (or ``convention``) and whose gold convention
    is annotated, the share with the same gold convention. None for a source without
    conventions."""
    if not s.conventions:
        return None
    given = [u for u in s.units if u.gold_convention is not None and _marked(u.pred_convention)
             and (convention is None or u.pred_convention == convention)]
    return ratio(sum(1 for u in given if u.gold_convention == u.pred_convention), len(given))


def convention_recall(s: AlignmentScores, convention: str | None = None) -> float | None:
    """Among gold units with a convention (or ``convention``), the share the source gave the
    same convention; an unresolved unit is a miss. None for a source without conventions."""
    if not s.conventions:
        return None
    gold = [u for u in s.units if _marked(u.gold_convention) and (convention is None or u.gold_convention == convention)]
    return ratio(sum(1 for u in gold if u.pred_convention == u.gold_convention), len(gold))


def convention_f1(s: AlignmentScores) -> float | None:
    return f1(convention_precision(s), convention_recall(s))


def convention_unsupported_rate(s: AlignmentScores) -> float | None:
    """Among scored units the source gave a convention, the share flagged
    ``convention_unsupported`` (by V12, or on a consensus link by ``collate.consensus`` when no
    majority reading with the voted relation holds the voted convention; gold annotation not
    needed). None without conventions."""
    if not s.conventions:
        return None
    given = [u for u in s.units if _marked(u.pred_convention)]
    return ratio(sum(1 for u in given if u.convention_unsupported), len(given))


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
    "status_kappa": status_kappa, "dany_kappa": dany_kappa, "dany_conv_kappa": dany_conv_kappa,
    "relation_kappa": relation_kappa, "relation_kappa_shared": relation_kappa_shared,
    "convention_precision": convention_precision, "convention_recall": convention_recall,
    "convention_f1": convention_f1, "convention_unsupported_rate": convention_unsupported_rate,
    "quote_failure_rate": quote_failure_rate, "invalid_handle_rate": invalid_handle_rate,
    "refusal_rate": refusal_rate,
}


BREAKDOWNS = ("status_agreement", "dany_agreement", "dany_conv_agreement", "relation_recall",
              "convention_precision", "convention_recall", "refusal_rate")
# keyed metrics of ``AlignmentScores.metrics``: "<breakdown>:<class, relation, convention or group>"

# Metric families added by collate.v2: scalar names and keyed-breakdown prefixes ("<family>:...").
COLLATE_V2_FAMILIES = frozenset({"relation_kappa", "relation_kappa_shared", "dany_conv_kappa", "dany_conv_agreement",
                                 "convention_precision", "convention_recall", "convention_f1",
                                 "convention_unsupported_rate"})


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


def without_each_window(s: AlignmentScores) -> dict[str, dict[str, float | None]]:
    """The metrics on all windows but one, for every window left out (empty for one window).

    Dev gold only: docs-zh/09 section 7 takes the collate.v1/v2 decision without I.1, the
    region collate.v2 was developed on, and reports every other region left out alike."""
    windows = s.windows()
    return {w: s.on_windows([x for x in windows if x != w]).metrics() for w in windows} if len(windows) > 1 else {}


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
