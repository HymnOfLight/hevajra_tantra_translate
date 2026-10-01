"""Read the committed human annotations (gold, verdicts, topic labels) and the run's
review plans, in the form the stages need them.

Committed files live under ``data/annotations/`` (ids and short quotes only); review plans
(with strata and inclusion probabilities) live in the run directory under ``review/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..core.ids import orphan_row_id
from ..core.types import Verdict
from ..evaluation import gold as gold_sets
from ..matrix.build import segment_fingerprint
from ..review import sampling
from ..review import verdicts as verdict_files
from ..topics import TopicCodebook, TopicLabel, current_labels, load_codebook, load_labels, topic_group
from .context import RunContext, Texts

GOLD_PRIMARY_SETS = ("dev", "test")          # the second annotator's set only measures human agreement


def gold_dir(ctx: RunContext, witness: str) -> Path:
    return ctx.data_dir / "annotations" / "gold" / witness


def verdict_dir(ctx: RunContext, witness: str) -> Path:
    return ctx.data_dir / "annotations" / "verdicts" / witness


def topic_labels_path(ctx: RunContext, reference: str) -> Path:
    return ctx.data_dir / "annotations" / "topics" / f"{reference}.csv"


def ledger_path(ctx: RunContext) -> Path:
    return ctx.data_dir / "ledger" / "test_evaluations.jsonl"


def load_gold(ctx: RunContext, texts: Texts, name: str) -> gold_sets.GoldSet | None:
    """Gold set ``name`` of the run's witness, or None when it has not been annotated yet."""
    path = gold_dir(ctx, texts.witness_id) / f"{name}.csv"
    return gold_sets.load(path, texts.reference_id) if path.is_file() else None


def gold_verdicts(ctx: RunContext, texts: Texts) -> list[Verdict]:
    """The primary annotator's gold (dev and test) as verdicts for ``build_cells``."""
    out: list[Verdict] = []
    for name in GOLD_PRIMARY_SETS:
        found = load_gold(ctx, texts, name)
        if found is not None:
            out.extend(gold_sets.to_verdicts(found))
    return out


def review_verdicts(ctx: RunContext, texts: Texts) -> list[Verdict]:
    """Committed verify/audit/resolve verdicts of the run's witness (all batches)."""
    directory = verdict_dir(ctx, texts.witness_id)
    return list(verdict_files.load(directory)) if directory.is_dir() else []


def current_verdicts(verdicts: Sequence[Verdict], texts: Texts) -> list[Verdict]:
    """Verdicts whose row (reference unit or orphan row) still exists with the fingerprint
    they were made on."""
    fp = {s.id: segment_fingerprint(s) for s in texts.units}
    fp.update({orphan_row_id(s.id): segment_fingerprint(s) for s in texts.witness})
    return [v for v in verdicts if fp.get(v.unit_id) == v.fingerprint]


def codebook(ctx: RunContext) -> TopicCodebook:
    return load_codebook(ctx.data_dir / "codebook" / "topics.yaml")


@dataclass(frozen=True)
class Topics:
    """Current human topic labels of the reference units, and what they imply."""

    labels: Mapping[str, TopicLabel]
    stale: tuple[str, ...]
    n_units: int

    @property
    def complete(self) -> bool:
        return self.n_units > 0 and sum(1 for x in self.labels.values() if x.topics) == self.n_units

    def groups(self) -> dict[str, str]:
        """Unit id -> topic group for labelled units (unlabelled units are absent)."""
        return {u: str(topic_group(x.topics)) for u, x in self.labels.items() if x.topics}

    def topic_sets(self) -> dict[str, frozenset[str]]:
        return {u: x.topics for u, x in self.labels.items() if x.topics}


def load_topics(ctx: RunContext, texts: Texts) -> Topics:
    """Committed topic labels (``data/annotations/topics/<reference>.csv``), fingerprint-checked."""
    path = topic_labels_path(ctx, texts.reference_id)
    if not path.is_file():
        return Topics({}, (), len(texts.units))
    current, stale = current_labels(load_labels(path, codebook(ctx)), texts.units)
    return Topics(current, stale, len(texts.units))


def review_plans(ctx: RunContext) -> list[sampling.ReviewItem]:
    """Every item of every ``review/plan_<batch>.csv`` of this run."""
    items: list[sampling.ReviewItem] = []
    for path in sorted(ctx.path("review").glob("plan_*.csv")) if ctx.path("review").is_dir() else []:
        items.extend(sampling.read_plan(path))
    return items


def plan_path(ctx: RunContext, batch: str) -> Path:
    return ctx.path("review", f"plan_{batch}.csv")


# --------------------------------------------------------------------------- frozen strata
def strata_path(ctx: RunContext, batch: str) -> Path:
    """``review/strata_<batch>.json``: every unit's stratum when plan ``batch`` was drawn."""
    return ctx.path("review", f"strata_{batch}.json")


def write_strata_snapshot(ctx: RunContext, batch: str, strata: Mapping[str, str],
                          items: Iterable[sampling.ReviewItem], topics_complete: bool) -> Path:
    """Freeze the strata of a plan (synthesis 6.2: strata are fixed at sampling time).

    ``families`` are the stratum prefixes the plan samples (``pos``, ``unresolved`` for a
    verification plan, ``neg`` for an audit): the snapshot fixes the units of those families.
    """
    families = sorted({i.stratum.split(":")[0] for i in items if i.stratum})
    path = strata_path(ctx, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"families": families, "topics_complete": topics_complete,
                                "strata": dict(sorted(strata.items()))}, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def sampling_strata(ctx: RunContext, current: Mapping[str, str]) -> dict[str, str]:
    """Unit -> stratum, frozen at sampling time where a plan's snapshot fixes it.

    ``current`` (``sampling.machine_strata`` on today's topic labels) covers the units no
    snapshot fixes, e.g. machine negatives before the audit is drawn. Without this, a unit
    whose topic label changed after sampling would fall into a stratum no verdict calibrates.
    """
    out = dict(current)
    review = ctx.path("review")
    for path in sorted(review.glob("strata_*.json")) if review.is_dir() else []:
        doc = json.loads(path.read_text(encoding="utf-8"))
        families = set(doc.get("families") or ())
        out.update({u: s for u, s in (doc.get("strata") or {}).items() if s.split(":")[0] in families})
    return out

