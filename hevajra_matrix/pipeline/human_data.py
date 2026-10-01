"""Read the committed human annotations (gold, verdicts, topic labels) and the run's
review plans, in the form the stages need them.

Committed files live under ``data/annotations/`` (ids and short quotes only). Review plans
(``plan_<batch>.csv``: item ids, strata and inclusion probabilities) and their frozen strata
(``strata_<batch>.json``) are committed next to the verdicts they calibrate, under
``data/annotations/verdicts/<witness>/``, so a new run directory still finds them. Runs made
before plans were committed kept them under ``<run>/review/``; those are read as a fallback
for any batch with no committed plan.
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


def annotations_dir(ctx: RunContext) -> Path:
    """``data/annotations/`` under the provisional Derge reference; under the Sanskrit
    reference ``data/annotations/by_reference/<reference>/`` with the same layout, because
    gold, verdicts and topic labels are made against the reference units."""
    base = ctx.data_dir / "annotations"
    return base if ctx.reference == ctx.derge else base / "by_reference" / ctx.reference


def gold_dir(ctx: RunContext, witness: str) -> Path:
    return annotations_dir(ctx) / "gold" / witness


def verdict_dir(ctx: RunContext, witness: str) -> Path:
    return annotations_dir(ctx) / "verdicts" / witness


def topic_labels_path(ctx: RunContext, reference: str) -> Path:
    return annotations_dir(ctx) / "topics" / f"{reference}.csv"


def ledger_path(ctx: RunContext) -> Path:
    """The evaluation ledger of this run's test set: ``test_evaluations.jsonl`` for the Derge
    reference and the target; every other (reference, witness) pair has its own test gold
    and so its own ledger ``test_evaluations.<reference>.<witness>.jsonl``."""
    if (ctx.reference, ctx.witness) == (ctx.derge, ctx.default_witness):
        return ctx.data_dir / "ledger" / "test_evaluations.jsonl"
    return ctx.data_dir / "ledger" / f"test_evaluations.{ctx.reference}.{ctx.witness}.jsonl"


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


PLAN_PREFIX, STRATA_PREFIX = "plan_", "strata_"


def _design_files(ctx: RunContext, witness: str, prefix: str, suffix: str) -> list[Path]:
    """Committed design files of ``witness``, then run-directory ones of batches not committed."""
    committed = sorted(verdict_dir(ctx, witness).glob(f"{prefix}*{suffix}")) if verdict_dir(ctx, witness).is_dir() else []
    names = {p.name for p in committed}
    review = ctx.path("review")
    legacy = [p for p in sorted(review.glob(f"{prefix}*{suffix}")) if p.name not in names] if review.is_dir() else []
    return committed + legacy


def review_plans(ctx: RunContext, texts: Texts) -> list[sampling.ReviewItem]:
    """Every item of every plan (committed ``plan_<batch>.csv``; old runs: ``<run>/review/``),
    without witness-only rows on ``note`` segments that a plan drawn before they were left
    out may hold (they are never exported, so G3 must not wait for them)."""
    notes = {orphan_row_id(s.id) for s in texts.witness if s.kind == sampling.NOTE_KIND}
    return [i for path in _design_files(ctx, texts.witness_id, PLAN_PREFIX, ".csv") for i in sampling.read_plan(path)
            if i.unit_id not in notes]


def plan_path(ctx: RunContext, witness: str, batch: str) -> Path:
    """Where plan ``batch`` is committed (``data/annotations/verdicts/<witness>/plan_<batch>.csv``)."""
    return verdict_dir(ctx, witness) / f"{PLAN_PREFIX}{batch}.csv"


def find_plan(ctx: RunContext, witness: str, batch: str) -> Path:
    """The committed plan of ``batch``, else the run directory's (runs made before plans
    were committed); the committed path when neither exists."""
    committed = plan_path(ctx, witness, batch)
    legacy = ctx.path("review", f"{PLAN_PREFIX}{batch}.csv")
    return legacy if not committed.is_file() and legacy.is_file() else committed


# --------------------------------------------------------------------------- frozen strata
def strata_path(ctx: RunContext, witness: str, batch: str) -> Path:
    """``strata_<batch>.json`` beside the plan: every unit's stratum when the plan was drawn."""
    return verdict_dir(ctx, witness) / f"{STRATA_PREFIX}{batch}.json"


def write_strata_snapshot(ctx: RunContext, witness: str, batch: str, strata: Mapping[str, str],
                          items: Iterable[sampling.ReviewItem], topics_complete: bool) -> Path:
    """Freeze the strata of a plan (synthesis 6.2: strata are fixed at sampling time).

    ``families`` are the stratum prefixes the plan samples (``pos``, ``unresolved`` for a
    verification plan, ``neg`` for an audit): the snapshot fixes the units of those families.
    """
    families = sorted({i.stratum.split(":")[0] for i in items if i.stratum})
    path = strata_path(ctx, witness, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"families": families, "topics_complete": topics_complete,
                                "strata": dict(sorted(strata.items()))}, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def sampling_strata(ctx: RunContext, witness: str, current: Mapping[str, str]) -> dict[str, str]:
    """Unit -> stratum, frozen at sampling time where a plan's snapshot fixes it.

    ``current`` (``sampling.machine_strata`` on today's topic labels) covers the units no
    snapshot fixes, e.g. machine negatives before the audit is drawn. Without this, a unit
    whose topic label changed after sampling would fall into a stratum no verdict calibrates.
    """
    out = dict(current)
    for path in _design_files(ctx, witness, STRATA_PREFIX, ".json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        families = set(doc.get("families") or ())
        out.update({u: s for u, s in (doc.get("strata") or {}).items() if s.split(":")[0] in families})
    return out

