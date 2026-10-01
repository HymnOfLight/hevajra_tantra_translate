"""Phase-2 samples, the budgeted review queue and coverage (synthesis 6.2, 7.1).

Strata are ``stats.twophase.stratum_of`` of the MACHINE cells (build the matrix without
gold and verdicts for sampling), so sampling and estimation can never disagree:

    pos:<class>:<topic>   machine positive; <class> is a census or a sampled class of
                          ``preregistration.yaml: verification``; <topic> sensitive | other
    neg:<B|C>:<topic>     machine negative (audit strata)
    unresolved            UNALIGNED, including every grade X cell (census class grade_x;
                          these items get task ``resolve``: there is no proposal to verify)
    pos:witness_only      orphan rows (witness-only claims; census class witness_only),
                          except rows on a witness ``note`` segment (``note_rows``): those
                          notes are classified deterministically at ingest, so they are
                          neither verified nor counted in E5 (a descriptive count instead)

Units already decided by a human (``verified``) are removed from every pool first: the
pool U_h of stratum h is its unverified units, and an item's inclusion probability is
n_h / |U_h| (1.0 for census strata). Draws use ``random.Random(seed)`` over pools sorted by
id, so a plan is reproducible from the pre-registered seed.

Verification (machine positives)
    census classes   every unit, never capped
    sampled classes  every unit while census + sampled units fit ``cap_units``; otherwise
                     ceil(sampled_fraction * |U_h|) per stratum, and if even that exceeds the
                     room left under the cap, the room is shared out proportionally to |U_h|
                     (largest remainder). The cap never touches a census class.
    floor            every sampled stratum gets at least min(|U_h|, ``min_per_sampled_stratum``)
                     (default 20) units, even when the census alone exceeds ``cap_units``:
                     a stratum with no phase-2 verdict could not be estimated (it would be
                     uncalibrated), so the floor may take the plan above the cap.
Audit (machine negatives)
    the four strata grade (B, C) x topic (sensitive, other) get min(min_per_stratum, |U_h|)
    each; the rest of ``total`` is shared proportionally to |U_h| x (grade_c_oversample for
    grade C), largest remainder, never above |U_h|.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Collection, Iterable, Mapping, Sequence

from ..config import ConfigError, from_mapping
from ..core.ids import sort_key
from ..core.io import read_csv, write_csv
from ..core.types import Cell, Grade, Verdict
from ..stats.twophase import UNRESOLVED, stratum_of
from .verdicts import TASKS, is_orphan

WITNESS_ONLY_STRATUM = "pos:witness_only"
GRADE_X_CLASS = "grade_x"
WITNESS_ONLY_CLASS = "witness_only"
AUDIT_GRADES = (Grade.B, Grade.C)
TOPIC_LEVELS = ("sensitive", "other")

# Priority when the budget is cut (synthesis 7.1): gold 1-3, topics 4, resolve 5,
# verify census 6, audit 7, verify sampled 8.
PRIORITY: Mapping[str, int] = MappingProxyType({
    "gold": 1, "topics": 4, "topics_second": 4, "resolve": 5, "verify_census": 6, "audit": 7, "verify_sampled": 8,
})
QUEUE_TASKS = (*TASKS, "topics", "topics_second")
PLAN_COLUMNS = ("item_id", "task", "unit_id", "stratum", "inclusion_prob", "priority")


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class ReviewParams:
    """``config/run.yaml: review``: minutes per item, required competence, quote limits."""

    minutes: Mapping[str, float] = field(default_factory=dict)
    competence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    quote_max_chars: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, run: Mapping[str, Any]) -> "ReviewParams":
        p = from_mapping(cls, run.get("review"), "run.yaml: review")
        return cls(minutes=MappingProxyType({k: float(v) for k, v in dict(p.minutes).items()}),
                   competence=MappingProxyType({k: tuple(v) for k, v in dict(p.competence).items()}),
                   quote_max_chars=MappingProxyType({k: int(v) for k, v in dict(p.quote_max_chars).items()}))

    def minutes_for(self, task: str) -> float:
        base = task.removesuffix("_second")
        if base not in self.minutes:
            raise ConfigError(f"run.yaml: review.minutes has no entry for task {base!r}")
        return self.minutes[base]

    def competence_for(self, task: str) -> frozenset[str]:
        base = task.removesuffix("_second")
        if base not in self.competence:
            raise ConfigError(f"run.yaml: review.competence has no entry for task {base!r}")
        return frozenset(self.competence[base])


@dataclass(frozen=True)
class VerificationSpec:
    """``preregistration.yaml: verification``."""

    census_classes: tuple[str, ...]
    sampled_classes: tuple[str, ...]
    cap_units: int
    sampled_fraction: float
    seed: int
    min_per_sampled_stratum: int = 20   # floor per sampled stratum, applied even above the cap

    @classmethod
    def from_config(cls, prereg: Mapping[str, Any]) -> "VerificationSpec":
        spec = from_mapping(cls, prereg.get("verification"), "preregistration.yaml: verification")
        if set(spec.census_classes) & set(spec.sampled_classes):
            raise ConfigError("preregistration.yaml: verification: a class is both census and sampled")
        if not 0 < spec.sampled_fraction <= 1:
            raise ConfigError("preregistration.yaml: verification.sampled_fraction must be in (0, 1]")
        return spec


@dataclass(frozen=True)
class AuditSpec:
    """``preregistration.yaml: audit``."""

    total: int
    min_per_stratum: int
    grade_c_oversample: float
    seed: int

    @classmethod
    def from_config(cls, prereg: Mapping[str, Any]) -> "AuditSpec":
        return from_mapping(cls, prereg.get("audit"), "preregistration.yaml: audit")


@dataclass(frozen=True)
class ReviewItem:
    """One unit (or orphan row) to review in one task, with its sampling design."""

    item_id: str
    task: str
    unit_id: str
    stratum: str = ""
    inclusion_prob: float = 1.0
    priority: int = 0


def make_item(task: str, unit_id: str, stratum: str = "", inclusion_prob: float = 1.0,
              priority: int | None = None) -> ReviewItem:
    """Item with id ``<task>:<unit id>`` and the task's default priority."""
    return ReviewItem(f"{task}:{unit_id}", task, unit_id, stratum, inclusion_prob,
                      PRIORITY.get(task, 9) if priority is None else priority)


# --------------------------------------------------------------------------- strata
def machine_strata(cells: Iterable[Cell], topics: Mapping[str, str], ref_kinds: Mapping[str, str]) -> dict[str, str]:
    """Row id -> sampling stratum (orphan rows -> ``pos:witness_only``).

    ``topics`` maps unit id -> topic group (``topics.topic_group``); missing units count as
    "other". ``ref_kinds`` maps unit id -> reference segment kind; it is required because a
    transliterated mantra is a negative but a transliterated prose unit a positive. Grade A
    cells are refused by ``stratum_of``: pass machine cells.
    """
    kinds = ref_kinds
    out = {}
    for cell in cells:
        if is_orphan(cell.unit_id):
            out[cell.unit_id] = WITNESS_ONLY_STRATUM
        else:
            out[cell.unit_id] = stratum_of(cell, topics.get(cell.unit_id, ""), kinds.get(cell.unit_id, ""))
    return out


NOTE_KIND = "note"


def note_rows(cells: Iterable[Cell], witness_kinds: Mapping[str, str]) -> set[str]:
    """Orphan row ids whose witness segments are all ``note`` segments (``witness_kinds``:
    witness segment id -> kind). Such notes are classified at ingest (``ingest.notes``);
    a machine witness-only claim on one adds nothing to verify."""
    return {c.unit_id for c in cells if is_orphan(c.unit_id) and c.wit_ids
            and all(witness_kinds.get(w) == NOTE_KIND for w in c.wit_ids)}


def stratum_class(stratum: str) -> str:
    """Verification class of a stratum: ``pos:absent:other`` -> ``absent``; ``unresolved`` -> ``grade_x``."""
    if stratum == UNRESOLVED:
        return GRADE_X_CLASS
    parts = stratum.split(":")
    return parts[1] if parts[0] == "pos" and len(parts) >= 2 else ""


def _pools(strata: Mapping[str, str], verified: Collection[str]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    done = set(verified)
    for uid, stratum in strata.items():
        if uid not in done:
            pools.setdefault(stratum, []).append(uid)
    return {s: sorted(units, key=sort_key) for s, units in sorted(pools.items())}


def _draw(pools: Mapping[str, list[str]], sizes: Mapping[str, int], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    return {s: sorted(rng.sample(pools[s], sizes[s]), key=sort_key) for s in sorted(sizes) if sizes[s] > 0}


def largest_remainder(total: int, weights: Mapping[str, float], caps: Mapping[str, int]) -> dict[str, int]:
    """Share ``total`` proportionally to ``weights`` in integers, never above ``caps``.

    Units that a capped stratum cannot take are redistributed among the others; ties in the
    remainders go to the stratum that sorts first.
    """
    alloc = {k: 0 for k in weights}
    left = total
    open_keys = [k for k in sorted(weights) if caps[k] > 0 and weights[k] > 0]
    while left > 0 and open_keys:
        w = sum(weights[k] for k in open_keys)
        quotas = {k: left * weights[k] / w for k in open_keys}
        give = {k: min(math.floor(quotas[k]), caps[k] - alloc[k]) for k in open_keys}
        rest = left - sum(give.values())
        for k in sorted(open_keys, key=lambda k: (-(quotas[k] - math.floor(quotas[k])), k)):
            if rest <= 0:
                break
            if give[k] < caps[k] - alloc[k]:
                give[k] += 1
                rest -= 1
        if not any(give.values()):
            break
        for k in open_keys:
            alloc[k] += give[k]
        left -= sum(give.values())
        open_keys = [k for k in open_keys if alloc[k] < caps[k]]
    return alloc


# --------------------------------------------------------------------------- draws
def draw_verification(cells: Iterable[Cell], topics: Mapping[str, str], spec: VerificationSpec,
                      seed: int | None = None, *, ref_kinds: Mapping[str, str],
                      verified: Collection[str] = ()) -> list[ReviewItem]:
    """Census and (capped) sample of machine positives, UNALIGNED units and witness-only rows.

    ``seed`` defaults to ``spec.seed`` (the pre-registered one); ``verified`` are row ids a
    human already decided (they leave the pools).
    """
    strata = machine_strata(cells, topics, ref_kinds)
    pools = _pools(strata, verified)
    census: dict[str, list[str]] = {}
    sampled: dict[str, list[str]] = {}
    for stratum, units in pools.items():
        cls = stratum_class(stratum)
        if not cls:
            continue                                   # machine negatives and excluded cells
        if cls in spec.census_classes:
            census[stratum] = units
        elif cls in spec.sampled_classes:
            sampled[stratum] = units
        else:
            raise ValueError(f"stratum {stratum}: class {cls!r} is neither a census nor a sampled class")
    n_census = sum(len(u) for u in census.values())
    room = max(spec.cap_units - n_census, 0)
    sizes = {s: len(u) for s, u in sampled.items()}
    if sum(sizes.values()) > room:
        sizes = {s: math.ceil(spec.sampled_fraction * len(u)) for s, u in sampled.items()}
        if sum(sizes.values()) > room:
            sizes = largest_remainder(room, {s: float(len(u)) for s, u in sampled.items()}, sizes)
        sizes = {s: max(n, min(len(sampled[s]), spec.min_per_sampled_stratum)) for s, n in sizes.items()}
    drawn = _draw(sampled, sizes, spec.seed if seed is None else seed)
    items = []
    for stratum, units in census.items():
        task = "resolve" if stratum == UNRESOLVED else "verify"
        priority = PRIORITY["resolve"] if task == "resolve" else PRIORITY["verify_census"]
        items += [make_item(task, u, stratum, 1.0, priority) for u in units]
    for stratum, units in drawn.items():
        p = len(units) / len(sampled[stratum])
        items += [make_item("verify", u, stratum, p, PRIORITY["verify_sampled"]) for u in units]
    return _ordered(items)


def audit_strata() -> tuple[str, ...]:
    return tuple(f"neg:{g.value}:{t}" for g in AUDIT_GRADES for t in TOPIC_LEVELS)


def draw_audit(cells: Iterable[Cell], topics: Mapping[str, str], spec: AuditSpec, seed: int | None = None, *,
               ref_kinds: Mapping[str, str], verified: Collection[str] = ()) -> list[ReviewItem]:
    """Stratified blind audit of machine negatives (grade B/C x topic group)."""
    pools = _pools(machine_strata(cells, topics, ref_kinds), verified)
    names = audit_strata()
    size = {s: len(pools.get(s, ())) for s in names}
    base = {s: min(spec.min_per_stratum, size[s]) for s in names}
    weights = {s: size[s] * (spec.grade_c_oversample if s.startswith("neg:C:") else 1.0) for s in names}
    extra = largest_remainder(max(spec.total - sum(base.values()), 0), weights,
                              {s: size[s] - base[s] for s in names})
    sizes = {s: base[s] + extra[s] for s in names}
    drawn = _draw({s: pools.get(s, []) for s in names}, sizes, spec.seed if seed is None else seed)
    items = [make_item("audit", u, s, len(units) / size[s]) for s, units in drawn.items() for u in units]
    return _ordered(items)


def _ordered(items: Sequence[ReviewItem]) -> list[ReviewItem]:
    return sorted(items, key=lambda i: (i.priority, sort_key(i.unit_id)))


# --------------------------------------------------------------------------- queue and coverage
def queue(items: Iterable[ReviewItem], budget_minutes: float, competence: Collection[str],
          params: ReviewParams, done: Collection[str] = ()) -> list[ReviewItem]:
    """Items a reviewer with ``competence`` (e.g. {"bo", "zh"}) can do in ``budget_minutes``.

    Items are taken in priority order (then their given order) and the queue stops at the
    first item that no longer fits, so a lower-priority task never jumps ahead. Items whose
    task needs a language the reviewer lacks are skipped; ``done`` item ids are skipped.
    """
    have = frozenset(competence)
    finished = set(done)
    ordered = sorted(enumerate(items), key=lambda p: (p[1].priority, p[0]))
    out: list[ReviewItem] = []
    spent = 0.0
    for _, item in ordered:
        if item.item_id in finished or not params.competence_for(item.task) <= have:
            continue
        cost = params.minutes_for(item.task)
        if spent + cost > budget_minutes + 1e-9:
            break
        spent += cost
        out.append(item)
    return out


def coverage(items: Iterable[ReviewItem], verdicts: Iterable[Verdict]) -> dict[str, dict[str, int]]:
    """Per stratum: planned items and how many have a blind and a final decision.

    A verdict counts for an item when it names the same item id and unit.
    """
    blind: set[tuple[str, str]] = set()
    final: set[tuple[str, str]] = set()
    for v in verdicts:
        if v.blind_relation:
            blind.add((v.item_id, v.unit_id))
        if v.final_relation:
            final.add((v.item_id, v.unit_id))
    out: dict[str, dict[str, int]] = {}
    for item in items:
        row = out.setdefault(item.stratum or item.task, {"planned": 0, "blind": 0, "final": 0})
        key = (item.item_id, item.unit_id)
        row["planned"] += 1
        row["blind"] += key in blind
        row["final"] += key in final
    return dict(sorted(out.items()))


def format_coverage(cov: Mapping[str, Mapping[str, int]]) -> list[str]:
    """``["pos:reversal:other final 12/12 (blind 12)", ...]`` for ``review status``."""
    return [f"{s} final {c['final']}/{c['planned']} (blind {c['blind']})" for s, c in cov.items()]


# --------------------------------------------------------------------------- plan files
def write_plan(items: Iterable[ReviewItem], path: Path) -> None:
    """``plan_<batch>.csv``: the drawn items with strata and inclusion probabilities."""
    write_csv(Path(path), [{"item_id": i.item_id, "task": i.task, "unit_id": i.unit_id, "stratum": i.stratum,
                            "inclusion_prob": f"{i.inclusion_prob:.6g}", "priority": i.priority} for i in items],
              PLAN_COLUMNS)


def read_plan(path: Path) -> list[ReviewItem]:
    out = []
    for row in read_csv(Path(path)):
        if row.get("task") not in QUEUE_TASKS:
            raise ValueError(f"{path}: unknown task {row.get('task')!r}")
        out.append(ReviewItem(row["item_id"], row["task"], row["unit_id"], row["stratum"],
                              float(row["inclusion_prob"] or 1.0), int(row["priority"] or 0)))
    return out
