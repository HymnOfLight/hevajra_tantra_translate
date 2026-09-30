"""Design of the over-attribution experiment: items, evidence lines and the trial schedule.

The experiment (synthesis section 8) asks whether Claude still names a translator motive
for an omission in the Chinese translation when the evidence it is shown explains the
omission by the source side, and whether that happens more for sensitive content.

    items    = load_items(data/experiments/overattribution/items.csv)   # coordinates only
    evidence = load_evidence(data/experiments/overattribution/evidence.yaml)
    schedule = trials(main_items(items), CONDITIONS, replicates=3, seed=prereg seed)

Design
    * Content arm (sensitive | neutral) varies between items; each neutral item is matched
      to one sensitive item (``pair_id``) from the same or an adjacent chapter.
    * Evidence varies within item: E0 (no line), EP (length-matched placebo) and EW (a
      witness fact). The EW version (EW-V manuscripts, EW-S independent translation) is
      assigned per item, balanced within each arm, from the seed.
    * Every item x condition is run ``replicates`` times; replicates are averaged within
      item x condition before any test, so the item is the unit of analysis.
    * The schedule order is a seeded shuffle of all trials. Calls are stateless, so order
      cannot carry over between trials; shuffling only spreads any drift of the served model
      over the campaign evenly across conditions.

The item bank holds coordinates only (licence rule). Items come from a ~300-unit
human-labelled candidate pool (critique B12); ``constructed`` omissions, where the
counterpart clauses are removed from the Chinese context shown, are allowed and are a
covariate. The single documented-truth case (the note at T0892:0592a27; critique A3) has no
Derge row and is analysed qualitatively outside this code.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from ...config import from_mapping
from ...core.io import read_csv, read_yaml

Arm = Literal["sensitive", "neutral"]
Condition = Literal["E0", "EP", "EW"]
Evidence = Literal["none", "EP", "EW-V", "EW-S"]

ARMS: tuple[str, ...] = ("sensitive", "neutral")
CONDITIONS: tuple[str, ...] = ("E0", "EP", "EW")
EW_VERSIONS: tuple[str, ...] = ("EW-V", "EW-S")
ORIGINS: tuple[str, ...] = ("real", "constructed")
PHASES: tuple[str, ...] = ("pilot", "main")
EVIDENCE_KEYS: tuple[str, ...] = ("EP",) + EW_VERSIONS

ITEM_COLUMNS: tuple[str, ...] = (
    "item_id", "phase", "arm", "pair_id", "chapter", "unit_ids", "omission_origin",
    "zh_context_from", "zh_context_to", "synthetic_facts", "note",
)
UNIT_SEPARATOR = ";"
_TRUE, _FALSE = ("1", "true", "yes"), ("0", "false", "no")


class DesignError(ValueError):
    """An invalid item bank, evidence file or design parameter."""


@dataclass(frozen=True)
class Item:
    """One passage whose Chinese counterpart is absent from the context shown.

    ``unit_ids``           reference (Tibetan) unit ids of the passage, in text order
    ``zh_context_from``    id of the last Chinese segment shown BEFORE the gap
    ``zh_context_to``      id of the first Chinese segment shown AFTER the gap; every content
                           segment strictly between the two is left out of the prompt (none
                           for a ``real`` omission, the counterpart clauses for a
                           ``constructed`` one)
    ``synthetic_facts``    True when the EW line shown is a generic synthetic fact (always,
                           in v1); recorded so that no output is mistaken for evidence
    """

    item_id: str
    arm: str
    unit_ids: tuple[str, ...]
    chapter: str
    pair_id: str
    omission_origin: str
    zh_context_from: str
    zh_context_to: str
    synthetic_facts: bool = True
    phase: str = "main"
    note: str = ""


@dataclass(frozen=True)
class EvidenceLine:
    """One evidence line and the scorer classes it points to (for Y_uptake)."""

    key: str
    text: str
    implies: frozenset[str]


@dataclass(frozen=True)
class Trial:
    """One subject call: an item under one condition, one replicate.

    ``evidence`` is the line actually shown ("none" for E0, "EP", or the item's EW version);
    ``order`` is the position in the seeded schedule.
    """

    trial_id: str
    item: Item
    condition: str
    evidence: str
    replicate: str
    order: int

    @property
    def item_id(self) -> str:
        return self.item.item_id

    @property
    def arm(self) -> str:
        return self.item.arm


@dataclass(frozen=True)
class ExperimentParams:
    """``config/preregistration.yaml: experiment`` (unknown keys raise ``ConfigError``)."""

    items_per_arm: int = 60
    pilot_items_per_arm: int = 10
    replicates: int = 3
    conditions: tuple[str, ...] = CONDITIONS
    seed: int = 0
    human_coded_responses: int = 150

    @classmethod
    def from_prereg(cls, prereg: Mapping[str, object]) -> "ExperimentParams":
        return from_mapping(cls, prereg.get("experiment"), "preregistration.yaml: experiment")  # type: ignore[arg-type]


def bank_shortfalls(items: Sequence[Item], params: ExperimentParams) -> list[str]:
    """How the item bank falls short of the pre-registered size (empty list = complete)."""
    targets = {"main": params.items_per_arm, "pilot": params.pilot_items_per_arm}
    have = Counter((it.phase, it.arm) for it in items)
    return [f"{phase}/{arm}: {have[(phase, arm)]} of {n} items"
            for phase, n in targets.items() for arm in ARMS if have[(phase, arm)] < n]


# --------------------------------------------------------------------------- items
def load_items(path: Path) -> list[Item]:
    """Read and validate ``items.csv``. Raises ``DesignError`` listing every problem."""
    rows = read_csv(path)
    if rows:
        missing = sorted(set(ITEM_COLUMNS) - set(rows[0]))
        if missing:
            raise DesignError(f"{path}: missing column(s) {missing}")
    errors: list[str] = []
    items = [_item(i, row, errors) for i, row in enumerate(rows, start=2)]
    errors += validate_items(items)
    if errors:
        raise DesignError(f"{path}: " + "; ".join(errors[:20]))
    return items


def _item(line: int, row: Mapping[str, str], errors: list[str]) -> Item:
    def cell(name: str) -> str:
        return (row.get(name) or "").strip()

    flag = cell("synthetic_facts").lower()
    if flag not in _TRUE + _FALSE:
        errors.append(f"line {line}: synthetic_facts must be true or false, not {flag!r}")
    units = tuple(u.strip() for u in cell("unit_ids").split(UNIT_SEPARATOR) if u.strip())
    return Item(item_id=cell("item_id"), arm=cell("arm"), unit_ids=units, chapter=cell("chapter"),
                pair_id=cell("pair_id"), omission_origin=cell("omission_origin"),
                zh_context_from=cell("zh_context_from"), zh_context_to=cell("zh_context_to"),
                synthetic_facts=flag in _TRUE, phase=cell("phase") or "main", note=cell("note"))


def validate_items(items: Sequence[Item]) -> list[str]:
    """Problems in an item bank (empty list = valid).

    Checks field vocabularies, unique ids, non-empty coordinates, and the matching rule:
    every ``pair_id`` holds exactly one sensitive and one neutral item of the same phase.
    """
    errors: list[str] = []
    for it in items:
        where = f"item {it.item_id or '?'}"
        if not it.item_id:
            errors.append("an item has an empty item_id")
        for field_name, allowed in (("arm", ARMS), ("omission_origin", ORIGINS), ("phase", PHASES)):
            value = getattr(it, field_name)
            if value not in allowed:
                errors.append(f"{where}: {field_name} must be one of {allowed}, not {value!r}")
        if not it.unit_ids:
            errors.append(f"{where}: unit_ids is empty")
        if len(set(it.unit_ids)) != len(it.unit_ids):
            errors.append(f"{where}: repeated unit id")
        for field_name in ("chapter", "pair_id", "zh_context_from", "zh_context_to"):
            if not getattr(it, field_name):
                errors.append(f"{where}: {field_name} is empty")
        if it.zh_context_from and it.zh_context_from == it.zh_context_to:
            errors.append(f"{where}: zh_context_from equals zh_context_to")
    repeated = [k for k, n in Counter(it.item_id for it in items).items() if n > 1]
    if repeated:
        errors.append(f"duplicate item_id(s) {sorted(repeated)[:5]}")
    pairs: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        pairs[it.pair_id].append(it)
    for pair_id, members in sorted(pairs.items()):
        arms = sorted(m.arm for m in members)
        if arms != ["neutral", "sensitive"]:
            errors.append(f"pair {pair_id!r}: needs one sensitive and one neutral item, has {arms}")
        elif members[0].phase != members[1].phase:
            errors.append(f"pair {pair_id!r}: items are in different phases")
    return errors


def items_in_phase(items: Iterable[Item], phase: str) -> list[Item]:
    """The pilot items are run first and excluded from the main analysis."""
    if phase not in PHASES:
        raise DesignError(f"phase must be one of {PHASES}, not {phase!r}")
    return [it for it in items if it.phase == phase]


# --------------------------------------------------------------------------- evidence
def load_evidence(path: Path) -> dict[str, EvidenceLine]:
    """Read ``evidence.yaml``: exactly the keys EP, EW-V and EW-S, each with text and implies."""
    doc = read_yaml(path)
    if not isinstance(doc, Mapping) or not isinstance(doc.get("lines"), Mapping):
        raise DesignError(f"{path}: expected a mapping with 'lines'")
    lines = doc["lines"]
    if set(lines) != set(EVIDENCE_KEYS):
        raise DesignError(f"{path}: lines must be exactly {EVIDENCE_KEYS}, found {sorted(lines)}")
    out: dict[str, EvidenceLine] = {}
    for key in EVIDENCE_KEYS:
        rec = lines[key]
        if not isinstance(rec, Mapping) or set(rec) != {"text", "implies"}:
            raise DesignError(f"{path}: lines.{key} needs exactly 'text' and 'implies'")
        text = " ".join(str(rec["text"] or "").split())
        if not text:
            raise DesignError(f"{path}: lines.{key}.text is empty")
        out[key] = EvidenceLine(key=key, text=text, implies=frozenset(rec["implies"] or ()))
    return out


# --------------------------------------------------------------------------- schedule
def ew_versions(items: Sequence[Item], seed: int) -> dict[str, str]:
    """Item id -> EW version, balanced within each arm (counts differ by at most one).

    Within an arm, items are sorted by id, shuffled with a seed derived from ``seed`` and the
    arm, and assigned the two versions alternately; the result depends only on the ids.
    """
    out: dict[str, str] = {}
    for arm in ARMS:
        ids = sorted(it.item_id for it in items if it.arm == arm)
        random.Random(f"{seed}:ew:{arm}").shuffle(ids)
        out.update({item_id: EW_VERSIONS[i % 2] for i, item_id in enumerate(ids)})
    return out


def trials(items: Sequence[Item], conditions: Sequence[str] = CONDITIONS, replicates: int = 3,
           seed: int = 0) -> list[Trial]:
    """Every item x condition x replicate, in a seeded random order (``Trial.order``).

    The same items, conditions, replicates and seed always give the same schedule.
    """
    if replicates < 1:
        raise DesignError("replicates must be >= 1")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown or len(set(conditions)) != len(conditions) or not conditions:
        raise DesignError(f"conditions must be distinct values from {CONDITIONS}, not {list(conditions)}")
    errors = validate_items(items)
    if errors:
        raise DesignError("; ".join(errors[:20]))
    versions = ew_versions(items, seed)
    cells = [(it, c, f"r{r}") for it in sorted(items, key=lambda i: i.item_id)
             for c in conditions for r in range(1, replicates + 1)]
    random.Random(f"{seed}:order").shuffle(cells)
    evidence = {"E0": lambda it: "none", "EP": lambda it: "EP", "EW": lambda it: versions[it.item_id]}
    return [Trial(trial_id=trial_id(it.item_id, c, rep), item=it, condition=c,
                  evidence=evidence[c](it), replicate=rep, order=n)
            for n, (it, c, rep) in enumerate(cells)]


def trial_id(item_id: str, condition: str, replicate: str) -> str:
    """Stable trial id: ``<item>:<condition>:<replicate>``."""
    return f"{item_id}:{condition}:{replicate}"


def double_score_ids(trial_ids: Iterable[str], fraction: float, seed: int) -> frozenset[str]:
    """The seeded ``fraction`` of trials whose explanation is scored a second time.

    The repeat estimates the scorer's own stochasticity (synthesis section 8: 20%).
    """
    if not 0.0 <= fraction <= 1.0:
        raise DesignError("fraction must be between 0 and 1")
    ids = sorted(set(trial_ids))
    k = round(fraction * len(ids))
    return frozenset(random.Random(f"{seed}:double").sample(ids, k))
