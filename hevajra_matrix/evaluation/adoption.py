"""The collate.v1/v2 adoption rule of docs-zh/09-collate-v2-draft.md section 7, made mechanical.

Fixed on 2026-10-13, before any dev result, and not changed by results. On the dev gold (every
preregistered dev region), the consensus of a collate.v1 run and of a collate.v2 run, each of the
dev chapters collated whole with ``REPLICATES`` replicates:

    (1) v2's ``relation_kappa_shared`` is at least v1's minus ``TOLERANCE`` (a point-estimate
        tolerance, not a non-inferiority test);
    (2) v2's convention precision is at least ``MIN_CONVENTION_PRECISION``;
    (3) the replicate-kappa floor, symmetric in the two versions: each run's
        ``collation/integrity.json: fleiss_kappa`` against the preregistered
        ``min_replicate_kappa``. If v1 reaches the floor, v2 must too; if v1 does not, the
        floor does not decide, and the chosen version, when below it, is tuned on the dev set
        before ``prereg freeze``.

(1) and (2) are judged on the dev windows outside ``TUNING_WINDOWS`` (I.1, where v2 was
developed); the figures on every dev window are reported, descriptive only. The
``IN_SAMPLE_UNITS`` (I.7 units the second v2 revision was written for) stay in (2); its value
without them is reported, not decisive. v2 is adopted when all three hold, v1 kept otherwise.

Deferred cells. A gold ``?`` is stored as convention "" with a note marker (review guide
docs-zh/03 section 1a). ``read_deferred`` reads it by its marker
(``data/codebook/collate.v2/note_markers.yaml``) as the default of the open question: the
primary reading ``DEFAULT``. A marker reading null (the Hevajra glossary marker) blocks: the
decision is "not ready" until that convention is decided. ``EXCLUDED`` (the ``?`` left out,
as stored) is the sensitivity reading. Every ``?`` must be handled before the comparison (so
that which units enter it does not depend on how many the annotator wrote): a linked row with
convention "" and no known marker also makes the decision "not ready", as does a dev region the
gold does not cover yet (``gold_gaps``), a run without consensus, an undefined value of (1) or
(2), or a missing replicate kappa.

Pure functions; ``pipeline.adoption`` reads the runs and writes ``evaluation/adoption.json``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..config import ConfigError
from ..core.types import Alignment
from ..review.verdicts import CONVENTIONS, takes_convention
from .gold import ROW_REF, GoldRow, GoldSet, score
from .resample import paired_difference, ratio
from .scores import (AlignmentScores, _codebooks_agree, convention_precision, convention_recall, dany_conv_kappa,
                     dany_kappa, relation_kappa_shared)

RULE = "docs-zh/09-collate-v2-draft.md section 7, fixed 2026-10-13"
TOLERANCE = 0.05
MIN_CONVENTION_PRECISION = 0.8
TUNING_WINDOWS = ("I.1",)                                         # dev windows collate.v2 was developed on
IN_SAMPLE_UNITS = ("D417:8b.5.2", "D417:8b.5.3", "D417:8b.5.4")   # the I.7 units of the second v2 revision
REPLICATES = 3                                                    # per version, as in the measurement
EPS = 1e-9                                  # float noise at a threshold counts as reaching it

DEFAULT, EXCLUDED = "default", "excluded"   # a deferred "?" read by its marker (primary), or left out
ALL, OUTSIDE, OUTSIDE_NO_IN_SAMPLE = "all", "outside_tuning", "outside_tuning_no_in_sample"
ADOPT, KEEP, NOT_READY = "adopt collate.v2", "keep collate.v1", "not ready"
MARKER_KEYS = frozenset({"marker", "reads", "question"})


# --------------------------------------------------------------------------- deferred cells
@dataclass(frozen=True)
class NoteMarker:
    """A note marker of a deferred convention and the convention its ``?`` reads as
    (``reads`` None: blocks until decided)."""

    marker: str
    reads: str | None
    question: str = ""


def note_markers(doc: Any, where: str) -> tuple[NoteMarker, ...]:
    """The markers of a parsed ``note_markers.yaml``, in precedence order (the first one a note
    contains decides); unknown keys, conventions and repeated markers are refused."""
    if not isinstance(doc, Mapping):
        raise ConfigError(f"{where}: expected a mapping")
    unknown = sorted(set(doc) - {"schema_version", "markers"})
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: ['markers', 'schema_version']")
    if doc.get("schema_version") != 1:
        raise ConfigError(f"{where}: schema_version 1 required")
    records = doc.get("markers")
    if not isinstance(records, list) or not records:
        raise ConfigError(f"{where}: markers must be a non-empty list")
    out: list[NoteMarker] = []
    for n, rec in enumerate(records):
        at = f"{where}: markers[{n}]"
        if not isinstance(rec, Mapping):
            raise ConfigError(f"{at}: expected a mapping")
        extra = sorted(set(rec) - MARKER_KEYS)
        if extra:
            raise ConfigError(f"{at}: unknown key(s) {extra}; allowed: {sorted(MARKER_KEYS)}")
        marker, reads, question = rec.get("marker"), rec.get("reads"), rec.get("question", "")
        if not isinstance(marker, str) or not marker.strip() or marker in {m.marker for m in out}:
            raise ConfigError(f"{at}: marker must be a non-blank string listed once")
        if "reads" not in rec or (reads is not None and reads not in CONVENTIONS):
            raise ConfigError(f"{at}: reads must be null or one of {', '.join(CONVENTIONS)}")
        if not isinstance(question, str):
            raise ConfigError(f"{at}: question must be a string")
        out.append(NoteMarker(marker, reads, question))
    return tuple(out)


def deciding_marker(note: str, markers: Sequence[NoteMarker]) -> NoteMarker | None:
    """The first marker (precedence order) contained in ``note``, or None."""
    return next((m for m in markers if m.marker in note), None)


def _open(row: GoldRow) -> bool:
    """A ref row with a linked relation whose convention is not annotated ("")."""
    return row.row_type == ROW_REF and not row.convention and takes_convention(row.unit_id, row.relation)


def read_deferred(gold: GoldSet, markers: Sequence[NoteMarker]
                  ) -> tuple[GoldSet, dict[str, int], tuple[str, ...]]:
    """The gold with every deferred ``?`` read by its deciding marker (the ``DEFAULT`` reading),
    the number of rows each marker decided (every marker listed), and the units still blocked
    (a marker reading null; their row stays ""). Rows without a known marker stay "" (not
    annotated); a decided convention is never changed."""
    rows, counts, blocked = [], Counter[str](), []
    for row in gold.rows:
        found = deciding_marker(row.note, markers) if _open(row) else None
        if found is not None:
            counts[found.marker] += 1
            if found.reads is None:
                blocked.append(row.unit_id)
            else:
                row = replace(row, convention=found.reads)
        rows.append(row)
    return replace(gold, rows=tuple(rows)), {m.marker: counts[m.marker] for m in markers}, tuple(blocked)


def gold_gaps(gold: GoldSet, regions: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Dev region id -> the number of its units without a ref row of that window in ``gold``,
    for the regions not fully covered yet (gold is imported one window at a time)."""
    have = {(r.window_id, r.unit_id) for r in gold.ref_rows()}
    gaps = {rid: sum(1 for u in units if (rid, u) not in have) for rid, units in regions.items()}
    return {rid: n for rid, n in gaps.items() if n}


# --------------------------------------------------------------------------- figures of one run
def outside_tuning(s: AlignmentScores) -> AlignmentScores:
    """The records of the windows outside ``TUNING_WINDOWS``."""
    return s.on_windows([w for w in s.windows() if w not in TUNING_WINDOWS])


def without_in_sample(s: AlignmentScores) -> AlignmentScores:
    """The records without ``IN_SAMPLE_UNITS``."""
    return replace(s, units=tuple(u for u in s.units if u.unit_id not in IN_SAMPLE_UNITS))


def dany_level(s: AlignmentScores) -> dict[str, Any]:
    """The D_any share of gold and of the source on the shared units (``scores._codebooks_agree``),
    and predicted minus gold (descriptive: v2 marks somewhat more deviations, docs-zh/09 section
    9.6). Both shares are over every shared unit, so the gold share is the same for each run on
    one gold; a unit the source left unresolved counts as no deviation (``unresolved``: how many)."""
    shared = [u for u in s.units if _codebooks_agree(u)]
    gold = ratio(sum(1 for u in shared if u.gold_dev), len(shared))
    predicted = ratio(sum(1 for u in shared if u.pred_dev), len(shared))
    return {"units": len(shared), "unresolved": sum(1 for u in shared if u.pred_dev is None), "gold": gold,
            "predicted": predicted, "difference": None if gold is None or predicted is None else predicted - gold}


def figures(s: AlignmentScores) -> dict[str, Any]:
    """The reported figures of one scope: the compared metrics, the units counted, the D_any level."""
    return {"windows": list(s.windows()), "units": len(s.units),
            "shared_units": sum(1 for u in s.units if _codebooks_agree(u)),
            "convention_annotated": sum(1 for u in s.units if u.gold_convention is not None),
            "relation_kappa_shared": relation_kappa_shared(s), "convention_precision": convention_precision(s),
            "convention_recall": convention_recall(s), "dany_kappa": dany_kappa(s),
            "dany_conv_kappa": dany_conv_kappa(s), "dany_level": dany_level(s)}


def scopes(s: AlignmentScores) -> dict[str, dict[str, Any]]:
    """One reading on every window (``ALL``), outside ``TUNING_WINDOWS`` (``OUTSIDE``) and, for
    convention precision only, outside them without ``IN_SAMPLE_UNITS``."""
    outside = outside_tuning(s)
    trimmed = without_in_sample(outside)
    return {ALL: figures(s), OUTSIDE: figures(outside),
            OUTSIDE_NO_IN_SAMPLE: {"units": len(trimmed.units), "convention_precision": convention_precision(trimmed)}}


@dataclass(frozen=True)
class RunScores:
    """One run's consensus on the dev gold in both readings of a deferred ``?``."""

    readings: Mapping[str, AlignmentScores]     # DEFAULT, EXCLUDED
    deferred: Mapping[str, int]                 # marker -> deferred rows it decided
    unmarked: tuple[str, ...]                   # units of linked rows with convention "" and no known marker
    blocked: tuple[str, ...]                    # units a null marker still blocks

    def figures(self) -> dict[str, Any]:
        return {"conventions": self.readings[DEFAULT].conventions, "deferred": dict(self.deferred),
                "unmarked": list(self.unmarked), "blocked": list(self.blocked),
                "readings": {name: scopes(s) for name, s in self.readings.items()}}


def score_run(alignment: Alignment, gold: GoldSet, markers: Sequence[NoteMarker], **score_args: Any) -> RunScores:
    """Score one run's consensus on ``gold`` read both ways (``score_args`` go to ``gold.score``:
    ``ref_kinds``, ``unresolved``, ``diagnostics``, ``conventions``)."""
    read, counts, blocked = read_deferred(gold, markers)
    unmarked = tuple(r.unit_id for r in gold.rows if _open(r) and deciding_marker(r.note, markers) is None)
    return RunScores({DEFAULT: score(alignment, read, **score_args), EXCLUDED: score(alignment, gold, **score_args)},
                     counts, unmarked, blocked)


def kappa_difference(v1: AlignmentScores, v2: AlignmentScores, n_boot: int, seed: int) -> dict[str, Any]:
    """v2 minus v1 ``relation_kappa_shared`` outside ``TUNING_WINDOWS``, with its paired
    window-cluster bootstrap interval (descriptive; both scored on the same gold)."""
    windows = [w for w in v2.windows() if w not in TUNING_WINDOWS]
    point, lo, hi = paired_difference(lambda s, ws: relation_kappa_shared(s.on_windows(ws)), v2, v1, windows,
                                      n_boot, seed)
    return {"point": point, "lo": lo, "hi": hi, "windows": windows, "n_boot": n_boot, "seed": seed}


# --------------------------------------------------------------------------- the decision
def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold - EPS


def _value(run: Mapping[str, Any] | None, reading: str, scope: str, key: str) -> Any:
    return None if run is None else run["readings"][reading][scope][key]


def _judge(v1: Mapping[str, Any] | None, v2: Mapping[str, Any] | None, v1_kappa: float | None,
           v2_kappa: float | None, floor: float, reading: str, gaps: Mapping[str, int]) -> dict[str, Any]:
    k1, k2 = (_value(r, reading, OUTSIDE, "relation_kappa_shared") for r in (v1, v2))
    precision = _value(v2, reading, OUTSIDE, "convention_precision")
    applies = v1_kappa is not None and _at_least(v1_kappa, floor)
    conditions = {
        "relation_kappa_shared": {
            "v1": k1, "v2": k2, "tolerance": TOLERANCE, "threshold": None if k1 is None else k1 - TOLERANCE,
            "passed": None if k1 is None or k2 is None else _at_least(k2, k1 - TOLERANCE)},
        "convention_precision": {
            "v2": precision, "threshold": MIN_CONVENTION_PRECISION,
            "v2_without_in_sample": _value(v2, reading, OUTSIDE_NO_IN_SAMPLE, "convention_precision"),
            "passed": None if precision is None else _at_least(precision, MIN_CONVENTION_PRECISION)},
        "replicate_kappa": {
            "v1": v1_kappa, "v2": v2_kappa, "threshold": floor, "applies": applies,     # v1 below: does not decide
            "passed": None if v1_kappa is None or v2_kappa is None
            else (_at_least(v2_kappa, floor) if applies else True)},
    }
    not_ready = [f"the {label} run has no consensus (run build)" for label, r in (("v1", v1), ("v2", v2)) if r is None]
    if gaps:
        not_ready.append("the dev gold does not cover every dev region yet: "
                         + ", ".join(f"{rid} ({n} unit(s) missing)" for rid, n in gaps.items()))
    blocked, unmarked = (sorted({u for r in (v1, v2) if r is not None for u in r[key]})
                         for key in ("blocked", "unmarked"))
    if blocked:
        not_ready.append(f"{len(blocked)} gold unit(s) wait for a deferred decision (a note marker reading null, "
                         f"e.g. the Hevajra glossary entry; replace the '?'): {', '.join(blocked[:5])}")
    if unmarked:
        not_ready.append(f"{len(unmarked)} gold unit(s) hold a '?' whose note has no known marker (add one of "
                         f"note_markers.yaml, or decide the convention): {', '.join(unmarked[:5])}")
    for name, c in conditions.items():
        if c["passed"] is None:
            not_ready.append(f"{name}: " + ", ".join(f"{k} undefined" for k in ("v1", "v2") if k in c and c[k] is None))
    decision = NOT_READY if not_ready else ADOPT if all(c["passed"] for c in conditions.values()) else KEEP
    return {"decision": decision, "not_ready": not_ready, "conditions": conditions}


def _floor_note(decision: str, v1_kappa: float | None, v2_kappa: float | None, floor: float) -> str | None:
    chosen, kappa = {ADOPT: ("collate.v2", v2_kappa), KEEP: ("collate.v1", v1_kappa)}.get(decision, ("", None))
    if kappa is None or _at_least(kappa, floor):
        return None
    tail = "; tuning collate.v1 makes a new version" if decision == KEEP else ""
    return (f"{chosen} has replicate kappa {kappa:.3f}, below the floor {floor}: tune it on the dev set before "
            f"`prereg freeze` (G2 is at risk{tail})")


def decide(v1: Mapping[str, Any] | None, v2: Mapping[str, Any] | None, v1_kappa: float | None,
           v2_kappa: float | None, floor: float, gaps: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The section-7 decision from the two runs' figures (``RunScores.figures``; None: the run
    has no consensus), their replicate kappas, the floor (``min_replicate_kappa``) and the dev
    regions the gold does not cover yet (``gold_gaps``).

    Returns ``decision`` (``ADOPT``, ``KEEP`` or ``NOT_READY``), ``not_ready`` (the reasons),
    ``conditions`` (each with its values, threshold and ``passed``; None: undefined) under the
    primary reading, ``note`` (the chosen version is below the floor), ``sensitivity`` (the
    same under ``EXCLUDED``), ``full_dev`` (the compared values on every dev window,
    descriptive) and ``rule`` (the constants)."""
    primary = _judge(v1, v2, v1_kappa, v2_kappa, floor, DEFAULT, gaps or {})
    return {**primary, "reading": DEFAULT, "note": _floor_note(primary["decision"], v1_kappa, v2_kappa, floor),
            "sensitivity": {"reading": EXCLUDED, **_judge(v1, v2, v1_kappa, v2_kappa, floor, EXCLUDED, gaps or {})},
            "full_dev": {"relation_kappa_shared": {label: _value(r, DEFAULT, ALL, "relation_kappa_shared")
                                                   for label, r in (("v1", v1), ("v2", v2))},
                         "convention_precision": {"v2": _value(v2, DEFAULT, ALL, "convention_precision")}},
            "rule": {"source": RULE, "tolerance": TOLERANCE, "min_convention_precision": MIN_CONVENTION_PRECISION,
                     "min_replicate_kappa": floor, "tuning_windows": list(TUNING_WINDOWS),
                     "in_sample_units": list(IN_SAMPLE_UNITS), "replicates": REPLICATES, "primary_reading": DEFAULT}}
