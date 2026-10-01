"""Blind gold: the committed format, and scoring any aligner against it (synthesis 5.1, 5.2, 7.3 A).

Committed file ``data/annotations/gold/<witness>/<set>.csv`` (set = dev | test | test_second),
exactly the columns ``GOLD_COLUMNS``; ``windows.csv`` beside it lists the windows
(``windows.TestWindow``). The format is documented for annotators in
``data/annotations/gold/README.md``. Rows come from blind gold sheets (``review.gold_sheets``)
via ``from_verdicts``; status is never stored, it is derived by ``matrix.status``.

Scoring (``score``) compares one ``Alignment`` (Claude's consensus, a DP baseline, the
shuffled placebo, an imported external alignment) with the gold on the gold's own units,
unit by unit, and keeps the per-unit records in ``scores.AlignmentScores`` so that every
metric (``scores``) can be recomputed on any resample of windows (``resample``). Units the aligner left unresolved
stay in every denominator: they count as missed links and as status disagreements (status
"UNALIGNED"), never as silently dropped units.

Gold rows whose decision is ``lacuna`` or ``unresolved`` are not scored (listed in
``AlignmentScores.excluded``).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..collate.verify import FLAG_HINT, FLAG_UNKNOWN_HANDLE
from ..core.ids import orphan_row_id
from ..core.io import read_csv, write_csv
from ..core.types import (REASON_UNASSESSED, REASON_VERIFICATION_FAILED, Alignment, Diagnostic, Link, Relation,
                          Verdict, WitnessOnlyKind)
from ..matrix.status import RELATION_STATUS, deviates, outcome_class
from ..review.verdicts import FLAGS, UNIT_VALUES, decision, format_flags, is_orphan, parse_bool, parse_flags, parse_ids
from .resample import Interval, McNemar, mcnemar, paired_difference, window_bootstrap
from .scores import (METRICS, UNALIGNED, AlignmentScores, UnitScore, WitnessOnlyScore, confidence_reliability,
                     dany_kappa, interval_estimates, invalid_handle_rate, link_f1, link_precision, link_recall,
                     null_precision, null_recall, quote_failure_rate, refusal_by_group, refusal_rate, relation_recall,
                     status_correct, status_kappa, witness_only_kind_agreement, witness_only_recall)
from .windows import (WINDOWS_COLUMNS, TestWindow, dev_region_units, draw_test_windows, region_window,
                      window_units)

# Re-exported so that callers read one module: gold.score, gold.link_f1, gold.window_bootstrap, ...
__all__ = [
    "GOLD_COLUMNS", "SETS", "GoldError", "GoldRow", "GoldSet", "load", "save", "read_windows", "write_windows",
    "from_verdicts", "to_verdicts", "score", "human_kappa",
    "UnitScore", "WitnessOnlyScore", "AlignmentScores", "METRICS", "interval_estimates", "status_correct",
    "confidence_reliability",
    "link_precision", "link_recall", "link_f1", "null_precision", "null_recall", "witness_only_recall",
    "witness_only_kind_agreement", "status_kappa", "dany_kappa", "relation_recall", "quote_failure_rate",
    "invalid_handle_rate", "refusal_rate", "refusal_by_group",
    "TestWindow", "dev_region_units", "draw_test_windows", "region_window", "window_units",
    "window_bootstrap", "paired_difference", "mcnemar", "McNemar", "Interval", "UNALIGNED",
]

GOLD_COLUMNS = ("set", "window_id", "row_type", "unit_id", "fingerprint", "wit_ids", "relation", "polarity_flip",
                "flags", "witness_only_kind", "annotator", "date", "minutes")
SETS = ("dev", "test", "test_second")
ROW_REF, ROW_WITNESS_ONLY = "ref", "witness_only"
WINDOWS_FILE = "windows.csv"
_RELATIONS = frozenset(r.value for r in Relation)
_KINDS = frozenset(k.value for k in WitnessOnlyKind)
_QUOTE_RULE = re.compile(r"\bV4\b")


class GoldError(ValueError):
    """A malformed gold file or row."""


@dataclass(frozen=True)
class GoldRow:
    """One committed gold row: a reference unit (``row_type`` ref) or a witness-only segment."""

    set_name: str
    window_id: str
    row_type: str
    unit_id: str
    fingerprint: str
    wit_ids: tuple[str, ...]
    relation: str = ""                 # ref rows: a Relation value, "lacuna" or "unresolved"
    polarity_flip: bool = False
    flags: frozenset[str] = frozenset()
    witness_only_kind: str = ""        # witness_only rows: a WitnessOnlyKind value
    annotator: str = ""
    date: str = ""
    minutes: float | None = None

    @property
    def scored(self) -> bool:
        """True for rows that enter the metrics (a relation or a witness-only kind)."""
        return self.row_type == ROW_WITNESS_ONLY or self.relation in _RELATIONS

    def link(self, source: str) -> Link:
        if self.row_type == ROW_WITNESS_ONLY:
            return Link(None, self.wit_ids, WitnessOnlyKind(self.witness_only_kind), source=source)
        return Link(self.unit_id, self.wit_ids, Relation(self.relation), polarity_flip=self.polarity_flip,
                    flags=self.flags, source=source)


@dataclass(frozen=True)
class GoldSet:
    """One gold set of one witness, with the windows its rows belong to."""

    name: str
    witness: str
    rows: tuple[GoldRow, ...]
    windows: tuple[TestWindow, ...] = ()
    reference: str = ""

    @property
    def source(self) -> str:
        return f"gold:{self.name}"

    @property
    def alignment(self) -> Alignment:
        """The gold as an ``Alignment`` (source ``gold:<set>``); unscored rows are left out."""
        return Alignment(self.source, self.reference, self.witness,
                         tuple(r.link(self.source) for r in self.rows if r.scored))

    def ref_rows(self) -> tuple[GoldRow, ...]:
        return tuple(r for r in self.rows if r.row_type == ROW_REF)

    def witness_only_rows(self) -> tuple[GoldRow, ...]:
        return tuple(r for r in self.rows if r.row_type == ROW_WITNESS_ONLY)

    def units(self) -> tuple[str, ...]:
        """Scored reference unit ids, in file order."""
        return tuple(r.unit_id for r in self.ref_rows() if r.scored)


# --------------------------------------------------------------------------- read / write
def load(path: Path, reference: str = "") -> GoldSet:
    """Read ``<witness>/<set>.csv`` and the windows its rows use from ``windows.csv`` beside it."""
    path = Path(path)
    name, witness = path.stem, path.parent.name
    if name not in SETS:
        raise GoldError(f"{path}: gold set must be one of {SETS}, got {name!r}")
    header, rows = _read(path)
    if header != set(GOLD_COLUMNS):
        raise GoldError(f"{path}: wrong columns (unexpected {sorted(header - set(GOLD_COLUMNS))}, "
                        f"missing {sorted(set(GOLD_COLUMNS) - header)})")
    parsed = [_row(r, f"{path.name} line {n}", name) for n, r in enumerate(rows, start=2)]
    _check_set(parsed, path)
    windows_path = path.with_name(WINDOWS_FILE)
    if not windows_path.is_file():
        raise GoldError(f"{path}: {WINDOWS_FILE} is missing beside it")
    known = {w.window_id: w for w in read_windows(windows_path)}
    used = list(dict.fromkeys(r.window_id for r in parsed))
    missing = [w for w in used if w not in known]
    if missing:
        raise GoldError(f"{path}: window(s) {missing} are not in {WINDOWS_FILE}")
    return GoldSet(name, witness, tuple(parsed), tuple(known[w] for w in used), reference)


def _read(path: Path) -> tuple[set[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(r) for r in reader]
        return {c.strip() for c in reader.fieldnames or ()}, rows


def _row(raw: Mapping[str, str], where: str, set_name: str) -> GoldRow:
    v = {k: (raw.get(k) or "").strip() for k in GOLD_COLUMNS}
    if v["set"] != set_name:
        raise GoldError(f"{where}: set {v['set']!r} differs from the file name {set_name!r}")
    try:
        flip = parse_bool(v["polarity_flip"])
        minutes = float(v["minutes"]) if v["minutes"] else None
    except ValueError as exc:
        raise GoldError(f"{where}: {exc}") from None
    row = GoldRow(set_name=set_name, window_id=v["window_id"], row_type=v["row_type"], unit_id=v["unit_id"],
                  fingerprint=v["fingerprint"], wit_ids=parse_ids(v["wit_ids"]), relation=v["relation"],
                  polarity_flip=flip, flags=parse_flags(v["flags"]), witness_only_kind=v["witness_only_kind"],
                  annotator=v["annotator"], date=v["date"], minutes=minutes)
    problem = _problem(row)
    if problem:
        raise GoldError(f"{where}: {row.unit_id or '(no unit_id)'}: {problem}")
    return row


def _problem(row: GoldRow) -> str | None:
    if not row.window_id or not row.unit_id:
        return "window_id and unit_id are required"
    unknown = sorted(row.flags - set(FLAGS))
    if unknown:
        return f"unknown flag(s) {unknown}"
    if row.row_type == ROW_WITNESS_ONLY:
        if row.relation or row.witness_only_kind not in _KINDS:
            return f"a witness_only row needs witness_only_kind in {sorted(_KINDS)} and no relation"
        if len(row.wit_ids) != 1 or row.unit_id != orphan_row_id(row.wit_ids[0]):
            return "a witness_only row names one segment, and unit_id is '+<that segment id>'"
        return None
    if row.row_type != ROW_REF:
        return f"row_type must be {ROW_REF} or {ROW_WITNESS_ONLY}, got {row.row_type!r}"
    if is_orphan(row.unit_id) or row.witness_only_kind:
        return "a ref row names a reference unit and has no witness_only_kind"
    if row.relation not in UNIT_VALUES:
        return f"relation {row.relation!r} is not one of {', '.join(UNIT_VALUES)}"
    if row.relation == Relation.NO_COUNTERPART.value and row.wit_ids:
        return "no_counterpart names no witness segment"
    if row.relation in _RELATIONS - {Relation.NO_COUNTERPART.value} and not row.wit_ids:
        return f"relation {row.relation} needs witness segment ids"
    return None


def _check_set(rows: Sequence[GoldRow], path: Path) -> None:
    repeated = sorted(u for u, n in Counter(r.unit_id for r in rows).items() if n > 1)
    if repeated:
        raise GoldError(f"{path}: unit(s) given more than once: {repeated[:5]}")
    linked = {w for r in rows if r.row_type == ROW_REF for w in r.wit_ids}
    both = sorted(w for r in rows if r.row_type == ROW_WITNESS_ONLY for w in r.wit_ids if w in linked)
    if both:
        raise GoldError(f"{path}: segment(s) both linked and witness-only: {both[:5]}")


def save(gold: GoldSet, path: Path) -> None:
    """Write the set file (UTF-8, no BOM); ``windows.csv`` is written by ``write_windows``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, [_to_row(r) for r in gold.rows], GOLD_COLUMNS)


def _to_row(r: GoldRow) -> dict[str, object]:
    return {"set": r.set_name, "window_id": r.window_id, "row_type": r.row_type, "unit_id": r.unit_id,
            "fingerprint": r.fingerprint, "wit_ids": " ".join(r.wit_ids), "relation": r.relation,
            "polarity_flip": "true" if r.polarity_flip else "false", "flags": format_flags(r.flags),
            "witness_only_kind": r.witness_only_kind, "annotator": r.annotator, "date": r.date,
            "minutes": "" if r.minutes is None else r.minutes}


def read_windows(path: Path) -> list[TestWindow]:
    rows = read_csv(path)
    if rows and set(rows[0]) != set(WINDOWS_COLUMNS):
        raise GoldError(f"{path}: columns must be exactly {WINDOWS_COLUMNS}")
    windows = [TestWindow.from_row(r) for r in rows]
    repeated = sorted(w for w, n in Counter(w.window_id for w in windows).items() if n > 1)
    if repeated:
        raise GoldError(f"{path}: window id(s) listed more than once: {repeated}")
    return windows


def write_windows(windows: Iterable[TestWindow], path: Path) -> None:
    write_csv(Path(path), [w.to_row() for w in windows], WINDOWS_COLUMNS)


# --------------------------------------------------------------------------- verdict conversion
def from_verdicts(verdicts: Iterable[Verdict], set_name: str, witness: str, reference: str = "") -> GoldSet:
    """Gold rows from imported blind gold verdicts (batch ``<set>_<window>``); windows are
    attached by the caller (``dataclasses.replace(gold, windows=...)``)."""
    prefix = f"{set_name}_"
    rows = []
    for v in verdicts:
        chosen = decision(v)
        if v.task != "gold" or not v.batch_id.startswith(prefix) or chosen is None:
            raise GoldError(f"{v.item_id}: not a decided gold verdict of batch {prefix}<window>")
        value, wit_ids = chosen
        orphan = is_orphan(v.unit_id)
        rows.append(GoldRow(
            set_name=set_name, window_id=v.batch_id[len(prefix):], row_type=ROW_WITNESS_ONLY if orphan else ROW_REF,
            unit_id=v.unit_id, fingerprint=v.fingerprint, wit_ids=wit_ids, relation="" if orphan else value,
            polarity_flip=v.polarity_flip, flags=v.flags, witness_only_kind=value if orphan else "",
            annotator=v.annotator, date=v.blind_date, minutes=v.minutes))
    for n, row in enumerate(rows):
        problem = _problem(row)
        if problem:
            raise GoldError(f"gold verdict {n + 1} ({row.unit_id}): {problem}")
    _check_set(rows, Path(f"{witness}/{set_name}"))
    return GoldSet(set_name, witness, tuple(rows), (), reference)


def to_verdicts(gold: GoldSet) -> list[Verdict]:
    """The gold as ``task="gold"`` verdicts, the form ``matrix.build.build_cells`` takes."""
    return [Verdict(batch_id=f"{gold.name}_{r.window_id}", item_id=f"gold:{r.unit_id}", task="gold",
                    unit_id=r.unit_id, fingerprint=r.fingerprint, blind_relation=r.relation or r.witness_only_kind,
                    blind_wit_ids=r.wit_ids, polarity_flip=r.polarity_flip, flags=r.flags, annotator=r.annotator,
                    blind_date=r.date, minutes=r.minutes)
            for r in gold.rows]


def score(alignment: Alignment, gold: GoldSet, units: Iterable[str] | None = None, *,
          ref_kinds: Mapping[str, str], unresolved: Mapping[str, str] | None = None,
          diagnostics: Sequence[Diagnostic] = (), topic_groups: Mapping[str, str] | None = None) -> AlignmentScores:
    """Score ``alignment`` against ``gold`` on the gold's scored units (or the subset ``units``).

    ``ref_kinds``     unit id -> reference segment kind (required: a transliterated mantra is
                      no deviation, a transliterated prose unit is)
    ``unresolved``    unit id -> UNALIGNED reason (``Consensus.reasons``), for refusal rates;
                      an unresolved unit without a reason is "unassessed"
    ``diagnostics``   verification diagnostics of the run; a V4 failure on a unit marks a
                      quote failure
    ``topic_groups``  unit id -> topic group, for refusal rates per group
    Witness-only rows are kept for the windows that keep at least one unit.
    """
    hinted = sum(1 for link in alignment.links if FLAG_HINT in link.flags)
    if hinted:
        raise ValueError(f"{alignment.source}: substituted-model hints are never scored ({hinted} found)")
    rows = {r.unit_id: r for r in gold.ref_rows() if r.scored}
    keep = list(rows) if units is None else list(dict.fromkeys(units))
    unknown = [u for u in keep if u not in rows]
    if unknown:
        raise ValueError(f"unit(s) not scored in gold:{gold.name}: {unknown[:5]}")
    missing_kind = [u for u in keep if u not in ref_kinds]
    if missing_kind:
        raise ValueError(f"ref_kinds lacks {len(missing_kind)} gold unit(s), e.g. {missing_kind[:3]}")
    reasons, groups = unresolved or {}, topic_groups or {}
    quote_failed = {uid for d in diagnostics if d.kind == REASON_VERIFICATION_FAILED and _QUOTE_RULE.search(d.detail)
                    for uid in d.ref_ids}
    found = alignment.by_ref()
    scored = tuple(_unit(rows[u], found.get(u), ref_kinds[u], reasons, u in quote_failed, groups.get(u, ""))
                   for u in keep)
    windows = {s.window_id for s in scored}
    linked = {w for link in alignment.links if link.ref_id is not None for w in link.wit_ids}
    predicted = {w: str(link.relation) for link in alignment.witness_only() for w in link.wit_ids if w not in linked}
    only = tuple(WitnessOnlyScore(r.unit_id, r.window_id, r.wit_ids[0], r.witness_only_kind, predicted.get(r.wit_ids[0]))
                 for r in gold.witness_only_rows() if r.window_id in windows)
    excluded = tuple(r.unit_id for r in gold.ref_rows() if not r.scored)
    return AlignmentScores(alignment.source, gold.name, scored, only, excluded)


def _unit(row: GoldRow, link: Link | None, kind: str, reasons: Mapping[str, str], quote_failure: bool,
          group: str) -> UnitScore:
    gold_rel = Relation(row.relation)
    gold_dev = deviates(outcome_class(gold_rel, row.polarity_flip, kind))
    common = dict(unit_id=row.unit_id, window_id=row.window_id, gold_relation=gold_rel.value,
                  gold_status=RELATION_STATUS[gold_rel].value, gold_dev=gold_dev, gold_wit=frozenset(row.wit_ids),
                  quote_failure=quote_failure, topic_group=group)
    if link is None:
        return UnitScore(**common, pred_relation=None, pred_status=UNALIGNED, pred_dev=None,  # type: ignore[arg-type]
                         pred_wit=frozenset(), reason=reasons.get(row.unit_id, REASON_UNASSESSED))
    rel = Relation(link.relation)
    return UnitScore(**common, pred_relation=rel.value, pred_status=RELATION_STATUS[rel].value,  # type: ignore[arg-type]
                     pred_dev=deviates(outcome_class(rel, link.polarity_flip, kind)), pred_wit=frozenset(link.wit_ids),
                     invalid_handle=FLAG_UNKNOWN_HANDLE in link.flags, pred_confidence=link.confidence)


def human_kappa(primary: GoldSet, second: GoldSet, ref_kinds: Mapping[str, str]) -> float | None:
    """Status kappa of the second annotator against the primary one, on the units both scored
    (the G1 ceiling). Units the second annotator left undecided count as disagreements."""
    scored = set(primary.units())
    both = [r.unit_id for r in second.ref_rows() if r.unit_id in scored]
    return status_kappa(score(second.alignment, primary, both, ref_kinds=ref_kinds))
