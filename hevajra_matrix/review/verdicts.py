"""Committed human verdicts: ``data/annotations/verdicts/<witness>/<batch>.csv``.

A verdict is a human decision about one matrix row, made on a review sheet (tasks verify,
audit, resolve; see ``review.sheets``) or on a blind gold sheet (task gold). Gold verdicts
exist only in memory: ``review import --task gold`` stores them as a gold set under
``data/annotations/gold/<witness>/<set>.csv`` (``evaluation.gold``), and ``read_file``
refuses a task-gold row in a committed verdict file. The committed
files hold ids, decisions and short quotes only (licence rule, synthesis 7.2); the text the
annotator read lives in the sheets under ``runs/``, which are never committed. The format
is documented for annotators in ``data/annotations/verdicts/README.md``.

Which decision the matrix applies (``decision``)
    gold             the final columns when filled, else the blind ones (gold is blind by
                     construction; its blind decision is its final one)
    verify, audit    the final columns only, i.e. after the reveal step (critique A1:
                     final is primary; blind feeds the automation-bias sensitivity)
    resolve          no reveal step: ``review.sheets`` copies the blind decision into the
                     final columns on import, so the rule above applies unchanged

Decision values
    reference unit   a ``Relation`` value, ``lacuna`` (witness physically damaged/lost) or
                     ``unresolved`` (the annotator cannot decide; the cell stays UNALIGNED)
    orphan row       a ``WitnessOnlyKind`` value, or ``has_counterpart`` (the claim is
                     rejected: the segment does translate some reference unit; record that
                     link with a verdict on the unit)

``VerdictRecord`` extends the frozen ``core.types.Verdict`` with the three committed columns
the core type does not carry (``quote_ref``, ``quote_zh``, ``minutes``). It is a ``Verdict``,
so every consumer of verdicts accepts it.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..core.ids import id_kind, orphan_source
from ..core.io import write_csv
from ..core.types import Relation, Segment, Verdict, WitnessOnlyKind

COLUMNS: tuple[str, ...] = (
    "batch_id", "item_id", "task", "stratum", "inclusion_prob", "unit_id", "fingerprint",
    "instrument_digest", "machine_relation", "machine_status", "machine_polarity_flip", "blind_relation",
    "blind_wit_ids",
    "final_relation", "final_wit_ids", "polarity_flip", "flags", "quote_ref", "quote_zh",
    "annotator", "blind_date", "final_date", "minutes", "note",
)
LEGACY_OPTIONAL = frozenset({"machine_polarity_flip"})   # absent from files written before 2026-10-01
TASKS: tuple[str, ...] = ("gold", "verify", "audit", "resolve")
REVIEW_TASKS: tuple[str, ...] = ("verify", "audit", "resolve")

LACUNA = "lacuna"
UNRESOLVED = "unresolved"
HAS_COUNTERPART = "has_counterpart"
UNIT_VALUES: tuple[str, ...] = tuple(r.value for r in Relation) + (LACUNA, UNRESOLVED)
ORPHAN_VALUES: tuple[str, ...] = tuple(k.value for k in WitnessOnlyKind) + (HAS_COUNTERPART,)

# Human flags (synthesis 7.3 A); free-form flags are refused so typos cannot hide.
FLAGS: tuple[str, ...] = ("scope_list", "uniform_across_list", "instruction_as_mantra", "reordered", "unsure")
ID_SEPARATOR = " "
FLAG_SEPARATOR = ";"
MAX_COMMITTED_QUOTE = 60           # hard licence cap, whatever the config says
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TRUE, _FALSE = ("true", "1", "yes", "y"), ("false", "0", "no", "n", "")


class VerdictError(ValueError):
    """A malformed verdict file or row."""


@dataclass(frozen=True)
class VerdictRecord(Verdict):
    """A committed verdict row: ``Verdict`` plus the short quotes and the minutes spent."""

    quote_ref: str = ""
    quote_zh: str = ""
    minutes: float | None = None


# --------------------------------------------------------------------------- decisions
def is_orphan(unit_id: str) -> bool:
    """True for an orphan (witness-only) row id such as ``+T0892:0601c01.2``."""
    return unit_id.startswith("+")


def decision(v: Verdict) -> tuple[str, tuple[str, ...]] | None:
    """(decision value, witness segment ids) the matrix applies, or None if not final yet."""
    if v.task == "gold" and not v.final_relation:
        return (v.blind_relation, v.blind_wit_ids) if v.blind_relation else None
    return (v.final_relation, v.final_wit_ids) if v.final_relation else None


def decision_date(v: Verdict) -> str:
    return v.final_date or v.blind_date


# --------------------------------------------------------------------------- parse / format
def parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise VerdictError(f"not a boolean: {value!r} (use true/false)")


def parse_ids(value: str) -> tuple[str, ...]:
    return tuple(x for x in value.replace(",", " ").split() if x)


def parse_flags(value: str) -> frozenset[str]:
    return frozenset(x.strip() for x in value.split(FLAG_SEPARATOR) if x.strip())


def format_flags(flags: Iterable[str]) -> str:
    return FLAG_SEPARATOR.join(sorted(flags))


def _float(value: str, column: str) -> float | None:
    if not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        raise VerdictError(f"{column}: not a number: {value!r}") from None


def from_row(row: Mapping[str, str]) -> VerdictRecord:
    """One committed CSV row -> ``VerdictRecord`` (values are stripped; lists parsed)."""
    v = {k: (row.get(k) or "").strip() for k in COLUMNS}
    return VerdictRecord(
        batch_id=v["batch_id"], item_id=v["item_id"], task=v["task"], unit_id=v["unit_id"],
        fingerprint=v["fingerprint"], stratum=v["stratum"],
        inclusion_prob=_float(v["inclusion_prob"], "inclusion_prob"),
        instrument_digest=v["instrument_digest"], machine_relation=v["machine_relation"],
        machine_status=v["machine_status"], machine_polarity_flip=parse_bool(v["machine_polarity_flip"]),
        blind_relation=v["blind_relation"],
        blind_wit_ids=parse_ids(v["blind_wit_ids"]), final_relation=v["final_relation"],
        final_wit_ids=parse_ids(v["final_wit_ids"]), polarity_flip=parse_bool(v["polarity_flip"]),
        flags=parse_flags(v["flags"]), annotator=v["annotator"], blind_date=v["blind_date"],
        final_date=v["final_date"], note=row.get("note", "") or "", quote_ref=v["quote_ref"],
        quote_zh=v["quote_zh"], minutes=_float(v["minutes"], "minutes"),
    )


def to_row(v: Verdict) -> dict[str, str]:
    """``Verdict`` (or ``VerdictRecord``) -> committed CSV row."""
    minutes = getattr(v, "minutes", None)
    return {
        "batch_id": v.batch_id, "item_id": v.item_id, "task": v.task, "stratum": v.stratum,
        "inclusion_prob": "" if v.inclusion_prob is None else f"{v.inclusion_prob:.6g}",
        "unit_id": v.unit_id, "fingerprint": v.fingerprint, "instrument_digest": v.instrument_digest,
        "machine_relation": v.machine_relation, "machine_status": v.machine_status,
        "machine_polarity_flip": "true" if v.machine_polarity_flip else "false",
        "blind_relation": v.blind_relation, "blind_wit_ids": ID_SEPARATOR.join(v.blind_wit_ids),
        "final_relation": v.final_relation, "final_wit_ids": ID_SEPARATOR.join(v.final_wit_ids),
        "polarity_flip": "true" if v.polarity_flip else "false", "flags": format_flags(v.flags),
        "quote_ref": getattr(v, "quote_ref", ""), "quote_zh": getattr(v, "quote_zh", ""),
        "annotator": v.annotator, "blind_date": v.blind_date, "final_date": v.final_date,
        "minutes": "" if minutes is None else f"{minutes:g}", "note": v.note,
    }


# --------------------------------------------------------------------------- files
def read_file(path: Path) -> list[VerdictRecord]:
    """Read one committed verdict CSV; the header must be exactly ``COLUMNS`` (any order;
    a file written before ``machine_polarity_flip`` was committed may lack that column,
    which then reads as false)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or ())
        rows = list(reader)
    extra = sorted(set(header) - set(COLUMNS))
    missing = sorted(set(COLUMNS) - set(header) - LEGACY_OPTIONAL)
    if extra or missing or len(header) != len(set(header)):
        raise VerdictError(f"{path}: wrong columns (unexpected {extra}, missing {missing}); "
                           f"a committed verdict file holds no text columns")
    out = []
    for line, row in enumerate(rows, start=2):
        if None in row:
            raise VerdictError(f"{path}: line {line}: more values than columns")
        try:
            record = from_row(row)
        except VerdictError as exc:
            raise VerdictError(f"{path}: line {line}: {exc}") from None
        if record.task not in REVIEW_TASKS:
            # Gold lives in data/annotations/gold/; here it would lose gold precedence.
            raise VerdictError(f"{path}: line {line}: task {record.task!r} does not belong in a verdict "
                               f"file (one of {', '.join(REVIEW_TASKS)}); gold sets are kept under "
                               f"data/annotations/gold/<witness>/<set>.csv")
        out.append(record)
    return out


PLAN_FILE_PREFIX = "plan_"         # review plans committed beside the verdicts (pipeline.human_data)


def load(directory: Path) -> list[VerdictRecord]:
    """All verdicts under ``directory`` (``<witness>/<batch>.csv``), files in path order;
    the review plans committed beside them (``plan_<batch>.csv``) are skipped."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return [v for path in sorted(root.rglob("*.csv")) if not path.name.startswith(PLAN_FILE_PREFIX)
            for v in read_file(path)]


def save(verdicts: Iterable[Verdict], path: Path) -> None:
    """Write one committed verdict CSV (UTF-8 without BOM, rows in the given order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, [to_row(v) for v in verdicts], COLUMNS)


# --------------------------------------------------------------------------- validation
def validate(verdicts: Sequence[Verdict], segment_index: Mapping[str, Segment],
             max_quote_chars: int = MAX_COMMITTED_QUOTE) -> list[str]:
    """Every structural problem, one message each (empty list: valid).

    ``segment_index`` maps segment id -> segment for the reference AND the witness text.
    A fingerprint that no longer matches is NOT an error here: ``matrix.build`` lists such
    verdicts as stale and never applies them.
    """
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for v in verdicts:
        where = f"{v.batch_id}/{v.item_id or '?'} ({v.unit_id or '?'})"
        problems.extend(f"{where}: {p}" for p in _row_problems(v, segment_index, max_quote_chars))
        key = (v.batch_id, v.item_id)
        if key in seen:
            problems.append(f"{where}: duplicate (batch_id, item_id)")
        seen.add(key)
    return problems


def _row_problems(v: Verdict, index: Mapping[str, Segment], max_quote: int) -> list[str]:
    out = [f"{name} is empty" for name in ("batch_id", "item_id", "unit_id", "fingerprint")
           if not getattr(v, name)]
    if v.task not in TASKS:
        out.append(f"task {v.task!r} is not one of {', '.join(TASKS)}")
    if v.unit_id:
        out += _unit_problems(v.unit_id, index)
    allowed = ORPHAN_VALUES if is_orphan(v.unit_id) else UNIT_VALUES
    for column, value, ids in (("blind", v.blind_relation, v.blind_wit_ids),
                               ("final", v.final_relation, v.final_wit_ids)):
        if value and value not in allowed:
            out.append(f"{column}_relation {value!r} is not one of {', '.join(allowed)}")
        if value == Relation.NO_COUNTERPART.value and ids:
            out.append(f"{column}: no_counterpart must have no witness ids")
        if value in {r.value for r in Relation} - {Relation.NO_COUNTERPART.value} and not ids:
            out.append(f"{column}: relation {value} needs witness ids")
        out += [f"{column}_wit_ids: unknown segment {i}" for i in ids if i not in index]
    if not (v.blind_relation or v.final_relation):
        out.append("no decision in blind_relation or final_relation")
    decided = decision(v)
    if decided and decided[0] == Relation.REVERSAL.value and not v.polarity_flip:
        out.append("reversal requires polarity_flip = true")
    out += [f"unknown flag {f!r}; allowed: {', '.join(FLAGS)}" for f in sorted(v.flags - set(FLAGS))]
    if v.inclusion_prob is not None and not 0 < v.inclusion_prob <= 1:
        out.append(f"inclusion_prob {v.inclusion_prob} is not in (0, 1]")
    if v.task in ("verify", "audit") and v.inclusion_prob is None:
        out.append("a verify/audit verdict needs its inclusion_prob")
    for name in ("blind_date", "final_date"):
        value = getattr(v, name)
        if value and not _ISO_DATE.fullmatch(value):
            out.append(f"{name} {value!r} is not YYYY-MM-DD")
    for name in ("quote_ref", "quote_zh"):
        if len(getattr(v, name, "")) > max_quote:
            out.append(f"{name} longer than {max_quote} characters (licence rule)")
    return out


def _unit_problems(unit_id: str, index: Mapping[str, Segment]) -> list[str]:
    try:
        kind = id_kind(unit_id)
    except ValueError as exc:
        return [str(exc)]
    target = orphan_source(unit_id) if kind == "orphan" else unit_id
    return [] if target in index else [f"unknown unit {unit_id}"]
