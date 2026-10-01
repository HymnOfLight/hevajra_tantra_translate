"""Load researcher-supplied Sanskrit data: reference units and per-manuscript readings.

The Sanskrit critical editions are under copyright and are not shipped with the
repository; the formats are described in ``data/reference/README.md``.

Reference units (``load_reference``), one TSV per edition::

    # comment lines start with '#'
    I.1.p01<TAB>evam maya srutam ...      prose proposition (unit id I.1.p01)
    I.1.1<TAB>vajrasattvo bhavet ...      verse
    I.2.5<TAB><TAB>LACUNA                 the edition reports a physical gap in its manuscripts
    I.2.6<TAB><TAB>ABSENT                 the edition has no such verse (another edition has it)

Unit ids follow Snellgrove's numbering (``core.ids.parse_snellgrove_id``) and are both the
segment id and its coordinate. The text is kept as written (NFC, whitespace collapsed).

Per-manuscript readings (``load_readings``), one TSV per reference chapter named after it
(``readings/I.7.tsv``), with a header row::

    unit_id  ms  status  reading  source  note

Unit ids are reference unit ids: Snellgrove ids, or Derge segment ids (``D417:8a.6.1``) while
the Derge is the provisional reference; a Derge id is not checked against the file's
chapter here (the stats stage reports ids that are not reference units).

``status`` is present | absent | variant | illegible | not_collated. A manuscript that
was not collated at a unit is ``not_collated``, never ``absent``: absence is a claim about
the manuscript, not about the state of the collation. Editions are not independent
witnesses and never appear here as manuscripts.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..core.ids import REF_CHAPTERS, SnellgroveId, parse_segment_id, parse_snellgrove_id
from ..core.textnorm import fingerprint
from ..core.types import Segment

LANG = "sa"
FLAG_LACUNA = "LACUNA"
FLAG_ABSENT = "ABSENT"
UNIT_FLAGS = frozenset({FLAG_LACUNA, FLAG_ABSENT})
READING_STATUSES = frozenset({"present", "absent", "variant", "illegible", "not_collated"})
READING_COLUMNS = ("unit_id", "ms", "status", "reading", "source", "note")


@dataclass(frozen=True)
class Reading:
    """What one manuscript has at one reference unit."""

    unit_id: str
    ms: str
    status: str
    reading: str = ""
    source: str = ""
    note: str = ""


@dataclass(frozen=True)
class ReadingTable:
    """All readings, grouped by reference unit (units in file order, then manuscripts)."""

    by_unit: Mapping[str, tuple[Reading, ...]]

    def for_unit(self, unit_id: str) -> tuple[Reading, ...]:
        return self.by_unit.get(unit_id, ())

    def manuscripts(self) -> frozenset[str]:
        return frozenset(r.ms for rows in self.by_unit.values() for r in rows)


def load_reference(path: Path, witness: str) -> list[Segment]:
    """Reference units of one edition, in file order."""
    path = Path(path)
    segments: list[Segment] = []
    seen: set[str] = set()
    for n, fields in _rows(path):
        where = f"{path}:{n}"
        if len(fields) > 3:
            raise ValueError(f"{where}: expected 'unit_id<TAB>text[<TAB>flag]', got {len(fields)} columns")
        uid = fields[0].strip()
        unit = _unit(uid, where)
        text = " ".join(unicodedata.normalize("NFC", fields[1] if len(fields) > 1 else "").split())
        flag = fields[2].strip() if len(fields) > 2 else ""
        if flag and flag not in UNIT_FLAGS:
            raise ValueError(f"{where}: unknown flag {flag!r}; use one of {sorted(UNIT_FLAGS)}")
        if not text and not flag:
            raise ValueError(f"{where}: unit {uid} has no text; mark it LACUNA or ABSENT")
        if uid in seen:
            raise ValueError(f"{where}: duplicate unit {uid}")
        seen.add(uid)
        segments.append(Segment(
            id=uid, witness=witness, lang=LANG, text=text, start=uid, end=uid,
            kind="prose" if unit.prose else "verse", local_chapter=unit.chapter_key,
            chapter=unit.chapter_key, fingerprint=fingerprint(text, LANG),
            extra={"flag": flag} if flag else {},
        ))
    return segments


def load_readings(directory: Path) -> ReadingTable:
    """Readings from every ``<chapter>.tsv`` in ``directory`` (which must exist)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"readings directory {directory} does not exist")
    by_unit: dict[str, list[Reading]] = {}
    seen: set[tuple[str, str]] = set()
    for path in sorted(directory.glob("*.tsv")):
        if path.stem not in REF_CHAPTERS:
            raise ValueError(f"{path}: file name must be a reference chapter such as 'I.7.tsv'")
        rows = _rows(path)
        if not rows or tuple(f.strip() for f in rows[0][1]) != READING_COLUMNS:
            raise ValueError(f"{path}: the first row must be the header {'<TAB>'.join(READING_COLUMNS)}")
        for n, fields in rows[1:]:
            reading = _reading(fields, path.stem, f"{path}:{n}")
            if (reading.unit_id, reading.ms) in seen:
                raise ValueError(f"{path}:{n}: second reading of {reading.ms} at {reading.unit_id}")
            seen.add((reading.unit_id, reading.ms))
            by_unit.setdefault(reading.unit_id, []).append(reading)
    return ReadingTable(MappingProxyType({u: tuple(rs) for u, rs in by_unit.items()}))


def _reading(fields: list[str], chapter: str, where: str) -> Reading:
    if not 3 <= len(fields) <= len(READING_COLUMNS):
        raise ValueError(f"{where}: expected 3 to {len(READING_COLUMNS)} columns, got {len(fields)}")
    values = dict(zip(READING_COLUMNS, [f.strip() for f in fields] + [""] * len(READING_COLUMNS)))
    if ":" in values["unit_id"]:          # a Derge segment id while the Derge is the provisional reference
        try:
            parse_segment_id(values["unit_id"])
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    elif (unit := _unit(values["unit_id"], where)).chapter_key != chapter:
        raise ValueError(f"{where}: unit {values['unit_id']} belongs to {unit.chapter_key}, not {chapter}")
    if not values["ms"]:
        raise ValueError(f"{where}: empty manuscript id")
    if values["status"] not in READING_STATUSES:
        raise ValueError(f"{where}: status {values['status']!r} not in {sorted(READING_STATUSES)}")
    if values["status"] == "variant" and not values["reading"]:
        raise ValueError(f"{where}: a variant needs the reading")
    values["reading"] = unicodedata.normalize("NFC", values["reading"])
    return Reading(**values)


def _unit(uid: str, where: str) -> SnellgroveId:
    try:
        return parse_snellgrove_id(uid)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _rows(path: Path) -> list[tuple[int, list[str]]]:
    """(line number, tab-separated fields) of the non-blank, non-comment lines."""
    out = []
    for n, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if raw.strip() and not raw.lstrip().startswith("#"):
            out.append((n, raw.rstrip("\r\n").split("\t")))
    return out
