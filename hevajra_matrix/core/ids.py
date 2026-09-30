"""Stable identifiers for witness segments, reference units and matrix rows.

Every key the pipeline stores (alignments, verdicts, gold, sentinels) is one of these ids,
so they must survive re-ingestion. They are therefore derived from the witness's own
coordinates and never from a running counter over the whole text.

Id kinds
    segment     ``T0892:0592a29.1``   first content segment starting on Taisho line 0592a29
                ``D417:8a.4.3``       third content segment starting on Derge folio 8a, line 4
    note        ``T0892:0592a27.n1``  first translator note starting on line 0592a27
    orphan row  ``+T0892:0601c01.2``  matrix row for witness-only material, named after the
                                      first witness segment of the block (unique by construction)
    snellgrove  ``I.5.12``, ``I.5.12a``, ``I.1.p03``
                                      Sanskrit reference units: verse 12 (half-verse a) of Part I
                                      chapter 5, and the third prose proposition of I.1

The *line* is the witness line on which a segment starts; the *ordinal* counts segments
of the same kind (content or note) that start on that line, in document order. Content
and notes are numbered separately, so adding or removing a note never renames a content
segment, and any change on one line renames nothing on another line (see ``assign_ids``).

The prefix names the witness text (``T0892``, ``D417``, ``D418``); mapping a witness to
its prefix is the ingester's job. Ids have exactly one spelling: every parser rejects a
non-canonical form (e.g. ``I.1.p3`` for ``I.1.p03``) instead of silently creating a
second key for the same unit.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

IdKind = Literal["segment", "note", "orphan", "snellgrove"]

PART_CHAPTERS: Mapping[str, int] = MappingProxyType({"I": 11, "II": 12})
REF_CHAPTERS: tuple[str, ...] = tuple(
    f"{part}.{n}" for part, count in PART_CHAPTERS.items() for n in range(1, count + 1)
)
_PART_RANK = {part: rank for rank, part in enumerate(PART_CHAPTERS)}

ORPHAN_MARK = "+"
NOTE_MARK = "n"

_PREFIX_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# Alphanumeric groups joined by dots: "0592a29" (Taisho), "8a.4" (Derge folio.line).
_LINE_RE = re.compile(r"[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*")
_SEGMENT_RE = re.compile(
    r"(?P<prefix>[A-Za-z][A-Za-z0-9_]*):(?P<line>[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)"
    r"\.(?P<note>n?)(?P<ordinal>[1-9][0-9]*)"
)
_SNELLGROVE_RE = re.compile(
    r"(?P<part>[A-Z]+)\.(?P<chapter>[1-9][0-9]?)\.(?:(?P<verse>[1-9][0-9]{0,2})(?P<half>[ab]?)|p(?P<prose>[0-9]{2,3}))"
)
_NATURAL_RE = re.compile(r"[0-9]+|[^0-9]+")


@dataclass(frozen=True)
class SegmentRef:
    """Parsed coordinate id of a witness segment or note."""

    prefix: str
    line: str
    ordinal: int
    is_note: bool = False

    def __str__(self) -> str:
        return f"{self.prefix}:{self.line}.{NOTE_MARK if self.is_note else ''}{self.ordinal}"


@dataclass(frozen=True)
class SnellgroveId:
    """Parsed Sanskrit reference unit id (Snellgrove's chapter and verse numbering)."""

    part: str
    chapter: int
    index: int
    prose: bool = False
    half: str = ""          # "a" or "b" for a half-verse; always "" for prose

    @property
    def chapter_key(self) -> str:
        return f"{self.part}.{self.chapter}"

    def __str__(self) -> str:
        if self.prose:
            return f"{self.chapter_key}.p{self.index:02d}"
        return f"{self.chapter_key}.{self.index}{self.half}"


# --------------------------------------------------------------------------- construction
def segment_id(prefix: str, line: str, ordinal: int) -> str:
    """Id of the ``ordinal``-th content segment starting on ``line``, e.g. ``T0892:0592a29.1``."""
    return str(_checked_ref(prefix, line, ordinal, is_note=False))


def note_id(prefix: str, line: str, ordinal: int) -> str:
    """Id of the ``ordinal``-th translator note starting on ``line``, e.g. ``T0892:0592a27.n1``."""
    return str(_checked_ref(prefix, line, ordinal, is_note=True))


def orphan_row_id(first_segment_id: str) -> str:
    """Matrix row id for witness-only material whose first segment is ``first_segment_id``.

    Two witness-only blocks never share a first segment, so two blocks on the same line
    get two rows (the v0.2 scheme keyed rows by line and silently merged them).
    """
    parse_segment_id(first_segment_id)
    return ORPHAN_MARK + first_segment_id


def assign_ids(prefix: str, entries: Iterable[tuple[str, bool]]) -> list[str]:
    """Ids for one witness text's segments, given in document order as ``(line, is_note)``.

    ``line`` is the coordinate of the line on which the segment starts. Content segments
    and notes are counted separately per line, which gives the two stability guarantees
    stated in the module docstring. Ids are unique for any input.
    """
    seen: Counter[tuple[str, bool]] = Counter()
    ids = []
    for line, is_note in entries:
        seen[(line, is_note)] += 1
        ids.append(str(_checked_ref(prefix, line, seen[(line, is_note)], is_note)))
    return ids


def _checked_ref(prefix: str, line: str, ordinal: int, is_note: bool) -> SegmentRef:
    if not _PREFIX_RE.fullmatch(prefix):
        raise ValueError(f"bad id prefix {prefix!r}: expected a letter then letters, digits or '_'")
    if not _LINE_RE.fullmatch(line):
        raise ValueError(f"bad line coordinate {line!r}: expected alphanumerics joined by dots")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ValueError(f"ordinal must be a positive int, got {ordinal!r}")
    return SegmentRef(prefix, line, ordinal, is_note)


# --------------------------------------------------------------------------- parsing
def id_kind(unit_id: str) -> IdKind:
    """Classify an id; raises ``ValueError`` if it is not a valid id of any kind."""
    if unit_id.startswith(ORPHAN_MARK):
        parse_segment_id(unit_id[len(ORPHAN_MARK):])
        return "orphan"
    if ":" in unit_id:
        return "note" if parse_segment_id(unit_id).is_note else "segment"
    parse_snellgrove_id(unit_id)
    return "snellgrove"


def parse_segment_id(sid: str) -> SegmentRef:
    """Parse a content or note id (not an orphan row id)."""
    m = _SEGMENT_RE.fullmatch(sid)
    if m is None:
        raise ValueError(f"bad segment id {sid!r}; expected e.g. 'T0892:0592a29.1' or 'T0892:0592a27.n1'")
    return SegmentRef(m["prefix"], m["line"], int(m["ordinal"]), bool(m["note"]))


def orphan_source(row_id: str) -> str:
    """The first witness segment id of an orphan row (inverse of ``orphan_row_id``)."""
    if not row_id.startswith(ORPHAN_MARK):
        raise ValueError(f"not an orphan row id: {row_id!r}")
    sid = row_id[len(ORPHAN_MARK):]
    parse_segment_id(sid)
    return sid


def parse_snellgrove_id(uid: str) -> SnellgroveId:
    """Parse a Sanskrit reference unit id such as ``I.5.12``, ``I.5.12a`` or ``I.1.p03``."""
    m = _SNELLGROVE_RE.fullmatch(uid)
    if m is None:
        raise ValueError(f"bad reference unit id {uid!r}; expected e.g. 'I.5.12', 'I.5.12a' or 'I.1.p03'")
    part, chapter = m["part"], int(m["chapter"])
    if part not in PART_CHAPTERS:
        raise ValueError(f"unknown part {part!r} in {uid!r}; parts are {sorted(PART_CHAPTERS)}")
    if chapter > PART_CHAPTERS[part]:
        raise ValueError(f"chapter out of range in {uid!r}: part {part} has {PART_CHAPTERS[part]} chapters")
    if m["prose"] is not None:
        parsed = SnellgroveId(part, chapter, int(m["prose"]), prose=True)
    else:
        parsed = SnellgroveId(part, chapter, int(m["verse"]), half=m["half"])
    if parsed.index < 1 or str(parsed) != uid:
        raise ValueError(f"non-canonical reference unit id {uid!r} (canonical form: {parsed})")
    return parsed


def chapter_key(unit_id: str) -> str:
    """Reference chapter (``"I.5"``) of a Snellgrove id.

    A coordinate id does not encode its chapter; the chapter of a witness segment is
    ``Segment.chapter``, assigned from the concordance at ingest.
    """
    if id_kind(unit_id) != "snellgrove":
        raise ValueError(f"{unit_id!r} is a coordinate id; read the chapter from Segment.chapter")
    return parse_snellgrove_id(unit_id).chapter_key


# --------------------------------------------------------------------------- ordering
def sort_key(unit_id: str) -> tuple:
    """Total order over all id kinds: Snellgrove units, then coordinate ids, then orphan rows.

    * Snellgrove: part, chapter, verses before prose, number, half-verse.
    * Coordinate: witness prefix, then natural order of the line (``1b.7`` < ``1b.10`` <
      ``2a.1``), then content before notes, then ordinal.
    * Orphan rows: after every other id, in the order of their first segment.
    """
    kind = id_kind(unit_id)
    if kind == "snellgrove":
        u = parse_snellgrove_id(unit_id)
        return (0, _PART_RANK[u.part], u.chapter, int(u.prose), u.index, u.half)
    if kind == "orphan":
        return (2, *_segment_key(parse_segment_id(orphan_source(unit_id))))
    return (1, *_segment_key(parse_segment_id(unit_id)))


def _segment_key(ref: SegmentRef) -> tuple:
    return (ref.prefix, _natural(ref.line), int(ref.is_note), ref.ordinal)


def _natural(line: str) -> tuple:
    """Digit runs compare as numbers, other runs as text; tagged so the two never meet."""
    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t) for t in _NATURAL_RE.findall(line))
