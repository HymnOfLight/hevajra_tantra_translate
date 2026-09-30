"""Row identifiers for the witness matrix.

Scheme (docs/02 §1.1):
    I.5.12      verse 12 of Part I chapter 5 (Snellgrove numbering)
    I.5.12a     first half-verse
    I.1.p03     third prose proposition of Part I chapter 1
    +zh_T0892_song:0594c15-0594c17
                orphan row: material present only in one witness
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PART_CHAPTERS = {"I": 11, "II": 12}
REF_CHAPTERS: list[str] = [f"I.{i}" for i in range(1, 12)] + [f"II.{i}" for i in range(1, 13)]

_UNIT_RE = re.compile(r"^(I|II)\.(\d{1,2})\.(?:(\d{1,3})([ab])?|p(\d{1,3})|t(\d{1,4}))$")
_ORPHAN_RE = re.compile(r"^\+([A-Za-z0-9_]+):(.+)$")


@dataclass(frozen=True)
class UnitId:
    part: str
    chapter: int
    index: int
    kind: str  # "verse" | "prose" | "orphan"
    sub: str | None = None
    witness: str | None = None
    coords: str | None = None

    @property
    def chapter_key(self) -> str:
        return f"{self.part}.{self.chapter}"

    def __str__(self) -> str:
        if self.kind == "orphan":
            return f"+{self.witness}:{self.coords}"
        if self.kind == "prose":
            return f"{self.part}.{self.chapter}.p{self.index:02d}"
        if self.kind == "provisional":
            return f"{self.part}.{self.chapter}.t{self.index:04d}"
        return f"{self.part}.{self.chapter}.{self.index}{self.sub or ''}"


def parse_unit_id(s: str) -> UnitId:
    m = _ORPHAN_RE.match(s)
    if m:
        return UnitId(part="", chapter=0, index=0, kind="orphan", witness=m.group(1), coords=m.group(2))
    m = _UNIT_RE.match(s)
    if not m:
        raise ValueError(f"bad unit id: {s!r}")
    part, chap = m.group(1), int(m.group(2))
    if chap < 1 or chap > PART_CHAPTERS[part]:
        raise ValueError(f"chapter out of range for part {part}: {s!r}")
    if m.group(5) is not None:
        return UnitId(part, chap, int(m.group(5)), "prose")
    if m.group(6) is not None:
        return UnitId(part, chap, int(m.group(6)), "provisional")
    return UnitId(part, chap, int(m.group(3)), "verse", sub=m.group(4))


def chapter_key(unit_id: str) -> str:
    return parse_unit_id(unit_id).chapter_key


def make_orphan_id(witness: str, start: str, end: str | None = None) -> str:
    coords = start if end is None or end == start else f"{start}-{end}"
    return f"+{witness}:{coords}"


def sort_key(unit_id: str) -> tuple:
    u = parse_unit_id(unit_id)
    if u.kind == "orphan":
        return (9, 0, 0, 0, u.witness or "", u.coords or "")
    part_rank = 0 if u.part == "I" else 1
    kind_rank = {"verse": 0, "prose": 1, "provisional": 2}[u.kind]
    return (part_rank, u.chapter, kind_rank, u.index, u.sub or "", "")
