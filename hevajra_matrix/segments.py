"""Common segment record produced by every ingester."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass
class Segment:
    witness: str
    seg_id: str            # witness-local id, e.g. "zh:0587c11:1"
    lang: str
    text: str
    start: str             # coordinate in the witness' native system
    end: str
    kind: str              # prose | verse_line | head | colophon | meta | mantra
    chapter: str | None = None   # reference chapter key (I.3) once mapped
    local_chapter: int | None = None
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        d["extra"] = "" if not self.extra else str(self.extra)
        return d


def by_chapter(segments: Iterable[Segment]) -> dict[str | None, list[Segment]]:
    out: dict[str | None, list[Segment]] = {}
    for s in segments:
        out.setdefault(s.chapter, []).append(s)
    return out
