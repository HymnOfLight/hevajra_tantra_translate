"""Collation windows: what one T1 request shows the model, and how its handles map back.

A window is one chunk of one reference chapter together with the complete witness text:

    units        at most ``max_ref_units`` consecutive alignable reference units of one
                 reference chapter; the last ``overlap_next`` of them are repeated at the
                 start of the next chunk of the same chapter (V10 compares the two readings)
    text         the whole witness text: its alignable segments plus headings and the
                 witness's own notes, in document order. Taisho footnotes are not segments
                 and can never be part of it.
    core_locals  the witness chapters mapped to the reference chapter by the concordance,
                 plus ``neighbours`` chapters on each side (``Concordance.core_window``).
                 The core window is where counterparts are expected and what the model must
                 account for line by line; links outside it are allowed and reported (V7).

Handles are prompt-local names, so the model never sees a coordinate: ``r001``, ``r002``,
... number the units of the chunk, and ``z0001``, ``z0002``, ... number the segments of the
whole witness text in order (the same handle names the same segment in every window built
over the same text, which keeps the cached context identical). Handles are derived from
the order of ``units`` and ``text`` and never stored, so a perturbed window (a segment
deleted, a chapter swapped) renumbers itself and leaks nothing through gaps.

``Window`` validates itself on construction (also on ``dataclasses.replace``): one
reference and one witness, one language each, no duplicate ids, only allowed kinds. This
is what enforces "one reference and one witness per call" by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..config import ConfigError, from_mapping
from ..core.ids import REF_CHAPTERS
from ..core.types import Segment
from ..ingest import CONTENT_KINDS
from ..registry import Concordance

# Witness segment kinds shown in the context: the text proper, headings and the witness's
# own notes (translator notes belong to the witness). Front matter ("meta"), colophons
# and paratext are left out.
WITNESS_KINDS = CONTENT_KINDS | frozenset({"head", "note"})
# Kind names shown to the model; every other kind is shown as it is.
PROMPT_KIND = MappingProxyType({"verse_line": "verse"})


def prompt_kind(kind: str) -> str:
    return PROMPT_KIND.get(kind, kind)


@dataclass(frozen=True)
class WindowParams:
    """``config/run.yaml: windows``."""

    max_ref_units: int = 150
    overlap: int = 10
    neighbours: int = 1

    def __post_init__(self) -> None:
        for name in ("max_ref_units", "overlap", "neighbours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"windows.{name} must be a non-negative integer, got {value!r}")
        if self.max_ref_units < 1 or self.overlap >= self.max_ref_units:
            raise ConfigError("windows: need max_ref_units >= 1 and overlap < max_ref_units")

    @classmethod
    def from_config(cls, run: Mapping[str, Any]) -> "WindowParams":
        return from_mapping(cls, run.get("windows"), "run.yaml: windows")


@dataclass(frozen=True)
class Window:
    """One collation request's material; see the module docstring."""

    key: str                          # "<chapter>.w<part>", e.g. "II.9.w1"
    chapter: str                      # reference chapter of every unit
    units: tuple[Segment, ...]
    text: tuple[Segment, ...]
    core_locals: frozenset[str]
    overlap_next: int = 0

    def __post_init__(self) -> None:
        if not self.units or not self.text:
            raise ValueError(f"window {self.key}: needs reference units and a witness text")
        _uniform(self.units, "reference", self.key)
        _uniform(self.text, "witness", self.key)
        for unit in self.units:
            if unit.kind not in CONTENT_KINDS or unit.chapter != self.chapter:
                raise ValueError(f"window {self.key}: unit {unit.id} ({unit.kind}, chapter {unit.chapter}) "
                                 f"is not an alignable unit of chapter {self.chapter}")
        bad = [s.id for s in self.text if s.kind not in WITNESS_KINDS]
        if bad:
            raise ValueError(f"window {self.key}: witness kinds outside {sorted(WITNESS_KINDS)}: {bad[:5]}")
        for label, segs in (("unit", self.units), ("witness segment", self.text)):
            if len({s.id for s in segs}) != len(segs):
                raise ValueError(f"window {self.key}: duplicate {label} ids")
        if not 0 <= self.overlap_next < len(self.units):
            raise ValueError(f"window {self.key}: overlap_next must be in [0, {len(self.units)})")

    # ------------------------------------------------------------------ identity
    @property
    def reference(self) -> str:
        return self.units[0].witness

    @property
    def witness(self) -> str:
        return self.text[0].witness

    @property
    def ref_lang(self) -> str:
        return self.units[0].lang

    @property
    def wit_lang(self) -> str:
        return self.text[0].lang

    # ------------------------------------------------------------------ handles
    @cached_property
    def ref_handles(self) -> Mapping[str, str]:
        """Handle ("r001") -> reference unit id, in chunk order."""
        width = max(3, len(str(len(self.units))))
        return MappingProxyType({f"r{i:0{width}d}": u.id for i, u in enumerate(self.units, start=1)})

    @cached_property
    def wit_handles(self) -> Mapping[str, str]:
        """Handle ("z0001") -> witness segment id, in text order."""
        width = max(4, len(str(len(self.text))))
        return MappingProxyType({f"z{i:0{width}d}": s.id for i, s in enumerate(self.text, start=1)})

    @cached_property
    def handle_of(self) -> Mapping[str, str]:
        """Segment or unit id -> its handle (reference and witness ids never collide)."""
        return MappingProxyType({sid: h for table in (self.ref_handles, self.wit_handles)
                                 for h, sid in table.items()})

    @cached_property
    def position(self) -> Mapping[str, int]:
        """Witness segment id -> 0-based position in the witness text."""
        return MappingProxyType({s.id: i for i, s in enumerate(self.text)})

    @cached_property
    def segment(self) -> Mapping[str, Segment]:
        """Id -> segment, for units and witness segments."""
        return MappingProxyType({s.id: s for s in (*self.units, *self.text)})

    # ------------------------------------------------------------------ core window
    @cached_property
    def core(self) -> frozenset[str]:
        """Ids of the witness segments whose local chapter is in the core window."""
        return frozenset(s.id for s in self.text if s.local_chapter in self.core_locals)

    def core_ranges(self) -> list[tuple[str, str]]:
        """Maximal runs of consecutive core handles, as (first, last) pairs."""
        runs: list[list[str]] = []
        previous = -2
        for i, (handle, sid) in enumerate(self.wit_handles.items()):
            if sid in self.core:
                if i == previous + 1:
                    runs[-1].append(handle)
                else:
                    runs.append([handle])
                previous = i
        return [(run[0], run[-1]) for run in runs]

    @property
    def overlap_ids(self) -> tuple[str, ...]:
        """Ids of the trailing units repeated at the start of the next chunk."""
        return tuple(u.id for u in self.units[len(self.units) - self.overlap_next:]) if self.overlap_next else ()


def _uniform(segments: Sequence[Segment], side: str, key: str) -> None:
    witnesses = {s.witness for s in segments}
    langs = {s.lang for s in segments}
    if len(witnesses) != 1 or len(langs) != 1:
        raise ValueError(f"window {key}: the {side} side must come from one witness in one language, "
                         f"got witnesses {sorted(witnesses)} and languages {sorted(langs)}")


# --------------------------------------------------------------------------- planning
def chunk_bounds(n: int, max_units: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open [start, end) bounds of the fewest chunks of at most ``max_units`` that
    cover ``n`` units, each sharing exactly ``overlap`` units with the next.

    Chunk sizes differ by at most one unit (larger chunks first) instead of leaving a
    small remainder at the end, so no chunk is read with much less context than another.
    """
    if n <= 0:
        return []
    if not 0 <= overlap < max_units:
        raise ValueError("need 0 <= overlap < max_units")
    if n <= max_units:
        return [(0, n)]
    k = -(-(n - overlap) // (max_units - overlap))          # ceil division: fewest chunks
    base, extra = divmod(n + (k - 1) * overlap, k)          # k chunk sizes summing to n + shared units
    bounds, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < extra else 0)
        bounds.append((start, end))
        start = end - overlap
    return bounds


def plan(reference: Sequence[Segment], witness: Sequence[Segment], concordance: Concordance,
         params: WindowParams) -> list[Window]:
    """All windows of a (reference, witness) pair, in reference chapter order.

    ``reference`` and ``witness`` are whole ingested texts in document order; only
    alignable reference units (``CONTENT_KINDS``) and the witness kinds in
    ``WITNESS_KINDS`` are used. Every reference unit must carry its reference chapter
    (for Derge: ``Concordance.assign_reference_chapters``); a missing one is an error
    rather than a silently skipped unit.
    """
    units = [s for s in reference if s.kind in CONTENT_KINDS]
    text = tuple(s for s in witness if s.kind in WITNESS_KINDS)
    if not units or not text:
        raise ValueError("plan: no alignable reference units or no witness text")
    by_chapter: dict[str, list[Segment]] = {}
    for unit in units:
        if unit.chapter not in REF_CHAPTERS:
            raise ValueError(f"reference unit {unit.id} has no known reference chapter ({unit.chapter!r}); "
                             "assign chapters with Concordance.assign_reference_chapters first")
        by_chapter.setdefault(unit.chapter, []).append(unit)
    witness_id = text[0].witness
    windows: list[Window] = []
    for chapter in (c for c in REF_CHAPTERS if c in by_chapter):
        chapter_units = by_chapter[chapter]
        core_locals = concordance.core_window(chapter, witness_id, params.neighbours)
        bounds = chunk_bounds(len(chapter_units), params.max_ref_units, params.overlap)
        for part, (a, b) in enumerate(bounds, start=1):
            windows.append(Window(
                key=f"{chapter}.w{part}", chapter=chapter, units=tuple(chapter_units[a:b]), text=text,
                core_locals=core_locals, overlap_next=params.overlap if part < len(bounds) else 0,
            ))
    return windows
