"""Where gold is annotated: purposive dev regions and the probability sample of test windows.

Dev gold (synthesis 5.1, critique A5) is purposive: whole chapters or parts of chapters
chosen for their known difficulties. It is for prompt development and baseline tuning and
never gates. The regions come from ``config/preregistration.yaml: gold.dev_regions``; each
entry names a reference chapter and at most one way to cut it:

    {chapter: I.7}                                          the whole chapter
    {chapter: II.3, around: "D418:17b.6", radius_units: 20} units within +/-20 of the
                                                            first unit starting on that line
    {chapter: II.9, first_units: 56}                        the first 56 units
    {chapter: II.11, last_units: 20, id: II.11-12}          the last 20 units

An optional ``id`` names the region (default: derived from the entry); entries sharing an
``id`` form one region, e.g. the tail of II.11 plus II.12 across the merged Chinese
chapter 20 boundary.

Test windows (G1) are a probability sample: ``n`` windows of ``width`` contiguous units;
each draw picks a chapter with probability proportional to its frame size (units not in a
dev region) and a start uniformly among the admissible starts of that chapter (window
entirely inside the chapter, no dev unit, no overlap with an earlier window).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

from ..config import ConfigError
from ..core.types import Segment

REGION_KEYS = frozenset({"chapter", "around", "radius_units", "first_units", "last_units", "id"})
WINDOWS_COLUMNS = ("window_id", "first_unit", "last_unit", "n_units", "chapter", "seed", "drawn_at")


@dataclass(frozen=True)
class TestWindow:
    """One gold window (a row of ``windows.csv``); ``seed`` is None for purposive dev regions."""

    __test__ = False    # not a pytest test class despite the name

    window_id: str
    first_unit: str
    last_unit: str
    n_units: int
    chapter: str
    seed: int | None = None
    drawn_at: str = ""

    def to_row(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "first_unit": self.first_unit, "last_unit": self.last_unit,
                "n_units": self.n_units, "chapter": self.chapter, "seed": self.seed, "drawn_at": self.drawn_at}

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "TestWindow":
        seed = (row.get("seed") or "").strip()
        return cls(window_id=row["window_id"].strip(), first_unit=row["first_unit"].strip(),
                   last_unit=row["last_unit"].strip(), n_units=int(row["n_units"]), chapter=row["chapter"].strip(),
                   seed=int(seed) if seed else None, drawn_at=(row.get("drawn_at") or "").strip())


# --------------------------------------------------------------------------- dev regions
def dev_region_units(reference_segments: Sequence[Segment], regions: Sequence[Mapping[str, Any]]
                     ) -> dict[str, tuple[str, ...]]:
    """Region id -> reference unit ids (document order) for the preregistered dev regions.

    ``reference_segments`` are the alignable reference units with ``Segment.chapter`` set
    (``Concordance.assign_reference_chapters``). Malformed entries raise ``ConfigError``; an
    ``around`` locus with no unit on that line raises ``ValueError``.
    """
    by_chapter: dict[str, list[Segment]] = {}
    for seg in reference_segments:
        if seg.chapter is not None:
            by_chapter.setdefault(seg.chapter, []).append(seg)
    out: dict[str, list[str]] = {}
    for n, region in enumerate(regions):
        where = f"preregistration gold.dev_regions[{n}]"
        units = by_chapter.get(_chapter(region, where, by_chapter), [])
        region_id, selected = _select(region, units, where)
        bucket = out.setdefault(region_id, [])
        bucket.extend(uid for uid in selected if uid not in bucket)
    return {k: tuple(v) for k, v in out.items()}


def _chapter(region: Mapping[str, Any], where: str, known: Collection[str]) -> str:
    if not isinstance(region, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {region!r}")
    unknown = sorted(set(region) - REGION_KEYS)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(REGION_KEYS)}")
    chapter = region.get("chapter")
    if chapter not in known:
        raise ConfigError(f"{where}: chapter {chapter!r} has no reference units")
    cuts = [k for k in ("around", "first_units", "last_units") if k in region]
    if len(cuts) > 1:
        raise ConfigError(f"{where}: give at most one of around / first_units / last_units, got {cuts}")
    if ("radius_units" in region) != ("around" in region):
        raise ConfigError(f"{where}: around and radius_units go together")
    for key in ("radius_units", "first_units", "last_units"):
        value = region.get(key)
        if key in region and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ConfigError(f"{where}: {key} must be a non-negative integer, got {value!r}")
    return str(chapter)


def _select(region: Mapping[str, Any], units: Sequence[Segment], where: str) -> tuple[str, list[str]]:
    chapter = str(region["chapter"])
    ids = [s.id for s in units]
    if "first_units" in region:
        n = region["first_units"]
        return str(region.get("id", f"{chapter}-first{n}")), ids[:n]
    if "last_units" in region:
        n = region["last_units"]
        return str(region.get("id", f"{chapter}-last{n}")), ids[max(0, len(ids) - n):] if n else []
    if "around" in region:
        locus, radius = str(region["around"]), region["radius_units"]
        prefix, _, line = locus.partition(":")
        hits = [i for i, s in enumerate(units) if s.id.startswith(f"{prefix}:") and s.start == line]
        if not hits:
            raise ValueError(f"{where}: no unit of {chapter} starts on {locus}")
        i = hits[0]
        default = f"{chapter}-around-{prefix}-{line}"
        return str(region.get("id", default)), ids[max(0, i - radius): i + radius + 1]
    return str(region.get("id", chapter)), ids


# --------------------------------------------------------------------------- test windows
def draw_test_windows(units_by_chapter: Mapping[str, Sequence[str]], n: int, width: int,
                      exclude: Collection[str], seed: int, drawn_at: str = "",
                      prefix: str = "w") -> list[TestWindow]:
    """Draw ``n`` non-overlapping windows of ``width`` contiguous units (ids ``w01``, ...).

    ``units_by_chapter`` maps each reference chapter to its unit ids in document order;
    chapters are visited in the mapping's order so the draw is reproducible from ``seed``.
    ``exclude`` (the dev units) is removed from the frame. Raises ``ValueError`` when the
    frame cannot hold ``n`` windows.
    """
    if n < 0 or width < 1:
        raise ValueError(f"need n >= 0 and width >= 1, got n={n}, width={width}")
    rng = random.Random(seed)
    excluded = set(exclude)
    chapters = list(units_by_chapter)
    size = {ch: sum(1 for u in units_by_chapter[ch] if u not in excluded) for ch in chapters}
    taken: dict[str, set[int]] = {ch: set() for ch in chapters}
    out: list[TestWindow] = []
    digits = max(2, len(str(n)))
    for k in range(1, n + 1):
        starts = {ch: _admissible(units_by_chapter[ch], width, excluded, taken[ch]) for ch in chapters}
        open_chapters = [ch for ch in chapters if starts[ch]]
        if not open_chapters:
            raise ValueError(f"the frame holds only {k - 1} windows of width {width}, {n} requested")
        chapter = rng.choices(open_chapters, weights=[size[ch] for ch in open_chapters])[0]
        start = rng.choice(starts[chapter])
        taken[chapter].update(range(start, start + width))
        units = units_by_chapter[chapter]
        out.append(TestWindow(window_id=f"{prefix}{k:0{digits}d}", first_unit=units[start],
                              last_unit=units[start + width - 1], n_units=width, chapter=chapter,
                              seed=seed, drawn_at=drawn_at))
    return out


def _admissible(units: Sequence[str], width: int, excluded: set[str], taken: set[int]) -> list[int]:
    return [i for i in range(len(units) - width + 1)
            if not any(units[j] in excluded or j in taken for j in range(i, i + width))]


def window_units(window: TestWindow, units_by_chapter: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """The unit ids a window covers, from its first to its last unit within its chapter."""
    units = list(units_by_chapter[window.chapter])
    i, j = units.index(window.first_unit), units.index(window.last_unit)
    if j < i:
        raise ValueError(f"window {window.window_id}: last unit precedes first unit")
    return tuple(units[i:j + 1])


def region_window(region_id: str, units: Sequence[str], chapter_of: Mapping[str, str]) -> TestWindow:
    """The ``windows.csv`` record of a dev region (chapters joined by "+" when it spans two)."""
    if not units:
        raise ValueError(f"dev region {region_id} has no units")
    chapters = list(dict.fromkeys(chapter_of[u] for u in units))
    return TestWindow(window_id=region_id, first_unit=units[0], last_unit=units[-1], n_units=len(units),
                      chapter="+".join(chapters))
