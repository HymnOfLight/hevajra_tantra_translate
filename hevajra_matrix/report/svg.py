"""SVG figures of the matrix, without plotting dependencies (adapted from v0.2 ``report.py``).

    status_strip     one thin column per reference unit, coloured by status, with chapter
                     ticks: where the text is present, partial, absent or undecided
    chapter_heatmap  reference chapter x status: the number of cells, shaded by the share
                     of the chapter's cells

Both take ``core.types.Cell`` lists (orphan rows are skipped) and a ``caption`` that the
caller sets from the report level, e.g. "unvalidated instrument output" at level 0.
Colours: green PRESENT, amber PARTIAL, red ABSENT, grey UNALIGNED/LACUNA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..core.ids import REF_CHAPTERS
from ..core.types import Cell, Status

STATUS_COLOR = {Status.PRESENT: "#2b8a3e", Status.PARTIAL: "#f59f00", Status.ABSENT: "#c92a2a",
                Status.LACUNA: "#868e96", Status.UNALIGNED: "#adb5bd", Status.NA: "#f1f3f5"}
HEATMAP_STATUSES = (Status.PRESENT, Status.PARTIAL, Status.ABSENT, Status.UNALIGNED, Status.LACUNA)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _unit_cells(cells: Sequence[Cell]) -> list[Cell]:
    return [c for c in cells if not c.unit_id.startswith("+")]


def status_strip(cells: Sequence[Cell], path: Path, caption: str = "", max_units: int = 4000) -> None:
    """Write the unit-by-status strip of one witness to ``path``."""
    units = _unit_cells(cells)[:max_units]
    px, row_h, left, top = max(1, min(4, 1600 // max(1, len(units)))), 18, 10, 40
    width, height = left + px * len(units) + 10, top + row_h + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           'font-family="sans-serif" font-size="11">',
           f'<text x="{left}" y="18">status per reference unit ({len(units)} units): green PRESENT, amber PARTIAL, '
           f'red ABSENT, grey undecided{_esc(" - " + caption) if caption else ""}</text>']
    for k, c in enumerate(units):
        out.append(f'<rect x="{left + k * px}" y="{top}" width="{px}" height="{row_h - 2}" '
                   f'fill="{STATUS_COLOR.get(c.status, "#f1f3f5")}"/>')
    last = None
    for k, c in enumerate(units):
        if c.chapter != last:
            x = left + k * px
            out.append(f'<line x1="{x}" y1="{top - 4}" x2="{x}" y2="{top + row_h}" stroke="#000" stroke-width="0.5"/>')
            out.append(f'<text x="{x + 2}" y="{top + row_h + 14}" font-size="9">{_esc(c.chapter)}</text>')
            last = c.chapter
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def chapter_heatmap(cells: Sequence[Cell], path: Path, caption: str = "") -> None:
    """Write the chapter x status count heatmap to ``path``."""
    units = _unit_cells(cells)
    seen = {c.chapter for c in units}
    chapters = [ch for ch in REF_CHAPTERS if ch in seen] + sorted(seen - set(REF_CHAPTERS))
    counts = {(c.chapter, c.status): 0 for c in units}
    for c in units:
        counts[(c.chapter, c.status)] += 1
    cell, left, top = 46, 60, 60
    width, height = max(420, left + cell * len(HEATMAP_STATUSES) + 20), top + cell * len(chapters) + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           'font-family="sans-serif" font-size="11">',
           f'<text x="{left}" y="20" font-size="13">cells per chapter and status'
           f'{_esc(" - " + caption) if caption else ""}</text>']
    for j, status in enumerate(HEATMAP_STATUSES):
        out.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 8}" text-anchor="middle" '
                   f'font-size="9">{status.value}</text>')
    for i, chapter in enumerate(chapters):
        y = top + i * cell
        total = sum(counts.get((chapter, s), 0) for s in HEATMAP_STATUSES) or 1
        out.append(f'<text x="{left - 6}" y="{y + cell * 0.6}" text-anchor="end">{_esc(chapter)}</text>')
        for j, status in enumerate(HEATMAP_STATUSES):
            n = counts.get((chapter, status), 0)
            opacity = 0.08 + 0.92 * n / total if n else 0.0
            out.append(f'<rect x="{left + j * cell}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                       f'fill="{STATUS_COLOR[status]}" fill-opacity="{opacity:.2f}" stroke="#dee2e6"/>')
            if n:
                out.append(f'<text x="{left + j * cell + cell / 2 - 1}" y="{y + cell * 0.6}" text-anchor="middle" '
                           f'font-size="10">{n}</text>')
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
