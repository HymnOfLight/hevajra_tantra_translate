"""Write the built matrix to ``<run>/matrix/`` (synthesis 2, stage 5).

    cells.csv           one row per cell: reference units first (reference order), then
                        orphan rows (witness order); columns ``CELL_COLUMNS``
    units.csv           one row per matrix row: its type, chapter and, when the reference
                        segments are passed, kind, fingerprint and length
    wide_status.csv     one row per matrix row, one status column per witness
    stale_verdicts.csv  gold rows and verdicts not applied because the text changed

Headers are English and ASCII. Files hold ids, statuses and numbers only (no text), so a
run's matrix can be shared without licence concerns. Floats are rounded to 4 decimals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..core.io import write_csv
from ..core.textnorm import length
from ..core.types import Cell, Segment
from .build import StaleVerdict, segment_fingerprint

CELL_COLUMNS: tuple[str, ...] = (
    "unit_id", "row_type", "witness", "chapter", "status", "relation", "polarity_flip", "grade",
    "reason", "wit_ids", "flags", "d_len", "d_ord", "d_lit", "source",
)
UNIT_COLUMNS: tuple[str, ...] = ("unit_id", "row_type", "chapter", "kind", "fingerprint", "length")
STALE_COLUMNS: tuple[str, ...] = (
    "unit_id", "batch_id", "item_id", "task", "reason", "stored_fingerprint", "current_fingerprint",
    "quote_ref", "quote_zh",
)
ROW_UNIT, ROW_ORPHAN = "unit", "orphan"
FILES = ("cells.csv", "units.csv", "wide_status.csv", "stale_verdicts.csv")


def _num(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def cell_row(cell: Cell, row_type: str) -> dict[str, str]:
    return {
        "unit_id": cell.unit_id, "row_type": row_type, "witness": cell.witness, "chapter": cell.chapter,
        "status": cell.status.value, "relation": cell.relation or "",
        "polarity_flip": "true" if cell.polarity_flip else "false", "grade": cell.grade.value,
        "reason": cell.reason or "", "wit_ids": " ".join(cell.wit_ids), "flags": ";".join(sorted(cell.flags)),
        "d_len": _num(cell.d_len), "d_ord": _num(cell.d_ord), "d_lit": _num(cell.d_lit), "source": cell.source,
    }


def write_matrix(cells: Sequence[Cell], orphan_cells: Sequence[Cell], stale: Iterable[StaleVerdict],
                 out_dir: Path, *, reference_segments: Sequence[Segment] = ()) -> list[Path]:
    """Write the four matrix files into ``out_dir``; returns their paths.

    ``cells`` may hold several witnesses (rows are keyed by unit id; one status column per
    witness in ``wide_status.csv``, in first-seen order).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    typed = [(c, ROW_UNIT) for c in cells] + [(c, ROW_ORPHAN) for c in orphan_cells]
    write_csv(out / "cells.csv", [cell_row(c, t) for c, t in typed], CELL_COLUMNS)

    segs = {s.id: s for s in reference_segments}
    rows: dict[str, dict[str, str]] = {}
    for c, row_type in typed:
        if c.unit_id in rows:
            continue
        seg = segs.get(c.unit_id)
        rows[c.unit_id] = {
            "unit_id": c.unit_id, "row_type": row_type, "chapter": c.chapter,
            "kind": seg.kind if seg else "", "fingerprint": segment_fingerprint(seg) if seg else "",
            "length": str(length(seg.text, seg.lang)) if seg else "",
        }
    write_csv(out / "units.csv", rows.values(), UNIT_COLUMNS)

    witnesses = list(dict.fromkeys(c.witness for c, _ in typed))
    wide: dict[str, dict[str, str]] = {}
    for c, _ in typed:
        wide.setdefault(c.unit_id, {"unit_id": c.unit_id, "chapter": c.chapter})[c.witness] = c.status.value
    write_csv(out / "wide_status.csv", wide.values(), ("unit_id", "chapter", *witnesses))

    write_csv(out / "stale_verdicts.csv", [
        {"unit_id": s.unit_id, "batch_id": s.batch_id, "item_id": s.item_id, "task": s.task, "reason": s.reason,
         "stored_fingerprint": s.stored_fingerprint, "current_fingerprint": s.current_fingerprint,
         "quote_ref": s.quote_ref, "quote_zh": s.quote_zh} for s in stale
    ], STALE_COLUMNS)
    return [out / name for name in FILES]
