"""Matrix assembly: reference units × witnesses → cells with deviation vectors.

See docs/02 §3–4 for the definitions implemented here. Everything computed
automatically carries ``evidence='C'`` (single machine alignment); human
review upgrades it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from . import normalize
from .align import Bead
from .anchors import AnchorLexicon
from .ids import REF_CHAPTERS, make_orphan_id, sort_key
from .segments import Segment

STATUSES = ("PRESENT", "PARTIAL", "ABSENT", "LACUNA", "UNALIGNED", "NA")
COUNTED = {"PRESENT", "PARTIAL", "ABSENT"}
DIMS = ("d_cov", "d_len", "d_lit", "d_ord", "d_split")


@dataclass
class Cell:
    unit: str
    witness: str
    status: str
    span_start: str = ""
    span_end: str = ""
    bead: str = "1:1"
    d_cov: float | None = None
    d_len: float | None = None
    d_lit: float | None = None
    d_ord: float | None = None
    d_split: float | None = None
    similarity: float | None = None
    evidence: str = "C"
    prior: bool = False
    text: str = ""            # short witness text for review; not for redistribution
    chapter: str | None = None
    kind: str = ""
    provenance: str = "auto"

    def to_row(self) -> dict:
        d = asdict(self)
        for k in DIMS + ("similarity",):
            if d[k] is not None:
                d[k] = round(d[k], 4)
        return d


@dataclass
class Unit:
    unit_id: str
    chapter: str
    kind: str          # verse | prose | provisional
    text: str
    length: int
    position: float    # normalised position within chapter (0–1)
    topic: str = ""    # filled later from reference-side annotation
    flag: str | None = None  # LACUNA / ABSENT in a Sanskrit edition


@dataclass
class WitnessMatrix:
    reference: str
    reference_grade: str            # "gold" | "provisional"
    units: dict[str, Unit] = field(default_factory=dict)
    witnesses: list[str] = field(default_factory=list)
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)   # path → sha256

    # ------------------------------------------------------------------ mutation
    def add_units(self, units: Iterable[Unit]) -> None:
        for u in units:
            self.units[u.unit_id] = u

    def add_cell(self, cell: Cell) -> None:
        if cell.witness not in self.witnesses:
            self.witnesses.append(cell.witness)
        self.cells[(cell.unit, cell.witness)] = cell

    def get(self, unit: str, witness: str) -> Cell | None:
        return self.cells.get((unit, witness))

    # ------------------------------------------------------------------ views
    def ordered_units(self) -> list[str]:
        return sorted(self.units, key=sort_key)

    def long_rows(self) -> list[dict]:
        rows = []
        for u in self.ordered_units():
            for w in self.witnesses:
                c = self.cells.get((u, w))
                if c is not None:
                    rows.append(c.to_row())
        return rows

    def wide(self, dim: str) -> list[dict]:
        rows = []
        for u in self.ordered_units():
            row = {"unit": u, "chapter": self.units[u].chapter}
            for w in self.witnesses:
                c = self.cells.get((u, w))
                row[w] = "" if c is None else (c.status if dim == "status" else getattr(c, dim))
            rows.append(row)
        return rows

    def structure(self) -> list[dict]:
        """L1: chapter × witness coverage summary."""
        rows = []
        for ch in REF_CHAPTERS:
            units = [u for u in self.ordered_units() if self.units[u].chapter == ch and not u.startswith("+")]
            for w in self.witnesses:
                cells = [self.cells[(u, w)] for u in units if (u, w) in self.cells]
                counted = [c for c in cells if c.status in COUNTED]
                present = [c for c in counted if c.status in ("PRESENT", "PARTIAL")]
                absent = [c for c in counted if c.status == "ABSENT"]
                orphans = [c for (u, ww), c in self.cells.items() if ww == w and u.startswith("+") and c.chapter == ch]
                rows.append({
                    "chapter": ch, "witness": w, "n_units": len(units), "n_counted": len(counted),
                    "n_present": len(present), "n_absent": len(absent), "n_orphan": len(orphans),
                    "coverage": round(len(present) / len(counted), 4) if counted else "",
                    "mean_d_len": round(statistics.fmean([c.d_len for c in present if c.d_len is not None]), 4) if any(c.d_len is not None for c in present) else "",
                    "mean_d_lit": round(statistics.fmean([c.d_lit for c in present if c.d_lit is not None]), 4) if any(c.d_lit is not None for c in present) else "",
                    "mean_d_ord": round(statistics.fmean([c.d_ord for c in present if c.d_ord is not None]), 4) if any(c.d_ord is not None for c in present) else "",
                })
        return rows

    def profile(self, witness: str) -> list[dict]:
        """Chapter × dimension means for one witness."""
        return [r for r in self.structure() if r["witness"] == witness]

    # ------------------------------------------------------------------ export
    def export(self, out_dir: Path, params: dict | None = None) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "cells.csv", self.long_rows())
        _write_csv(out_dir / "structure.csv", self.structure())
        _write_csv(out_dir / "units.csv", [asdict(u) for u in (self.units[k] for k in self.ordered_units())])
        for dim in ("status",) + DIMS:
            _write_csv(out_dir / f"wide_{dim}.csv", self.wide(dim))
        manifest = {
            "reference": self.reference,
            "reference_grade": self.reference_grade,
            "witnesses": self.witnesses,
            "n_units": len(self.units),
            "n_cells": len(self.cells),
            "inputs": self.inputs,
            "params": params or {},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------- units from reference segments
def units_from_reference(segments: list[Segment], provisional: bool = False) -> list[Unit]:
    """Turn reference-side segments into matrix rows.

    With a Sanskrit TSV the seg_id already is the unit id. With a provisional
    (Tibetan) reference we mint ids ``I.3.t0012`` so they remain sortable and
    visibly provisional.
    """
    by_ch: dict[str, list[Segment]] = {}
    for s in segments:
        if s.chapter is None:
            continue
        by_ch.setdefault(s.chapter, []).append(s)
    units: list[Unit] = []
    for ch, segs in by_ch.items():
        lens = [max(1, normalize.length(s.text, s.lang)) for s in segs]
        total = sum(lens) or 1
        acc = 0
        for i, (s, ln) in enumerate(zip(segs, lens)):
            pos = (acc + ln / 2) / total
            acc += ln
            uid = f"{ch}.t{i + 1:04d}" if provisional else s.seg_id
            units.append(Unit(unit_id=uid, chapter=ch, kind=("provisional" if provisional else s.kind),
                              text=s.text, length=ln, position=pos, flag=(s.extra or {}).get("flag")))
    return units


# ---------------------------------------------------------------------- cells from beads
def cells_from_beads(ref_units: list[Unit], ref_segs: list[Segment], wit_segs: list[Segment],
                     beads: list[Bead], witness: str, lexicon: AnchorLexicon, chapter: str) -> list[Cell]:
    """Convert one chapter's beads into cells (reference rows + orphan rows)."""
    assert len(ref_units) == len(ref_segs)
    wit_lens = [max(1, normalize.length(s.text, s.lang)) for s in wit_segs]
    wit_total = sum(wit_lens) or 1
    wit_pos: list[float] = []
    acc = 0
    for ln in wit_lens:
        wit_pos.append((acc + ln / 2) / wit_total)
        acc += ln
    cells: list[Cell] = []
    for b in beads:
        n, m = b.shape
        if n == 0:
            ws = [wit_segs[j] for j in b.wit]
            text = " ".join(s.text for s in ws)
            cells.append(Cell(
                unit=make_orphan_id(witness, ws[0].start, ws[-1].end), witness=witness, status="PRESENT",
                span_start=ws[0].start, span_end=ws[-1].end, bead="0:1", d_cov=None, d_len=None,
                d_lit=lexicon.transliteration_density(text, ws[0].lang), d_ord=None, d_split=0.0,
                similarity=0.0, text=text[:80], chapter=chapter, kind=ws[0].kind,
            ))
            continue
        ref_len = sum(ref_units[i].length for i in b.ref)
        if m == 0:
            for i in b.ref:
                u = ref_units[i]
                status = "LACUNA" if u.flag == "LACUNA" else "ABSENT"
                cells.append(Cell(unit=u.unit_id, witness=witness, status=status, bead="1:0",
                                  d_cov=0.0, d_split=0.0, similarity=0.0, chapter=chapter, kind=u.kind))
            continue
        ws = [wit_segs[j] for j in b.wit]
        text = " ".join(s.text for s in ws)
        wl = sum(wit_lens[j] for j in b.wit)
        raw_log_ratio = math.log(wl / ref_len) if ref_len else 0.0
        wpos = sum(wit_pos[j] for j in b.wit) / m
        for i in b.ref:
            u = ref_units[i]
            cells.append(Cell(
                unit=u.unit_id, witness=witness, status="PRESENT",
                span_start=ws[0].start, span_end=ws[-1].end, bead=f"{n}:{m}",
                d_cov=1.0, d_len=raw_log_ratio, d_lit=lexicon.transliteration_density(text, ws[0].lang),
                d_ord=abs(wpos - u.position), d_split=math.log(n * m), similarity=b.similarity,
                text=text[:80], chapter=chapter, kind=u.kind,
            ))
    return cells


def calibrate_length_residuals(cells: list[Cell], witness: str) -> dict[str, float]:
    """Subtract the witness' own median log-ratio per reference kind (docs/02 §4.2).

    Without topic annotation the baseline is the median over *all* present
    units of that kind — a proxy that is documented as such in the manifest.
    Also marks PARTIAL when the witness is shorter than half the baseline
    expectation.
    """
    by_kind: dict[str, list[float]] = {}
    for c in cells:
        if c.witness == witness and c.status == "PRESENT" and c.d_len is not None and not c.unit.startswith("+"):
            by_kind.setdefault(c.kind, []).append(c.d_len)
    med = {k: statistics.median(v) for k, v in by_kind.items() if v}
    for c in cells:
        if c.witness == witness and c.d_len is not None and c.kind in med:
            c.d_len = c.d_len - med[c.kind]
            if c.status == "PRESENT" and c.d_len < -math.log(2):
                c.status = "PARTIAL"
                c.d_cov = 0.5
    return med
