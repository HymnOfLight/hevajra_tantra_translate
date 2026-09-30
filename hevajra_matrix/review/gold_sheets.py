"""Blind gold annotation sheets (synthesis 5.1, 7.3 A).

    <batch>.ref.csv   one row per reference unit of the window: its text and empty answer
                      columns (``links`` = witness handles, ``relation``, ``polarity_flip``,
                      ``flags``, ``note``)
    <batch>.wit.csv   every witness segment of the given witness chapters (the core window)
                      under a short handle (z0001, ...), with a ``witness_only`` answer column

Gold is blind by construction: nothing on either sheet comes from a machine. The batch is
``<set>_<window>`` (e.g. ``test_w03``). Import returns task ``gold`` verdicts with blind
columns only (gold's blind decision is its final one), plus one verdict per segment marked
witness-only, keyed by its orphan row id. Written and read through ``review.sheets.export``
and ``review.sheets.import_`` with task "gold".
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ..core.ids import orphan_row_id
from ..core.io import write_csv
from ..core.textnorm import fingerprint
from ..core.types import Relation, Segment, WitnessOnlyKind
from ..matrix.build import segment_fingerprint
from .sampling import ReviewItem, ReviewParams
from .sheets import batch_of, check_decision, expect_columns, one_line, parse_flip, read_sheet, truncate_quote
from .verdicts import VerdictRecord, parse_flags, parse_ids

GOLD_REF_COLUMNS = ("row", "unit_id", "fingerprint", "locus", "kind", "text", "links", "relation",
                    "polarity_flip", "flags", "note")
GOLD_WIT_COLUMNS = ("handle", "seg_id", "locus", "kind", "text", "witness_only")
WITNESS_ONLY_KINDS = tuple(k.value for k in WitnessOnlyKind)


def export_gold(items: Sequence[ReviewItem], out: Path, batch: str, index: Mapping[str, Segment],
                wit: Sequence[Segment], witness_chapters: Sequence[str]) -> list[Path]:
    """Write the ref and wit sheets of one gold window; returns both paths."""
    if not witness_chapters:
        raise ValueError("a gold sheet needs witness_chapters (the core window's local chapters)")
    ref_rows = []
    for n, item in enumerate(items, start=1):
        seg = index[item.unit_id]
        ref_rows.append({"row": n, "unit_id": seg.id, "fingerprint": segment_fingerprint(seg), "locus": seg.start,
                         "kind": seg.kind, "text": one_line(seg.text)})
    shown = [s for s in wit if s.local_chapter in set(witness_chapters)]
    width = max(4, len(str(len(shown))))
    wit_rows = [{"handle": f"z{n:0{width}d}", "seg_id": s.id, "locus": s.start, "kind": s.kind,
                 "text": one_line(s.text)} for n, s in enumerate(shown, start=1)]
    ref_path, wit_path = out / f"{batch}.ref.csv", out / f"{batch}.wit.csv"
    write_csv(ref_path, ref_rows, GOLD_REF_COLUMNS, bom=True)
    write_csv(wit_path, wit_rows, GOLD_WIT_COLUMNS, bom=True)
    return [ref_path, wit_path]


def import_gold(path: Path, rows: Sequence[Mapping[str, str]], header: Sequence[str], params: ReviewParams,
                annotator: str, date: str, ref_lang: str, wit_lang: str, minutes: float | None,
                errors: list[str]) -> list[VerdictRecord]:
    """Gold verdicts from a filled ``.ref.csv`` (``rows``) and the ``.wit.csv`` beside it.

    Every reference row must carry a relation (gold windows are complete); problems are
    appended to ``errors``.
    """
    expect_columns(header, GOLD_REF_COLUMNS, path)
    batch = batch_of(path, ".ref.csv")
    wit_path = path.with_name(f"{batch}.wit.csv")
    wit_rows, wit_header = read_sheet(wit_path)
    expect_columns(wit_header, GOLD_WIT_COLUMNS, wit_path)
    handles = {r["handle"].strip(): r for r in wit_rows}
    common = dict(batch_id=batch, task="gold", annotator=annotator, blind_date=date, minutes=minutes)
    out: list[VerdictRecord] = []
    linked: set[str] = set()
    for n, row in enumerate(rows, start=2):
        where = f"{path.name} line {n}"
        value, unit_id = row["relation"].strip(), row["unit_id"].strip()
        if not value:
            errors.append(f"{where}: {unit_id} has no relation (gold windows must be complete)")
            continue
        named = parse_ids(row["links"])
        unknown = [h for h in named if h not in handles]
        if unknown:
            errors.append(f"{where}: unknown handle(s) {unknown}")
        known = [handles[h] for h in named if h in handles]
        ids = tuple(r["seg_id"].strip() for r in known)
        linked.update(ids)
        flags = parse_flags(row["flags"])
        check_decision(unit_id, value, ids, flags, where, errors)
        flip = parse_flip(row["polarity_flip"], where, errors) or value == Relation.REVERSAL.value
        out.append(VerdictRecord(
            item_id=f"gold:{unit_id}", unit_id=unit_id, fingerprint=row["fingerprint"].strip(),
            blind_relation=value, blind_wit_ids=ids, polarity_flip=flip, flags=flags,
            note=row.get("note", "").strip(), quote_ref=truncate_quote(row["text"], ref_lang, params),
            quote_zh=truncate_quote(" ".join(r["text"] for r in known), wit_lang, params),
            **common))  # type: ignore[arg-type]
    for n, row in enumerate(wit_rows, start=2):
        kind, sid = row["witness_only"].strip(), row["seg_id"].strip()
        if not kind:
            continue
        where = f"{wit_path.name} line {n}"
        if kind not in WITNESS_ONLY_KINDS:
            errors.append(f"{where}: witness_only {kind!r} is not one of {', '.join(WITNESS_ONLY_KINDS)}")
            continue
        if sid in linked:
            errors.append(f"{where}: {sid} is both linked and witness-only")
        out.append(VerdictRecord(
            item_id=f"gold:{orphan_row_id(sid)}", unit_id=orphan_row_id(sid),
            fingerprint=fingerprint(row["text"], wit_lang), blind_relation=kind, blind_wit_ids=(sid,),
            quote_zh=truncate_quote(row["text"], wit_lang, params), **common))  # type: ignore[arg-type]
    return out
