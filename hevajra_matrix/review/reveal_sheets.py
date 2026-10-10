"""Reveal sheets: the blind answers beside the machine output, and the final decision.

    <batch>.reveal.csv   the blind sheet's columns (blind answers filled in, for reference),
                         the machine cell (``machine_relation``, ``machine_polarity_flip``,
                         ``machine_convention``, ``machine_wit_loci``, ``machine_quotes``,
                         ``machine_grade``) and the final answer columns, prefilled with the
                         blind decision; ``revised_reason`` is required when the final
                         decision differs from the blind one (relation, witness ids, flags or
                         convention; "" and ``none`` count as the same convention)

``machine_convention`` is the convention flag of the built machine cell (``none`` without
one; empty when there is no cell or the cell is UNALIGNED). A convention the code checks
did not support (V12, ``convention_unsupported``) is shown as ``<convention>:unsupported``
and imported without the suffix. The resolve task has no reveal step. Written and read
through ``review.sheets.export`` and ``review.sheets.import_`` with task "reveal".
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from ..core.io import write_csv
from ..core.types import (FLAG_CONVENTION_UNSUPPORTED, Cell, Link, Relation, Segment, Status, Verdict, WitnessOnlyKind,
                          convention_of)
from ..matrix.status import RELATION_STATUS
from .sampling import ReviewItem, ReviewParams
from .sheets import (
    REVEAL_COLUMNS,
    GlossaryCheck,
    blind_row,
    check_decision,
    context_texts,
    one_line,
    truncate_quote,
)
from .verdicts import (
    CONVENTIONS,
    VerdictRecord,
    format_flags,
    normalise_convention,
    parse_flags,
    parse_ids,
    same_convention,
)

UNSUPPORTED_SUFFIX = ":unsupported"


# --------------------------------------------------------------------------- export
def export_reveal(items: Sequence[ReviewItem], out: Path, batch: str, index: Mapping[str, Segment],
                  ref: Sequence[Segment], wit: Sequence[Segment], cells: Sequence[Cell],
                  links: Mapping[str, Link], blind: Sequence[Verdict]) -> Path:
    """Write ``<batch>.reveal.csv`` for ``items``, each of which needs its blind verdict."""
    if any(i.task == "resolve" for i in items):
        raise ValueError("the resolve task has no reveal step")
    by_item = {v.item_id: v for v in blind}
    by_unit = {c.unit_id: c for c in cells}
    rows = []
    for item in items:
        v = by_item.get(item.item_id)
        if v is None or not v.blind_relation:
            raise ValueError(f"{item.item_id}: no blind verdict to reveal against (import the blind sheet first)")
        cell = by_unit.get(item.unit_id)
        row = blind_row(item, index, ref, wit, cells)
        blind_ids, blind_flags = " ".join(v.blind_wit_ids), format_flags(v.flags)
        row.update(blind_relation=v.blind_relation, blind_wit_loci=blind_ids, blind_convention=v.blind_convention,
                   blind_flags=blind_flags, note=v.note,
                   machine_relation=machine_relation(cell),
                   machine_polarity_flip="1" if cell is not None and cell.polarity_flip else "0",
                   machine_convention=machine_convention(cell),
                   machine_wit_loci=" ".join(cell.wit_ids) if cell else "",
                   machine_quotes=_quotes(links.get(item.unit_id)), machine_grade=cell.grade.value if cell else "",
                   final_relation=v.blind_relation, final_wit_loci=blind_ids, final_convention=v.blind_convention,
                   final_flags=blind_flags)
        rows.append(row)
    path = out / f"{batch}.reveal.csv"
    write_csv(path, rows, REVEAL_COLUMNS, bom=True)
    return path


def machine_relation(cell: Cell | None) -> str:
    if cell is None:
        return ""
    if cell.status is Status.UNALIGNED:
        return f"UNALIGNED:{cell.reason or ''}"
    return cell.relation or cell.status.value


def machine_convention(cell: Cell | None) -> str:
    """The machine cell's convention as shown on the sheet ("" without a decided cell)."""
    if cell is None or cell.status is Status.UNALIGNED:
        return ""
    value = convention_of(cell.flags).value
    return f"{value}{UNSUPPORTED_SUFFIX}" if FLAG_CONVENTION_UNSUPPORTED in cell.flags else value


def _quotes(link: Link | None) -> str:
    if link is None:
        return ""
    return " | ".join(f"{q.side}: {one_line(q.text)}" for q in link.quotes if q.text)


# --------------------------------------------------------------------------- import
def import_reveal(rows: Sequence[Mapping[str, str]], blind: Sequence[Verdict], params: ReviewParams, date: str,
                  instrument_digest: str, wit_lang: str, errors: list[str],
                  check: GlossaryCheck | None = None) -> list[VerdictRecord]:
    """Every blind verdict of the batch, in its order; those on the sheet get their reveal columns."""
    out = {v.item_id: v if isinstance(v, VerdictRecord) else VerdictRecord(**{f: getattr(v, f) for f in
                                                                             Verdict.__dataclass_fields__})
           for v in blind}
    for n, row in enumerate(rows, start=2):
        where = f"line {n}"
        v = out.get(row["item_id"].strip())
        if v is None:
            errors.append(f"{where}: item {row['item_id']} has no committed blind verdict")
            continue
        machine = row["machine_relation"].strip()
        conv = row.get("machine_convention", "").strip().removesuffix(UNSUPPORTED_SUFFIX)
        if conv and conv not in CONVENTIONS:
            errors.append(f"{where}: machine_convention {conv!r} is not one of {', '.join(CONVENTIONS)}")
        v = replace(v, machine_relation=machine, machine_status=_machine_status(machine),
                    machine_polarity_flip=row.get("machine_polarity_flip", "").strip() in {"1", "true", "True"},
                    machine_convention=conv, instrument_digest=instrument_digest or v.instrument_digest)
        value = row["final_relation"].strip()
        if value:
            v = _final(v, row, value, params, date, wit_lang, where, errors, check)
        out[v.item_id] = v
    return list(out.values())


def _final(v: VerdictRecord, row: Mapping[str, str], value: str, params: ReviewParams, date: str, wit_lang: str,
           where: str, errors: list[str], check: GlossaryCheck | None) -> VerdictRecord:
    ids, flags = parse_ids(row["final_wit_loci"]), parse_flags(row["final_flags"])
    # A sheet exported before the convention columns leaves the final convention unannotated
    # (""), and then a convention cannot have been revised on it.
    annotated = "final_convention" in row
    conv = normalise_convention(row["final_convention"]) if annotated else ""
    check_decision(v.unit_id, value, ids, flags, where, errors, conv, check)
    reason = row["revised_reason"].strip()
    revised = (value, ids, flags) != (v.blind_relation, v.blind_wit_ids, v.flags)
    if (revised or (annotated and not same_convention(conv, v.blind_convention))) and not reason:
        errors.append(f"{where}: the final decision differs from the blind one; give revised_reason")
    note = row.get("note", "").strip()
    if reason:
        note = f"{note} | revised: {reason}" if note else f"revised: {reason}"
    texts = context_texts(row)
    quote_zh = v.quote_zh
    if ids != v.blind_wit_ids and all(i in texts for i in ids):
        quote_zh = truncate_quote(" ".join(texts[i] for i in ids), wit_lang, params)
    return replace(v, final_relation=value, final_wit_ids=ids, final_convention=conv, flags=flags, final_date=date,
                   polarity_flip=value == Relation.REVERSAL.value, note=note, quote_zh=quote_zh)


def _machine_status(machine_relation: str) -> str:
    if not machine_relation:
        return ""
    if machine_relation.startswith(f"{Status.UNALIGNED.value}:"):
        return Status.UNALIGNED.value
    if machine_relation in {k.value for k in WitnessOnlyKind}:
        return Status.NA.value
    try:
        return RELATION_STATUS[Relation(machine_relation)].value
    except ValueError:
        return ""
