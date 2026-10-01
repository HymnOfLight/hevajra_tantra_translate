"""Review sheets: CSV files with text for annotators, and their import as committed verdicts.

Sheets (synthesis 7.3; exact columns in the ``*_COLUMNS`` constants of this module,
``gold_sheets`` and ``topic_sheets``) are UTF-8 with
a BOM so spreadsheets open them cleanly. They contain licensed text, so they are written
under ``runs/<id>/review/`` only and never committed (licence rule, synthesis 7.2). Beside
every text-bearing sheet, ``witness_text/<local chapter>.txt`` holds each witness chapter
in full ("<segment id> TAB <kind> TAB <text>") for searching.

    task           files                              blind?
    gold           <batch>.ref.csv + <batch>.wit.csv  yes: no machine output at all
                                                      (``review.gold_sheets``)
    verify, audit  <batch>.blind.csv                  yes: no machine_* column
    resolve        <batch>.blind.csv                  yes, and there is no reveal step; one
                                                      extra column ``hint_other_model`` shows
                                                      what a substituted model proposed
                                                      (stripped on import)
    reveal         <batch>.reveal.csv                 blind answers + machine output + final
    topics         topics_<batch>.csv                 reference text only, with prelabels
    topics_second  topics_<batch>.second.csv          reference text only, NO prelabel
                                                      columns (critique A6)
                                                      (``review.topic_sheets``)

Witness segments are named by their segment ids everywhere (``zh_context_loci``,
``blind_wit_loci``, ...), separated by spaces. ``zh_context`` shows the witness region
spanned by the machine links of the three reference units on either side of the unit,
never the unit's own link (the unit's counterpart may of course lie inside that region);
each line is "[<segment id>] <text>".

Import (``import_``) keeps decisions and ids and drops every text column; the only text
kept is a short quote per side, cut to ``run.yaml: review.quote_max_chars`` (and never
more than 60 characters), to help re-resolve a verdict when its fingerprint goes stale.
"""

from __future__ import annotations

import csv
import re
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from ..core.ids import orphan_source
from ..core.io import write_csv
from ..core.types import Cell, Link, Relation, Segment, Status, Verdict, WitnessOnlyKind
from ..matrix.build import segment_fingerprint
from ..matrix.status import RELATION_STATUS
from ..topics.prelabel import Prelabel
from .sampling import ReviewItem, ReviewParams
from .verdicts import (
    FLAGS,
    MAX_COMMITTED_QUOTE,
    ORPHAN_VALUES,
    UNIT_VALUES,
    VerdictError,
    VerdictRecord,
    format_flags,
    is_orphan,
    parse_bool,
    parse_flags,
    parse_ids,
)

BLIND_COLUMNS = ("item_id", "task", "unit_id", "fingerprint", "locus", "ref_text", "zh_context_loci",
                 "zh_context", "blind_relation", "blind_wit_loci", "blind_flags", "note")
REVEAL_COLUMNS = BLIND_COLUMNS + ("machine_relation", "machine_polarity_flip", "machine_wit_loci", "machine_quotes",
                                  "machine_grade",
                                  "final_relation", "final_wit_loci", "final_flags", "revised_reason")
HINT_COLUMN = "hint_other_model"
RESOLVE_COLUMNS = BLIND_COLUMNS + (HINT_COLUMN,)
HINT_LABEL = "FROM ANOTHER MODEL (substituted; not the instrument, never measured): "
TEXT_COLUMNS = frozenset({"text", "ref_text", "zh_context", "context_before", "context_after", "machine_quotes"})
SHEET_TASKS = ("gold", "verify", "audit", "resolve", "reveal", "topics", "topics_second")
BLIND_TASKS = ("verify", "audit", "resolve")

CONTEXT_UNITS = 3                 # reference units on each side whose links span zh_context
MAX_CONTEXT_SEGMENTS = 40         # a longer span (e.g. a relocated neighbour) is cut around its median
TSHEG = "\u0f0b"                  # Tibetan intersyllabic mark
_CONTEXT_LINE = re.compile(r"^\[([^\]\s]+)\] ?(.*)$")


# --------------------------------------------------------------------------- export
def export(items: Sequence[ReviewItem], task: str, out_dir: Path, segments: Sequence[Segment],
           cells: Sequence[Cell], *, batch: str | None = None, links: Mapping[str, Link] | None = None,
           blind: Sequence[Verdict] = (), prelabels: Mapping[str, Prelabel] | None = None,
           witness_chapters: Sequence[str] = (), hints: Mapping[str, Sequence[Link]] | None = None) -> list[Path]:
    """Write the sheet(s) of one batch; returns the written paths (sheets first).

    ``segments``  reference and witness segments (both texts; topics sheets use only the
                  reference text of their units, so they stay blind to the witness)
    ``cells``     the MACHINE matrix (unit and orphan cells); its witness is the witness text
                  (gold and topics sheets need no cells)
    ``links``     consensus ``Alignment.by_ref()``, for ``machine_quotes`` (reveal only)
    ``blind``     the committed blind verdicts of the batch (reveal only)
    ``prelabels`` T3 prelabels by unit id (topics only)
    ``witness_chapters``  witness local chapters shown on a gold wit sheet (the core window)
    ``hints``     reference unit id -> substituted-model hint links (``Collation.hints``;
                  resolve only): shown as relation and witness loci in ``hint_other_model``
    """
    if task not in SHEET_TASKS:
        raise ValueError(f"unknown sheet task {task!r}; expected one of {SHEET_TASKS}")
    batch = batch or task
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {s.id: s for s in segments}
    missing = [i.unit_id for i in items if (orphan_source(i.unit_id) if is_orphan(i.unit_id) else i.unit_id) not in index]
    if missing:
        raise ValueError(f"items name unknown units: {missing[:5]}")
    if task in ("topics", "topics_second"):
        from .topic_sheets import export_topics      # local import: topic_sheets reuses this module's helpers
        return [export_topics(items, task, out, batch, segments, prelabels or {})]
    ref, wit = split_texts(items, segments, cells)
    if task == "gold":
        from .gold_sheets import export_gold          # local import: gold_sheets reuses this module's helpers
        paths = export_gold(items, out, batch, index, wit, witness_chapters)
    elif task == "reveal":
        paths = [_reveal_sheet(items, out, batch, index, ref, wit, cells, links or {}, blind)]
    else:
        if any(i.task != task for i in items):
            raise ValueError(f"every item of a {task} sheet must have task {task}")
        rows = [_blind_row(i, index, ref, wit, cells) for i in items]
        columns = BLIND_COLUMNS
        if task == "resolve":
            columns = RESOLVE_COLUMNS
            for row in rows:
                row[HINT_COLUMN] = hint_text((hints or {}).get(row["unit_id"], ()))
        path = out / f"{batch}.blind.csv"
        write_csv(path, rows, columns, bom=True)
        paths = [path]
    return paths + write_witness_text(wit, out / "witness_text")


def hint_text(links: Sequence[Link]) -> str:
    """``hint_other_model`` cell: each distinct hint as "<relation> [<witness loci>]"."""
    shown = dict.fromkeys(f"{link.relation} [{' '.join(link.wit_ids)}]" for link in links)
    return HINT_LABEL + " | ".join(shown) if shown else ""


def split_texts(items: Sequence[ReviewItem], segments: Sequence[Segment],
                cells: Sequence[Cell]) -> tuple[list[Segment], list[Segment]]:
    """(reference segments, witness segments), each in the given order.

    The witness is the cells' witness; without cells (e.g. gold before any machine run) it
    is every text other than the one holding the items' reference units.
    """
    if cells:
        witness = cells[0].witness
        return [s for s in segments if s.witness != witness], [s for s in segments if s.witness == witness]
    index = {s.id: s for s in segments}
    units = [i.unit_id for i in items if not is_orphan(i.unit_id)]
    if not units:
        raise ValueError("pass the cells: the witness text cannot be told from orphan items alone")
    reference = index[units[0]].witness
    return [s for s in segments if s.witness == reference], [s for s in segments if s.witness != reference]


def write_witness_text(witness_segments: Sequence[Segment], out_dir: Path) -> list[Path]:
    """One searchable file per witness local chapter: "<id> TAB <kind> TAB <text>" lines."""
    chapters: dict[str, list[Segment]] = {}
    for s in witness_segments:
        chapters.setdefault(s.local_chapter or "front", []).append(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for local, segs in chapters.items():
        path = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', local)}.txt"
        path.write_text("".join(f"{s.id}\t{s.kind}\t{one_line(s.text)}\n" for s in segs), encoding="utf-8")
        paths.append(path)
    return paths


def one_line(text: str) -> str:
    return " ".join(text.split())


def _blind_row(item: ReviewItem, index: Mapping[str, Segment], ref: Sequence[Segment],
               wit: Sequence[Segment], cells: Sequence[Cell]) -> dict[str, str]:
    if is_orphan(item.unit_id):
        seg = index[orphan_source(item.unit_id)]
        context = _around(wit, seg.id)
        ref_text = ""
    else:
        seg = index[item.unit_id]
        context = context_segments(item.unit_id, ref, wit, cells)
        ref_text = one_line(seg.text)
    return {"item_id": item.item_id, "task": item.task, "unit_id": item.unit_id, "fingerprint": segment_fingerprint(seg),
            "locus": seg.start, "ref_text": ref_text, "zh_context_loci": " ".join(s.id for s in context),
            "zh_context": "\n".join(f"[{s.id}] {one_line(s.text)}" for s in context)}


def context_segments(unit_id: str, ref: Sequence[Segment], wit: Sequence[Segment],
                     cells: Sequence[Cell]) -> list[Segment]:
    """Witness segments spanned by the links of units u-3..u+3 except u (in witness order)."""
    order = [s.id for s in ref]
    i = order.index(unit_id)
    neighbours = set(order[max(i - CONTEXT_UNITS, 0):i] + order[i + 1:i + 1 + CONTEXT_UNITS])
    position = {s.id: n for n, s in enumerate(wit)}
    hits = sorted(position[w] for c in cells if c.unit_id in neighbours for w in c.wit_ids if w in position)
    if not hits:
        return []
    lo, hi = hits[0], hits[-1]
    if hi - lo + 1 > MAX_CONTEXT_SEGMENTS:
        mid = hits[len(hits) // 2]
        lo = max(mid - MAX_CONTEXT_SEGMENTS // 2, 0)
        hi = lo + MAX_CONTEXT_SEGMENTS - 1
    return list(wit[lo:hi + 1])


def _around(wit: Sequence[Segment], seg_id: str) -> list[Segment]:
    i = next(n for n, s in enumerate(wit) if s.id == seg_id)
    return list(wit[max(i - CONTEXT_UNITS, 0):i + CONTEXT_UNITS + 1])


def _reveal_sheet(items: Sequence[ReviewItem], out: Path, batch: str, index: Mapping[str, Segment],
                  ref: Sequence[Segment], wit: Sequence[Segment], cells: Sequence[Cell],
                  links: Mapping[str, Link], blind: Sequence[Verdict]) -> Path:
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
        row = _blind_row(item, index, ref, wit, cells)
        blind_ids, blind_flags = " ".join(v.blind_wit_ids), format_flags(v.flags)
        row.update(blind_relation=v.blind_relation, blind_wit_loci=blind_ids, blind_flags=blind_flags, note=v.note,
                   machine_relation=_machine_relation(cell),
                   machine_polarity_flip="1" if cell is not None and cell.polarity_flip else "0", machine_wit_loci=" ".join(cell.wit_ids) if cell else "",
                   machine_quotes=_quotes(links.get(item.unit_id)), machine_grade=cell.grade.value if cell else "",
                   final_relation=v.blind_relation, final_wit_loci=blind_ids, final_flags=blind_flags)
        rows.append(row)
    path = out / f"{batch}.reveal.csv"
    write_csv(path, rows, REVEAL_COLUMNS, bom=True)
    return path


def _machine_relation(cell: Cell | None) -> str:
    if cell is None:
        return ""
    if cell.status is Status.UNALIGNED:
        return f"UNALIGNED:{cell.reason or ''}"
    return cell.relation or cell.status.value


def _quotes(link: Link | None) -> str:
    if link is None:
        return ""
    return " | ".join(f"{q.side}: {one_line(q.text)}" for q in link.quotes if q.text)


# --------------------------------------------------------------------------- import
def truncate_quote(text: str, lang: str, params: ReviewParams) -> str:
    """Whitespace-normalised ``text`` cut to the configured maximum for ``lang``.

    Tibetan is cut back to the last whole syllable when the limit falls inside one.
    """
    text = one_line(text)
    limit = min(params.quote_max_chars.get(lang, MAX_COMMITTED_QUOTE), MAX_COMMITTED_QUOTE)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if lang == "bo" and text[limit] not in (TSHEG, " "):
        last = max(cut.rfind(TSHEG), cut.rfind(" "))
        if last > 0:
            cut = cut[:last + 1]
    return cut.strip()


def import_(path: Path, task: str, *, params: ReviewParams, annotator: str = "", date: str = "",
            items: Sequence[ReviewItem] = (), blind: Sequence[Verdict] = (), instrument_digest: str = "",
            ref_lang: str = "bo", wit_lang: str = "zh", minutes: float | None = None) -> list[VerdictRecord]:
    """Read a filled sheet and return committed-format verdicts (no text columns).

    ``gold``      ``path`` is the ``.ref.csv``; the ``.wit.csv`` beside it maps handles to ids
    ``verify``, ``audit``, ``resolve``  the ``.blind.csv``; rows without ``blind_relation``
                  are skipped (a batch may be imported in parts); ``items`` (the plan)
                  supplies stratum and inclusion probability; resolve copies blind to final
    ``reveal``    the ``.reveal.csv``; ``blind`` (the committed blind verdicts of the batch)
                  are returned with their final and machine columns filled
    Every problem in the file is reported at once as a ``VerdictError``.
    """
    rows, header = read_sheet(Path(path))
    errors: list[str] = []
    if task == "gold":
        from .gold_sheets import import_gold
        out = import_gold(Path(path), rows, header, params, annotator, date, ref_lang, wit_lang, minutes, errors)
    elif task in BLIND_TASKS:
        # resolve sheets carry the hint column (sheets exported before it existed do not)
        expect_columns(header, RESOLVE_COLUMNS if task == "resolve" and HINT_COLUMN in header else BLIND_COLUMNS,
                       path)
        plan = {i.item_id: i for i in items}
        out = [v for n, row in enumerate(rows, start=2)
               if (v := _import_blind(row, n, task, plan, params, annotator, date, ref_lang, wit_lang,
                                      minutes, errors, batch_of(path, ".blind.csv"))) is not None]
    elif task == "reveal":
        expect_columns(header, REVEAL_COLUMNS, path)
        out = _import_reveal(rows, blind, params, date, instrument_digest, wit_lang, errors)
    else:
        raise ValueError(f"import_ handles gold, verify, audit, resolve and reveal, not {task!r}")
    if errors:
        raise VerdictError(f"{path}: {len(errors)} problem(s):\n  " + "\n  ".join(errors))
    return out


def read_sheet(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Rows (values never None) and the header of a sheet; a byte-order mark is ignored."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [{k: (v or "") for k, v in row.items() if k is not None} for row in reader]
        return rows, list(reader.fieldnames or ())


def expect_columns(header: Sequence[str], columns: Sequence[str], path: Path | str) -> None:
    if sorted(header) != sorted(columns):
        raise VerdictError(f"{path}: expected columns {', '.join(columns)}; found {', '.join(header)}")


def batch_of(path: Path | str, suffix: str) -> str:
    name = Path(path).name
    if not name.endswith(suffix):
        raise VerdictError(f"{path}: expected a file name ending in {suffix}")
    return name[: -len(suffix)]


def _context_texts(row: Mapping[str, str]) -> dict[str, str]:
    out = {}
    for line in row.get("zh_context", "").splitlines():
        m = _CONTEXT_LINE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def check_decision(unit_id: str, value: str, ids: Sequence[str], flags: frozenset[str], where: str,
                    errors: list[str]) -> None:
    allowed = ORPHAN_VALUES if is_orphan(unit_id) else UNIT_VALUES
    if value not in allowed:
        errors.append(f"{where}: {value!r} is not one of {', '.join(allowed)}")
    if value == Relation.NO_COUNTERPART.value and ids:
        errors.append(f"{where}: no_counterpart must name no witness segment")
    if value in {r.value for r in Relation} - {Relation.NO_COUNTERPART.value} and not ids:
        errors.append(f"{where}: relation {value} needs witness segment ids")
    unknown = sorted(flags - set(FLAGS))
    if unknown:
        errors.append(f"{where}: unknown flag(s) {unknown}; allowed: {', '.join(FLAGS)}")


def parse_flip(value: str, where: str, errors: list[str]) -> bool:
    try:
        return parse_bool(value)
    except VerdictError as exc:
        errors.append(f"{where}: {exc}")
        return False


def _import_blind(row: Mapping[str, str], n: int, task: str, plan: Mapping[str, ReviewItem],
                  params: ReviewParams, annotator: str, date: str, ref_lang: str, wit_lang: str,
                  minutes: float | None, errors: list[str], batch: str) -> VerdictRecord | None:
    value = row["blind_relation"].strip()
    if not value:
        return None
    where = f"line {n}"
    if row["task"].strip() != task:
        errors.append(f"{where}: task {row['task']!r} in a {task} import")
    item_id, unit_id = row["item_id"].strip(), row["unit_id"].strip()
    ids, flags = parse_ids(row["blind_wit_loci"]), parse_flags(row["blind_flags"])
    check_decision(unit_id, value, ids, flags, where, errors)
    item = plan.get(item_id)
    if item is None and task != "resolve":
        errors.append(f"{where}: item {item_id} is not in the plan (pass the plan's items)")
    texts = _context_texts(row)
    quote_zh = " ".join(texts.get(i, "") for i in (ids or ((orphan_source(unit_id),) if is_orphan(unit_id) else ())))
    final = task == "resolve"
    return VerdictRecord(
        batch_id=batch, item_id=item_id, task=task, unit_id=unit_id, fingerprint=row["fingerprint"].strip(),
        stratum=item.stratum if item else "", inclusion_prob=item.inclusion_prob if item else None,
        blind_relation=value, blind_wit_ids=ids, final_relation=value if final else "",
        final_wit_ids=ids if final else (), polarity_flip=value == Relation.REVERSAL.value,
        flags=flags, annotator=annotator, blind_date=date, final_date=date if final else "",
        note=row.get("note", "").strip(), quote_ref=truncate_quote(row["ref_text"], ref_lang, params),
        quote_zh=truncate_quote(quote_zh, wit_lang, params), minutes=minutes,
    )


def _import_reveal(rows: Sequence[Mapping[str, str]], blind: Sequence[Verdict], params: ReviewParams, date: str,
                   instrument_digest: str, wit_lang: str, errors: list[str]) -> list[VerdictRecord]:
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
        v = replace(v, machine_relation=machine, machine_status=_machine_status(machine),
                    machine_polarity_flip=row.get("machine_polarity_flip", "").strip() in {"1", "true", "True"},
                    instrument_digest=instrument_digest or v.instrument_digest)
        value = row["final_relation"].strip()
        if value:
            v = _final(v, row, value, params, date, wit_lang, where, errors)
        out[v.item_id] = v
    return list(out.values())


def _final(v: VerdictRecord, row: Mapping[str, str], value: str, params: ReviewParams, date: str, wit_lang: str,
           where: str, errors: list[str]) -> VerdictRecord:
    ids, flags = parse_ids(row["final_wit_loci"]), parse_flags(row["final_flags"])
    check_decision(v.unit_id, value, ids, flags, where, errors)
    reason = row["revised_reason"].strip()
    if (value, ids, flags) != (v.blind_relation, v.blind_wit_ids, v.flags) and not reason:
        errors.append(f"{where}: the final decision differs from the blind one; give revised_reason")
    note = row.get("note", "").strip()
    if reason:
        note = f"{note} | revised: {reason}" if note else f"revised: {reason}"
    texts = _context_texts(row)
    quote_zh = v.quote_zh
    if ids != v.blind_wit_ids and all(i in texts for i in ids):
        quote_zh = truncate_quote(" ".join(texts[i] for i in ids), wit_lang, params)
    return replace(v, final_relation=value, final_wit_ids=ids, flags=flags, final_date=date,
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
