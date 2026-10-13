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
                                                      (``review.reveal_sheets``)
    topics         topics_<batch>.csv                 reference text only, with prelabels
    topics_second  topics_<batch>.second.csv          reference text only, NO prelabel
                                                      columns (critique A6)
                                                      (``review.topic_sheets``)

Witness segments are named by their segment ids everywhere (``zh_context_loci``,
``blind_wit_loci``, ...), separated by spaces. ``zh_context`` shows the witness region
spanned by the machine links of the three reference units on either side of the unit,
never the unit's own link (the unit's counterpart may of course lie inside that region);
each line is "[<segment id>] <text>".

Conventions (collate.v2): blind sheets carry ``blind_convention``, reveal sheets also
``machine_convention`` and ``final_convention``, gold ref sheets ``convention`` (values in
``review.verdicts``). A blank cell imports as ``none``; a cell holding ``?``
(``verdicts.CONVENTION_DEFERRED``: not decided yet) imports as "" (not annotated) and is
refused where no convention can be recorded (orphan rows; no_counterpart, lacuna and
unresolved); a sheet exported before the columns existed imports with "" (not annotated),
but a sheet with only some of them is refused.
Beside every gold, blind and reveal sheet, ``standing_renderings.csv`` lists the verified
standing renderings (the same entries the collate.v2 prompt shows; header only while none is
verified). A human ``standing_rendering`` is accepted only when the import's glossary check
(``GlossaryCheck``, built by the pipeline from the texts and that glossary) confirms it;
without a check every one is refused.

Import (``import_``) keeps decisions and ids and drops every text column; the only text
kept is a short quote per side, cut to ``run.yaml: review.quote_max_chars`` (and never
more than 60 characters), to help re-resolve a verdict when its fingerprint goes stale.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Collection, Mapping, Sequence

from ..core.ids import orphan_source
from ..core.io import write_csv
from ..core.lexicon import StandingRenderings
from ..core.types import Cell, Convention, Link, Relation, Segment, Verdict
from ..matrix.build import segment_fingerprint
from ..topics.prelabel import Prelabel
from .sampling import ReviewItem, ReviewParams
from .verdicts import (
    LINKED_RELATIONS,
    MAX_COMMITTED_QUOTE,
    ORPHAN_VALUES,
    UNIT_VALUES,
    VerdictError,
    VerdictRecord,
    convention_problem,
    deferred_problem,
    flag_problems,
    is_orphan,
    normalise_convention,
    parse_bool,
    parse_flags,
    parse_ids,
)

BLIND_COLUMNS = ("item_id", "task", "unit_id", "fingerprint", "locus", "ref_text", "zh_context_loci",
                 "zh_context", "blind_relation", "blind_wit_loci", "blind_convention", "blind_flags", "note")
REVEAL_COLUMNS = BLIND_COLUMNS + ("machine_relation", "machine_polarity_flip", "machine_convention", "machine_wit_loci",
                                  "machine_quotes", "machine_grade",
                                  "final_relation", "final_wit_loci", "final_convention", "final_flags",
                                  "revised_reason")
# Missing together from sheets exported before collate.v2 (gold ref sheets: ``convention``).
CONVENTION_SHEET_COLUMNS = frozenset({"blind_convention", "machine_convention", "final_convention"})
GLOSSARY_FILE = "standing_renderings.csv"
GLOSSARY_COLUMNS = ("entry", "reference_forms", "witness_renderings")
GLOSSARY_SEPARATOR = " | "
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

# (unit id, linked witness segment ids) -> why the linked text does not support a human
# standing_rendering, or None; built by the pipeline from the texts and the verified glossary.
GlossaryCheck = Callable[[str, tuple[str, ...]], "str | None"]
NO_GLOSSARY = "standing_rendering is accepted only for a verified glossary entry, and no glossary check was given"


# --------------------------------------------------------------------------- export
def export(items: Sequence[ReviewItem], task: str, out_dir: Path, segments: Sequence[Segment],
           cells: Sequence[Cell], *, batch: str | None = None, links: Mapping[str, Link] | None = None,
           blind: Sequence[Verdict] = (), prelabels: Mapping[str, Prelabel] | None = None,
           witness_chapters: Sequence[str] = (), hints: Mapping[str, Sequence[Link]] | None = None,
           glossary: StandingRenderings | None = None) -> list[Path]:
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
    ``glossary``  the verified standing renderings, written beside gold, blind and reveal
                  sheets as ``standing_renderings.csv`` (not written when None)
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
        from .reveal_sheets import export_reveal      # local import: reveal_sheets reuses this module's helpers
        paths = [export_reveal(items, out, batch, index, ref, wit, cells, links or {}, blind)]
    else:
        if any(i.task != task for i in items):
            raise ValueError(f"every item of a {task} sheet must have task {task}")
        rows = [blind_row(i, index, ref, wit, cells) for i in items]
        columns = BLIND_COLUMNS
        if task == "resolve":
            columns = RESOLVE_COLUMNS
            for row in rows:
                row[HINT_COLUMN] = hint_text((hints or {}).get(row["unit_id"], ()))
        path = out / f"{batch}.blind.csv"
        write_csv(path, rows, columns, bom=True)
        paths = [path]
    if glossary is not None:
        paths.append(write_glossary(glossary, out))
    return paths + write_witness_text(wit, out / "witness_text")


def hint_text(links: Sequence[Link]) -> str:
    """``hint_other_model`` cell: each distinct hint as "<relation> [<witness loci>]"."""
    shown = dict.fromkeys(f"{link.relation} [{' '.join(link.wit_ids)}]" for link in links)
    return HINT_LABEL + " | ".join(shown) if shown else ""


def write_glossary(glossary: StandingRenderings, out_dir: Path) -> Path:
    """``standing_renderings.csv``: one row per verified entry, forms joined by " | "."""
    rows = [{"entry": e.id, "reference_forms": GLOSSARY_SEPARATOR.join(e.reference_forms),
             "witness_renderings": GLOSSARY_SEPARATOR.join(e.witness_renderings)} for e in glossary.entries]
    path = out_dir / GLOSSARY_FILE
    write_csv(path, rows, GLOSSARY_COLUMNS, bom=True)
    return path


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


def blind_row(item: ReviewItem, index: Mapping[str, Segment], ref: Sequence[Segment],
              wit: Sequence[Segment], cells: Sequence[Cell]) -> dict[str, str]:
    """The text columns of one blind sheet row; the answer columns are left empty."""
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
            ref_lang: str = "bo", wit_lang: str = "zh", minutes: float | None = None,
            check: GlossaryCheck | None = None) -> list[VerdictRecord]:
    """Read a filled sheet and return committed-format verdicts (no text columns).

    ``gold``      ``path`` is the ``.ref.csv``; the ``.wit.csv`` beside it maps handles to ids
    ``verify``, ``audit``, ``resolve``  the ``.blind.csv``; rows without ``blind_relation``
                  are skipped (a batch may be imported in parts); ``items`` (the plan)
                  supplies stratum and inclusion probability; resolve copies blind to final
    ``reveal``    the ``.reveal.csv``; ``blind`` (the committed blind verdicts of the batch)
                  are returned with their final and machine columns filled
    ``check`` confirms a human ``standing_rendering`` against the verified glossary (None:
    every one is refused). Every problem in the file is reported at once as a ``VerdictError``.
    """
    rows, header = read_sheet(Path(path))
    errors: list[str] = []
    if task == "gold":
        from .gold_sheets import import_gold
        out = import_gold(Path(path), rows, header, params, annotator, date, ref_lang, wit_lang, minutes, errors,
                          check)
    elif task in BLIND_TASKS:
        # resolve sheets carry the hint column (sheets exported before it existed do not)
        expect_columns(header, RESOLVE_COLUMNS if task == "resolve" and HINT_COLUMN in header else BLIND_COLUMNS,
                       path, CONVENTION_SHEET_COLUMNS)
        plan = {i.item_id: i for i in items}
        out = [v for n, row in enumerate(rows, start=2)
               if (v := _import_blind(row, n, task, plan, params, annotator, date, ref_lang, wit_lang,
                                      minutes, errors, batch_of(path, ".blind.csv"), check)) is not None]
    elif task == "reveal":
        from .reveal_sheets import import_reveal
        expect_columns(header, REVEAL_COLUMNS, path, CONVENTION_SHEET_COLUMNS)
        out = import_reveal(rows, blind, params, date, instrument_digest, wit_lang, errors, check)
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


def expect_columns(header: Sequence[str], columns: Sequence[str], path: Path | str,
                   optional: Collection[str] = ()) -> None:
    """The header is ``columns`` (any order), or ``columns`` without ALL of ``optional`` (a
    sheet exported before those columns existed); a sheet with only some of them is refused."""
    if sorted(header) not in (sorted(columns), sorted(c for c in columns if c not in optional)):
        raise VerdictError(f"{path}: expected columns {', '.join(columns)}; found {', '.join(header)}")


def batch_of(path: Path | str, suffix: str) -> str:
    name = Path(path).name
    if not name.endswith(suffix):
        raise VerdictError(f"{path}: expected a file name ending in {suffix}")
    return name[: -len(suffix)]


def context_texts(row: Mapping[str, str]) -> dict[str, str]:
    """Segment id -> text of the ``zh_context`` lines of a sheet row."""
    out = {}
    for line in row.get("zh_context", "").splitlines():
        m = _CONTEXT_LINE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def check_decision(unit_id: str, value: str, ids: Sequence[str], flags: frozenset[str], where: str,
                   errors: list[str], convention: str = "", check: GlossaryCheck | None = None,
                   cell: str | None = None) -> None:
    """Append every problem of one human decision to ``errors``.

    ``convention`` is the normalised cell: "" when the sheet has no convention column or the
    cell holds ``?``; ``cell`` is the raw cell (None without the column), so a ``?`` on a
    decision that takes no convention is refused. A ``standing_rendering`` on a linked
    decision is confirmed by ``check`` (no check: refused).
    """
    allowed = ORPHAN_VALUES if is_orphan(unit_id) else UNIT_VALUES
    if value not in allowed:
        errors.append(f"{where}: {value!r} is not one of {', '.join(allowed)}")
    if value == Relation.NO_COUNTERPART.value and ids:
        errors.append(f"{where}: no_counterpart must name no witness segment")
    if value in LINKED_RELATIONS and not ids:
        errors.append(f"{where}: relation {value} needs witness segment ids")
    errors.extend(f"{where}: {p}" for p in flag_problems(flags))
    problem = convention_problem(unit_id, value, convention) or deferred_problem(unit_id, value, cell)
    if problem:
        errors.append(f"{where}: {problem}")
    elif convention == Convention.STANDING_RENDERING.value and ids:
        problem = check(unit_id, tuple(ids)) if check is not None else NO_GLOSSARY
        if problem:
            errors.append(f"{where}: {problem}")


def parse_flip(value: str, where: str, errors: list[str]) -> bool:
    try:
        return parse_bool(value)
    except VerdictError as exc:
        errors.append(f"{where}: {exc}")
        return False


def _import_blind(row: Mapping[str, str], n: int, task: str, plan: Mapping[str, ReviewItem],
                  params: ReviewParams, annotator: str, date: str, ref_lang: str, wit_lang: str,
                  minutes: float | None, errors: list[str], batch: str,
                  check: GlossaryCheck | None) -> VerdictRecord | None:
    value = row["blind_relation"].strip()
    if not value:
        return None
    where = f"line {n}"
    if row["task"].strip() != task:
        errors.append(f"{where}: task {row['task']!r} in a {task} import")
    item_id, unit_id = row["item_id"].strip(), row["unit_id"].strip()
    ids, flags = parse_ids(row["blind_wit_loci"]), parse_flags(row["blind_flags"])
    cell = row.get("blind_convention")
    conv = "" if cell is None else normalise_convention(cell)
    check_decision(unit_id, value, ids, flags, where, errors, conv, check, cell)
    item = plan.get(item_id)
    if item is None and task != "resolve":
        errors.append(f"{where}: item {item_id} is not in the plan (pass the plan's items)")
    texts = context_texts(row)
    quote_zh = " ".join(texts.get(i, "") for i in (ids or ((orphan_source(unit_id),) if is_orphan(unit_id) else ())))
    final = task == "resolve"
    return VerdictRecord(
        batch_id=batch, item_id=item_id, task=task, unit_id=unit_id, fingerprint=row["fingerprint"].strip(),
        stratum=item.stratum if item else "", inclusion_prob=item.inclusion_prob if item else None,
        blind_relation=value, blind_wit_ids=ids, final_relation=value if final else "",
        final_wit_ids=ids if final else (), blind_convention=conv, final_convention=conv if final else "",
        polarity_flip=value == Relation.REVERSAL.value,
        flags=flags, annotator=annotator, blind_date=date, final_date=date if final else "",
        note=row.get("note", "").strip(), quote_ref=truncate_quote(row["ref_text"], ref_lang, params),
        quote_zh=truncate_quote(quote_zh, wit_lang, params), minutes=minutes,
    )
