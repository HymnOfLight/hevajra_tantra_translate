"""Import and export of alignments as TSV files (critique amendment A4).

``load_tsv`` is the import path for alignments computed outside this package, e.g. by
MITRA-E or DharmaNexus: once converted to our segment ids, an external alignment becomes
an ``Alignment`` and is scored on the same gold windows as every control. There is no
model code here. ``write_tsv`` writes the same format, e.g. to hand a baseline to an
external tool.

Format: UTF-8 (a leading byte-order mark is ignored), tab-separated, one header row,
blank lines skipped. Columns:

    ref_id    reference unit id; empty for a witness-only row
    wit_ids   witness segment ids separated by spaces; empty means no counterpart
    relation  optional column; an empty cell takes the default
                  reference row with witness ids       equivalent
                  reference row without witness ids    no_counterpart
                  witness-only row                     addition
              otherwise a ``Relation`` value (reference rows) or a ``WitnessOnlyKind``
              value (witness-only rows)

Every id must be well formed (``core.ids``), a reference unit may occur in one row only,
and a witness-only row needs witness ids. Other ``Link`` fields (polarity, quotes,
flags, confidence) are not represented, so ``write_tsv`` drops them.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Mapping

from ..core import ids
from ..core.types import Alignment, Link, Relation, WitnessOnlyKind

COLUMNS = ("ref_id", "wit_ids", "relation")
REQUIRED = frozenset({"ref_id", "wit_ids"})


class ExternalAlignmentError(ValueError):
    """A malformed alignment TSV; the message names the file and line."""


def load_tsv(path: Path, reference: str, witness: str, source: str) -> Alignment:
    """Read an alignment TSV; every link gets ``source`` (e.g. "external:mitra-e")."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or not REQUIRED <= set(header) or set(header) - set(COLUMNS):
            raise ExternalAlignmentError(
                f"{path}: header must contain {sorted(REQUIRED)}, optionally 'relation', and nothing else; "
                f"got {header}")
        links = []
        seen: set[str] = set()
        for row in reader:
            where = f"{path}:{reader.line_num}"
            if None in row:
                raise ExternalAlignmentError(f"{where}: more cells than header columns")
            if not any((v or "").strip() for v in row.values()):
                continue
            link = _link(row, source, where)
            if link.ref_id is not None:
                if link.ref_id in seen:
                    raise ExternalAlignmentError(f"{where}: reference unit {link.ref_id} occurs twice")
                seen.add(link.ref_id)
            links.append(link)
    return Alignment(source=source, reference=reference, witness=witness, links=tuple(links))


def _link(row: Mapping[str, str | None], source: str, where: str) -> Link:
    ref_id = (row.get("ref_id") or "").strip() or None
    wit_ids = tuple((row.get("wit_ids") or "").split())
    name = (row.get("relation") or "").strip()
    try:
        if ref_id is not None and ids.id_kind(ref_id) == "orphan":
            raise ValueError(f"{ref_id!r} is a matrix row id, not a reference unit id")
        for wid in wit_ids:
            ids.parse_segment_id(wid)
    except ValueError as exc:
        raise ExternalAlignmentError(f"{where}: {exc}") from exc
    if len(set(wit_ids)) != len(wit_ids):
        raise ExternalAlignmentError(f"{where}: repeated witness id in {wit_ids}")
    if ref_id is None:
        if not wit_ids:
            raise ExternalAlignmentError(f"{where}: a witness-only row needs witness ids")
        kind = _enum(WitnessOnlyKind, name or WitnessOnlyKind.ADDITION.value, where)
        return Link(ref_id=None, wit_ids=wit_ids, relation=kind, source=source)
    default = Relation.EQUIVALENT if wit_ids else Relation.NO_COUNTERPART
    relation = _enum(Relation, name or default.value, where)
    if (relation is Relation.NO_COUNTERPART) != (not wit_ids):
        raise ExternalAlignmentError(
            f"{where}: relation {relation.value} does not fit {'no' if not wit_ids else len(wit_ids)} witness ids")
    return Link(ref_id=ref_id, wit_ids=wit_ids, relation=relation, source=source)


def _enum(cls: type, value: str, where: str):
    try:
        return cls(value)
    except ValueError:
        allowed = ", ".join(m.value for m in cls)
        raise ExternalAlignmentError(f"{where}: unknown relation {value!r}; allowed: {allowed}") from None


def write_tsv(alignment: Alignment, path: Path) -> None:
    """Write ``alignment`` in the format read by ``load_tsv`` (relations written explicitly)."""
    Path(path).write_text(dump_tsv(alignment.links), encoding="utf-8")


def dump_tsv(links: Iterable[Link]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
    writer.writerow(COLUMNS)
    for link in links:
        writer.writerow([link.ref_id or "", " ".join(link.wit_ids), str(link.relation)])
    return buf.getvalue()
