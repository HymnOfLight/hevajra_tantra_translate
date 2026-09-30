"""Witness registry, Sanskrit manuscript list and chapter concordance (``data/registry/``).

    witnesses.yaml       the matrix columns and what is known about each witness
    sa_manuscripts.yaml  Sanskrit manuscripts usable as decomposition columns (not editions)
    concordance.yaml     reference chapter -> witness-local chapter spans (schema version 2)

The concordance is many-to-many: a reference chapter may be spread over several
witness chapters, or be only partly present, and several reference chapters may share
one witness chapter (Chinese chapter 11 holds I.11 and II.1). Each mapping is a
``ChapterSpan`` with a status and the evidence for it, so a reader can see why the
pipeline looks for a passage where it does. The concordance only proposes where to look:
the collator sees the full witness text, and a link outside the mapped chapters is
reported as a relocation instead of being prevented.

Loaders validate strictly (unknown keys and values raise ``RegistryError``) because a typo
in these files silently changes which witness chapters the collator treats as core.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .core.ids import REF_CHAPTERS
from .core.io import read_yaml
from .core.types import Segment

LAYERS = frozenset({"primary", "edition", "indirect", "pivot"})
STATISTICAL_LAYERS = frozenset({"primary", "edition", "indirect"})
INDEPENDENCE = frozenset({"independent", "derived", "revised", "unknown"})
WITNESS_STATUSES = frozenset({"available", "license_restricted", "to_verify", "not_digitized"})
ROLES = frozenset({"gold", "reference_variant", "target", "control", "pivot", "indirect"})
_WITNESS_REQUIRED = frozenset({"id", "lang", "layer", "role", "status", "title"})
_WITNESS_OPTIONAL = frozenset({"source_lang", "independence", "date", "notes", "editor", "translator",
                               "reviser", "source_url", "license", "coords", "unit_scheme"})

MANUSCRIPTS_SCHEMA = 1
MS_INDEPENDENCE = frozenset({"independent", "derived", "to_verify"})
_MS_KEYS = frozenset({"id", "description", "group", "independence", "shelfmark", "source", "access"})

CONCORDANCE_SCHEMA = 2
SPAN_STATUSES = frozenset({"proposed", "verified", "absent_candidate", "absent"})
ABSENT_STATUSES = frozenset({"absent_candidate", "absent"})
_SPAN_KEYS = frozenset({"local", "status", "ref_from", "ref_to", "evidence", "source"})


class RegistryError(ValueError):
    """A registry file is malformed or inconsistent."""


# --------------------------------------------------------------------------- witnesses
@dataclass(frozen=True)
class Witness:
    id: str
    lang: str
    layer: str
    role: str
    status: str
    title: str
    source_lang: str | None = None     # language translated from; None for Sanskrit witnesses
    independence: str = "unknown"
    date: str = ""
    notes: str = ""
    info: Mapping[str, str] = field(default_factory=dict)   # editor, translator, licence, URL, ...

    @property
    def counts_in_statistics(self) -> bool:
        """Pivot witnesses (modern translations) are bridges for humans, never data."""
        return self.layer in STATISTICAL_LAYERS


def load_witnesses(path: Path) -> dict[str, Witness]:
    """Witness id -> ``Witness`` from ``witnesses.yaml`` (replication sets are not loaded)."""
    doc = _mapping(read_yaml(path), str(path))
    _keys(doc, {"witnesses"}, {"witnesses", "external_replication_sets"}, str(path))
    out: dict[str, Witness] = {}
    for i, rec in enumerate(_list(doc["witnesses"], f"{path}: witnesses")):
        where = f"{path}: witnesses[{i}]"
        _keys(rec, _WITNESS_REQUIRED, _WITNESS_REQUIRED | _WITNESS_OPTIONAL, where)
        for key, allowed in (("layer", LAYERS), ("role", ROLES), ("status", WITNESS_STATUSES),
                             ("independence", INDEPENDENCE)):
            if key in rec and rec[key] not in allowed:
                raise RegistryError(f"{where}: {key} {rec[key]!r} not in {sorted(allowed)}")
        if rec["id"] in out:
            raise RegistryError(f"{where}: duplicate id")
        info = {k: str(rec[k]).strip() for k in sorted(_WITNESS_OPTIONAL - {"source_lang", "independence",
                                                                             "date", "notes"}) if k in rec}
        out[rec["id"]] = Witness(
            id=rec["id"], lang=rec["lang"], layer=rec["layer"], role=rec["role"], status=rec["status"],
            title=str(rec["title"]).strip(), source_lang=rec.get("source_lang"),
            independence=rec.get("independence", "unknown"), date=str(rec.get("date", "")).strip(),
            notes=str(rec.get("notes", "")).strip(), info=MappingProxyType(info),
        )
    return out


# --------------------------------------------------------------------------- manuscripts
@dataclass(frozen=True)
class Manuscript:
    """A Sanskrit manuscript; unknown facts are "to_verify", never guessed."""

    id: str
    description: str
    group: str
    independence: str
    shelfmark: str
    source: str
    access: str


def load_manuscripts(path: Path) -> dict[str, Manuscript]:
    """Manuscript id -> ``Manuscript`` from ``sa_manuscripts.yaml`` (schema version 1)."""
    doc = _mapping(read_yaml(path), str(path))
    _keys(doc, {"schema_version", "manuscripts"}, {"schema_version", "manuscripts"}, str(path))
    if doc["schema_version"] != MANUSCRIPTS_SCHEMA:
        raise RegistryError(f"{path}: schema_version must be {MANUSCRIPTS_SCHEMA}")
    out: dict[str, Manuscript] = {}
    for i, rec in enumerate(_list(doc["manuscripts"], f"{path}: manuscripts")):
        where = f"{path}: manuscripts[{i}]"
        _keys(rec, _MS_KEYS, _MS_KEYS, where)
        if rec["independence"] not in MS_INDEPENDENCE:
            raise RegistryError(f"{where}: independence {rec['independence']!r} not in {sorted(MS_INDEPENDENCE)}")
        if rec["id"] in out:
            raise RegistryError(f"{where}: duplicate id")
        out[rec["id"]] = Manuscript(**{k: str(rec[k]).strip() for k in _MS_KEYS})
    return out


# --------------------------------------------------------------------------- concordance
@dataclass(frozen=True)
class ChapterSpan:
    """Where (part of) one reference chapter lies in one witness.

    ``local``      witness-local chapter key ("pin18", "D418:9"); None when the span is absent
    ``status``     proposed (titles, order, length; not checked unit by unit), verified
                   (checked in the source texts; ``source`` names by whom), absent_candidate
                   (no counterpart found yet), absent (absence verified)
    ``ref_from``, ``ref_to``  inclusive reference-witness coordinates ("D418:27b.1") when the
                   span covers only part of the chapter; None for the whole chapter. Line
                   coordinates are coarse: adjacent spans may share their boundary line.
    ``evidence``   why the mapping holds (notes, parallel passages, sentinels)
    ``source``     who established it
    """

    local: str | None
    status: str
    ref_from: str | None = None
    ref_to: str | None = None
    evidence: tuple[str, ...] = ()
    source: str = ""

    @property
    def is_absent(self) -> bool:
        return self.status in ABSENT_STATUSES


@dataclass(frozen=True)
class Concordance:
    reference_scheme: str
    local_order: Mapping[str, tuple[str, ...]]                            # witness -> local keys
    spans: Mapping[str, Mapping[str, tuple[ChapterSpan, ...]]]           # ref chapter -> witness -> spans

    def witnesses(self) -> tuple[str, ...]:
        return tuple(self.local_order)

    def locals_for(self, ref_chapter: str, witness: str) -> list[ChapterSpan]:
        """All spans of ``ref_chapter`` in ``witness``, absent ones included, in file order."""
        self._check(witness=witness, ref_chapter=ref_chapter)
        return list(self.spans[ref_chapter][witness])

    def refs_for_local(self, witness: str, local: str) -> list[str]:
        """Reference chapters (in reference order) with a span in witness chapter ``local``."""
        self._check(witness=witness, local=local)
        return [ref for ref in REF_CHAPTERS if any(s.local == local for s in self.spans[ref][witness])]

    def core_window(self, ref_chapter: str, witness: str, neighbours: int = 1) -> frozenset[str]:
        """Witness chapters mapped to ``ref_chapter`` plus ``neighbours`` on each side.

        When the chapter is mapped to nothing (all spans absent), the chapters mapped to
        the nearest mapped reference chapters before and after it are used instead, so
        the window still covers the place where the material would stand.
        """
        if neighbours < 0:
            raise ValueError(f"neighbours must be >= 0, got {neighbours}")
        self._check(witness=witness, ref_chapter=ref_chapter)
        order = self.local_order[witness]
        mapped = self._mapped(ref_chapter, witness)
        if not mapped:
            i = REF_CHAPTERS.index(ref_chapter)
            before = (self._mapped(r, witness) for r in reversed(REF_CHAPTERS[:i]))
            after = (self._mapped(r, witness) for r in REF_CHAPTERS[i + 1:])
            mapped = next((m for m in before if m), set()) | next((m for m in after if m), set())
        positions = {order.index(local) for local in mapped}
        return frozenset(order[j] for i in positions
                         for j in range(max(i - neighbours, 0), min(i + neighbours + 1, len(order))))

    def assign_reference_chapters(self, segments: Iterable[Segment], witness: str) -> list[Segment]:
        """Set ``Segment.chapter`` where the segment's local chapter maps to exactly one
        reference chapter (e.g. every Derge chapter: D417:n -> I.n, D418:n -> II.n).

        Segments without a local chapter (front matter, paratext) and segments of merged
        witness chapters (Chinese pin11 = I.11 + II.1) are returned unchanged: their
        reference chapter is decided by the alignment, not by the concordance.
        """
        out = []
        for seg in segments:
            refs = self.refs_for_local(witness, seg.local_chapter) if seg.local_chapter else []
            out.append(replace(seg, chapter=refs[0]) if len(refs) == 1 else seg)
        return out

    def _mapped(self, ref_chapter: str, witness: str) -> set[str]:
        return {s.local for s in self.spans[ref_chapter][witness] if s.local is not None}

    def _check(self, witness: str, ref_chapter: str | None = None, local: str | None = None) -> None:
        if witness not in self.local_order:
            raise KeyError(f"witness {witness!r} is not in the concordance; known: {list(self.local_order)}")
        if ref_chapter is not None and ref_chapter not in self.spans:
            raise KeyError(f"unknown reference chapter {ref_chapter!r}")
        if local is not None and local not in self.local_order[witness]:
            raise KeyError(f"{witness} has no chapter {local!r}")


def load_concordance(path: Path) -> Concordance:
    """The chapter concordance (schema version 2); every reference chapter must have spans
    for every declared witness, and every span evidence and a source."""
    doc = _mapping(read_yaml(path), str(path))
    top = {"schema_version", "reference_scheme", "witnesses", "chapters"}
    _keys(doc, top, top, str(path))
    if doc["schema_version"] != CONCORDANCE_SCHEMA:
        raise RegistryError(f"{path}: schema_version must be {CONCORDANCE_SCHEMA}")
    order: dict[str, tuple[str, ...]] = {}
    for witness, rec in _mapping(doc["witnesses"], f"{path}: witnesses").items():
        _keys(_mapping(rec, f"{path}: {witness}"), {"locals"}, {"locals"}, f"{path}: {witness}")
        locals_ = tuple(str(x) for x in _list(rec["locals"], f"{path}: {witness}.locals"))
        if not locals_ or len(set(locals_)) != len(locals_):
            raise RegistryError(f"{path}: {witness}.locals must be a non-empty list of distinct keys")
        order[witness] = locals_
    spans: dict[str, Mapping[str, tuple[ChapterSpan, ...]]] = {}
    for i, rec in enumerate(_list(doc["chapters"], f"{path}: chapters")):
        where = f"{path}: chapters[{i}]"
        _keys(rec, {"ref", "spans"}, {"ref", "sa_title", "spans"}, where)
        ref = rec["ref"]
        if ref not in REF_CHAPTERS or ref in spans:
            raise RegistryError(f"{where}: unknown or repeated reference chapter")
        by_witness = _mapping(rec["spans"], where)
        if set(by_witness) != set(order):
            raise RegistryError(f"{where}: spans must be given for exactly the witnesses {sorted(order)}")
        spans[ref] = MappingProxyType({
            w: tuple(_span(s, order[w], f"{where}, {w}") for s in _list(items, f"{where}, {w}"))
            for w, items in by_witness.items()
        })
    missing = [r for r in REF_CHAPTERS if r not in spans]
    if missing:
        raise RegistryError(f"{path}: reference chapters without an entry: {missing}")
    return Concordance(reference_scheme=str(doc["reference_scheme"]), local_order=MappingProxyType(order),
                       spans=MappingProxyType({r: spans[r] for r in REF_CHAPTERS}))


def _span(rec: Any, locals_: tuple[str, ...], where: str) -> ChapterSpan:
    rec = _mapping(rec, where)
    _keys(rec, {"local", "status", "evidence", "source"}, _SPAN_KEYS, where)
    status, local = rec["status"], rec["local"]
    if status not in SPAN_STATUSES:
        raise RegistryError(f"{where}: status {status!r} not in {sorted(SPAN_STATUSES)}")
    if (local is None) != (status in ABSENT_STATUSES):
        raise RegistryError(f"{where}: local must be null exactly when the status is absent")
    if local is not None and local not in locals_:
        raise RegistryError(f"{where}: local {local!r} is not declared for this witness")
    ref_from, ref_to = rec.get("ref_from"), rec.get("ref_to")
    if (ref_from is None) != (ref_to is None):
        raise RegistryError(f"{where}: give both ref_from and ref_to, or neither")
    evidence = tuple(str(e).strip() for e in _list(rec["evidence"], f"{where}: evidence"))
    source = str(rec["source"] or "").strip()
    if not evidence or not all(evidence) or not source:
        raise RegistryError(f"{where}: every span needs evidence and a source")
    return ChapterSpan(local=local, status=status, ref_from=ref_from, ref_to=ref_to, evidence=evidence,
                       source=source)


# --------------------------------------------------------------------------- helpers
def _mapping(node: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise RegistryError(f"{where}: expected a mapping")
    return node


def _list(node: Any, where: str) -> list[Any]:
    if not isinstance(node, list):
        raise RegistryError(f"{where}: expected a list")
    return node


def _keys(rec: Any, required: set[str] | frozenset[str], allowed: set[str] | frozenset[str], where: str) -> None:
    rec = _mapping(rec, where)
    missing, unknown = sorted(set(required) - set(rec)), sorted(set(rec) - set(allowed))
    if missing or unknown:
        raise RegistryError(f"{where}: missing keys {missing}, unknown keys {unknown}")
