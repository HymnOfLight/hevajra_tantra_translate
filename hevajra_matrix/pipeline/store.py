"""JSON forms of the records that stages hand to each other through the run directory.

Stages communicate through files only (synthesis 2), so every record a later stage reads
back (segments, alignments, verified collations, cells, estimates, gate reports) has a
plain JSON form here. The functions are pure pairs ``*_to_dict`` / ``*_from_dict`` whose
round trip is the identity on the frozen dataclasses; the file helpers at the bottom
only combine them with ``core.io``.

Run-directory files written with these helpers contain source text (segments, quotes)
and live under ``runs/``, which is never committed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..collate.verify import Collation
from ..core.io import read_jsonl, write_jsonl
from ..core.types import (
    Alignment,
    Cell,
    Diagnostic,
    Estimate,
    Grade,
    Link,
    Quote,
    Relation,
    Segment,
    Status,
    WitnessOnlyKind,
)
from ..evaluation.gate import GateReport
from ..evaluation.sentinels import SentinelResult


# --------------------------------------------------------------------------- segments
def segment_to_dict(s: Segment) -> dict[str, Any]:
    return {"id": s.id, "witness": s.witness, "lang": s.lang, "text": s.text, "start": s.start, "end": s.end,
            "kind": s.kind, "local_chapter": s.local_chapter, "chapter": s.chapter, "fingerprint": s.fingerprint,
            "extra": dict(s.extra)}


def segment_from_dict(d: Mapping[str, Any]) -> Segment:
    return Segment(id=d["id"], witness=d["witness"], lang=d["lang"], text=d["text"], start=d["start"], end=d["end"],
                   kind=d["kind"], local_chapter=d.get("local_chapter"), chapter=d.get("chapter"),
                   fingerprint=d.get("fingerprint", ""), extra=dict(d.get("extra") or {}))


# --------------------------------------------------------------------------- links and alignments
def link_to_dict(link: Link) -> dict[str, Any]:
    return {"ref_id": link.ref_id, "wit_ids": list(link.wit_ids), "relation": str(link.relation),
            "polarity_flip": link.polarity_flip, "confidence": link.confidence,
            "quotes": [{"side": q.side, "text": q.text, "segment_ids": list(q.segment_ids)} for q in link.quotes],
            "flags": sorted(link.flags), "source": link.source}


def link_from_dict(d: Mapping[str, Any]) -> Link:
    ref_id = d.get("ref_id")
    relation = Relation(d["relation"]) if ref_id is not None else WitnessOnlyKind(d["relation"])
    quotes = tuple(Quote(q["side"], q["text"], tuple(q.get("segment_ids") or ())) for q in d.get("quotes") or ())
    return Link(ref_id=ref_id, wit_ids=tuple(d.get("wit_ids") or ()), relation=relation,
                polarity_flip=bool(d.get("polarity_flip", False)), confidence=d.get("confidence"), quotes=quotes,
                flags=frozenset(d.get("flags") or ()), source=d.get("source", ""))


def alignment_records(a: Alignment) -> list[dict[str, Any]]:
    """A header record (source, reference, witness) followed by one record per link."""
    return [{"source": a.source, "reference": a.reference, "witness": a.witness},
            *(link_to_dict(link) for link in a.links)]


def alignment_from_records(records: list[Mapping[str, Any]]) -> Alignment:
    if not records or "source" not in records[0] or "wit_ids" in records[0]:
        raise ValueError("an alignment file starts with a header record (source, reference, witness)")
    head = records[0]
    return Alignment(head["source"], head["reference"], head["witness"],
                     tuple(link_from_dict(r) for r in records[1:]))


# --------------------------------------------------------------------------- collations
def diagnostic_to_dict(d: Diagnostic) -> dict[str, Any]:
    return {"kind": d.kind, "ref_ids": list(d.ref_ids), "wit_ids": list(d.wit_ids), "detail": d.detail}


def diagnostic_from_dict(d: Mapping[str, Any]) -> Diagnostic:
    return Diagnostic(d["kind"], tuple(d.get("ref_ids") or ()), tuple(d.get("wit_ids") or ()), d.get("detail", ""))


def collation_to_dict(c: Collation) -> dict[str, Any]:
    """Everything of a verified replicate except its alignment (stored as its own file)."""
    return {"unresolved": dict(c.unresolved), "diagnostics": [diagnostic_to_dict(d) for d in c.diagnostics],
            "hints": [link_to_dict(link) for link in c.hints]}


def collation_from_dict(alignment: Alignment, d: Mapping[str, Any]) -> Collation:
    return Collation(alignment=alignment, unresolved=MappingProxyType(dict(d.get("unresolved") or {})),
                     diagnostics=tuple(diagnostic_from_dict(x) for x in d.get("diagnostics") or ()),
                     hints=tuple(link_from_dict(x) for x in d.get("hints") or ()))


# --------------------------------------------------------------------------- cells
def cell_to_dict(c: Cell) -> dict[str, Any]:
    return {"unit_id": c.unit_id, "witness": c.witness, "status": c.status.value, "grade": c.grade.value,
            "chapter": c.chapter, "relation": c.relation, "polarity_flip": c.polarity_flip, "reason": c.reason,
            "wit_ids": list(c.wit_ids), "flags": sorted(c.flags), "d_len": c.d_len, "d_ord": c.d_ord,
            "d_lit": c.d_lit, "source": c.source}


def cell_from_dict(d: Mapping[str, Any]) -> Cell:
    return Cell(unit_id=d["unit_id"], witness=d["witness"], status=Status(d["status"]), grade=Grade(d["grade"]),
                chapter=d.get("chapter", ""), relation=d.get("relation"), polarity_flip=bool(d.get("polarity_flip")),
                reason=d.get("reason"), wit_ids=tuple(d.get("wit_ids") or ()), flags=frozenset(d.get("flags") or ()),
                d_len=d.get("d_len"), d_ord=d.get("d_ord"), d_lit=d.get("d_lit"), source=d.get("source", ""))


# --------------------------------------------------------------------------- estimates, gates, sentinels
def estimate_to_dict(e: Estimate) -> dict[str, Any]:
    out = asdict(e)
    out["sources"] = list(e.sources)
    return out


def estimate_from_dict(d: Mapping[str, Any]) -> Estimate:
    return Estimate(name=d["name"], point=d.get("point"), lo=d.get("lo"), hi=d.get("hi"), n=int(d.get("n", 0)),
                    scope=d.get("scope", ""), sources=tuple(d.get("sources") or ()), level=int(d.get("level", 0)),
                    not_estimable=d.get("not_estimable"))


def gate_to_dict(g: GateReport) -> dict[str, Any]:
    return {"level": g.level, "confirmatory": g.confirmatory, "passed": dict(g.passed), "reasons": list(g.reasons),
            "deferred_null": g.deferred_null, "not_estimable": dict(g.not_estimable), "scope_note": g.scope_note}


def gate_from_dict(d: Mapping[str, Any]) -> GateReport:
    return GateReport(level=int(d["level"]), confirmatory=bool(d["confirmatory"]),
                      passed=MappingProxyType(dict(d.get("passed") or {})), reasons=tuple(d.get("reasons") or ()),
                      deferred_null=bool(d.get("deferred_null", False)),
                      not_estimable=MappingProxyType(dict(d.get("not_estimable") or {})),
                      scope_note=d.get("scope_note", ""))


def sentinel_result_to_dict(r: SentinelResult) -> dict[str, Any]:
    return asdict(r)


def sentinel_result_from_dict(d: Mapping[str, Any]) -> SentinelResult:
    return SentinelResult(**{k: d[k] for k in ("sentinel_id", "check", "stage", "sentinel_stage", "status",
                                               "blocking", "passed")}, detail=d.get("detail", ""))


# --------------------------------------------------------------------------- files
def write_segments(path: Path, segments: Iterable[Segment]) -> None:
    write_jsonl(path, (segment_to_dict(s) for s in segments))


def read_segments(path: Path) -> list[Segment]:
    return [segment_from_dict(r) for r in read_jsonl(path)]


def write_alignment(path: Path, alignment: Alignment) -> None:
    write_jsonl(path, alignment_records(alignment))


def read_alignment(path: Path) -> Alignment:
    return alignment_from_records(read_jsonl(path))


def write_cells(path: Path, cells: Iterable[Cell]) -> None:
    write_jsonl(path, (cell_to_dict(c) for c in cells))


def read_cells(path: Path) -> list[Cell]:
    return [cell_from_dict(r) for r in read_jsonl(path)]
