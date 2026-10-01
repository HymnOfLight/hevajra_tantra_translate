"""Sentinel facts: known collation facts checked on every run (synthesis 5.4, 7.3 D).

The facts live in ``data/sentinels/sentinels.yaml`` (schema 2), whose header defines the
eight check kinds, the loci and the range rules; this module implements that header.

Loci are witness line coordinates ("D417:8a.6", "T0892:0592a27"). A locus resolves to the
segments of that text (the id prefix) starting on that line; neighbouring ranges can share
a line, so the stored quote picks the exact segment (``core.textnorm.quote_in``). Without a
matching quote a ``*_from`` / single locus takes the first segment on the line and a
``*_to`` locus the last. Ranges count content segments only (``ingest.CONTENT_KINDS``),
except for the ``paratext`` check, which counts every segment.

Stages are ordered ingest < proposal < final. ``check`` at a stage evaluates every
non-retired sentinel whose own stage is not later: ingest checks read the segments only,
proposal checks read the machine ``Alignment`` (Claude's consensus), final checks read the
matrix ``Cell``s (after verdicts, orphan cells included). Only ``verified`` sentinels
block a gate; ``proposed`` ones are evaluated and reported beside them. A locus that does
not resolve, or a missing input, fails the check (``passed`` False) with the reason in
``detail``: a fact that cannot be checked is not a fact that holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core.io import read_yaml
from ..core.textnorm import quote_in
from ..core.types import Alignment, Cell, Relation, Segment, Status
from ..ingest import CONTENT_KINDS
from ..matrix.status import RELATION_STATUS

SCHEMA_VERSION = 2
STAGES = ("ingest", "proposal", "final")
STATUSES = ("proposed", "verified", "retired")
CHECKS = ("note_kind", "paratext", "adjacent", "relation_in", "status_in", "none_status", "linked_local",
          "witness_only")
_FIELDS = frozenset({"id", "status", "verified_by", "source", "stage", "check", "loci", "quotes", "expect"})


class SentinelError(ValueError):
    """A malformed sentinel file or entry."""


@dataclass(frozen=True)
class Sentinel:
    id: str
    status: str
    stage: str
    check: str
    loci: Mapping[str, str]
    quotes: Mapping[str, str] = field(default_factory=dict)
    expect: Mapping[str, Any] = field(default_factory=dict)
    verified_by: str | None = None
    source: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == "verified"


@dataclass(frozen=True)
class SentinelResult:
    """``stage`` is the stage the check ran at; ``sentinel_stage`` the sentinel's own stage."""

    sentinel_id: str
    check: str
    stage: str
    sentinel_stage: str
    status: str
    blocking: bool
    passed: bool
    detail: str = ""


def load(path: Path) -> list[Sentinel]:
    doc = read_yaml(Path(path)) or {}
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise SentinelError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    out = []
    for n, rec in enumerate(doc.get("sentinels") or []):
        where = f"{path}: sentinels[{n}]"
        if not isinstance(rec, Mapping) or set(rec) != _FIELDS:
            raise SentinelError(f"{where}: fields must be exactly {sorted(_FIELDS)}")
        if rec["status"] not in STATUSES or rec["stage"] not in STAGES or rec["check"] not in CHECKS:
            raise SentinelError(f"{where}: bad status, stage or check ({rec['status']}, {rec['stage']}, {rec['check']})")
        if rec["status"] == "verified" and not rec["verified_by"]:
            raise SentinelError(f"{where}: a verified sentinel names who verified it")
        out.append(Sentinel(id=str(rec["id"]), status=rec["status"], stage=rec["stage"], check=rec["check"],
                            loci=dict(rec["loci"] or {}), quotes=dict(rec["quotes"] or {}),
                            expect=dict(rec["expect"] or {}), verified_by=rec["verified_by"],
                            source=str(rec["source"])))
    ids = [s.id for s in out]
    if len(ids) != len(set(ids)):
        raise SentinelError(f"{path}: duplicate sentinel ids")
    return out


# --------------------------------------------------------------------------- locus resolution
class _Unresolved(Exception):
    pass


class _Text:
    """The segments of one text prefix in document order, indexed by starting line."""

    def __init__(self, segments: Sequence[Segment]):
        self.segments = list(segments)
        self.position = {s.id: i for i, s in enumerate(self.segments)}


def _texts(segments: Iterable[Segment]) -> dict[str, _Text]:
    grouped: dict[str, list[Segment]] = {}
    for seg in segments:
        grouped.setdefault(seg.id.split(":", 1)[0], []).append(seg)
    return {prefix: _Text(segs) for prefix, segs in grouped.items()}


def _resolve(texts: Mapping[str, _Text], sentinel: Sentinel, key: str, kinds: frozenset[str] | None,
             last: bool = False) -> tuple[_Text, Segment]:
    locus = sentinel.loci[key]
    prefix, _, line = locus.partition(":")
    text = texts.get(prefix)
    if text is None:
        raise _Unresolved(f"no segments of {prefix} ({key} {locus})")
    on_line = [s for s in text.segments if s.start == line and (kinds is None or s.kind in kinds)]
    quote = sentinel.quotes.get(key)
    if quote:
        on_line = [s for s in on_line if quote_in(quote, s.text, s.lang)]
    if not on_line:
        raise _Unresolved(f"{key} {locus} does not resolve" + (" with its quote" if quote else ""))
    return text, on_line[-1] if last else on_line[0]


def _range(texts: Mapping[str, _Text], sentinel: Sentinel, side: str,
           kinds: frozenset[str] | None = CONTENT_KINDS) -> list[Segment]:
    text, first = _resolve(texts, sentinel, f"{side}from", kinds)
    other, last = _resolve(texts, sentinel, f"{side}to", kinds, last=True)
    if other is not text:
        raise _Unresolved(f"{side}from and {side}to lie in different texts")
    i, j = text.position[first.id], text.position[last.id]
    if j < i:
        raise _Unresolved(f"{side}to precedes {side}from")
    return [s for s in text.segments[i:j + 1] if kinds is None or s.kind in kinds]


# --------------------------------------------------------------------------- machine / matrix view
@dataclass(frozen=True)
class _Unit:
    relation: str | None           # None: unresolved
    status: str
    polarity_flip: bool
    flags: frozenset[str]
    wit_ids: tuple[str, ...]


class _View:
    """Reference units and witness-only segments as the checked stage sees them."""

    def __init__(self, alignment: Alignment | None, cells: Sequence[Cell] | None):
        self.units: dict[str, _Unit] = {}
        self.witness_only: dict[str, str] = {}
        if cells is not None:
            for c in cells:
                if c.unit_id.startswith("+"):
                    if c.relation is not None:
                        self.witness_only.update({w: c.relation for w in c.wit_ids})
                else:
                    self.units[c.unit_id] = _Unit(c.relation, c.status.value, c.polarity_flip, c.flags, c.wit_ids)
        elif alignment is not None:
            for uid, link in alignment.by_ref().items():
                self.units[uid] = _Unit(str(link.relation), RELATION_STATUS[Relation(link.relation)].value,
                                        link.polarity_flip, link.flags, link.wit_ids)
            self.witness_only = {w: str(link.relation) for link in alignment.witness_only() for w in link.wit_ids}
        self.linked = {w for u in self.units.values() for w in u.wit_ids}

    def unit(self, uid: str) -> _Unit:
        return self.units.get(uid, _Unit(None, Status.UNALIGNED.value, False, frozenset(), ()))


# --------------------------------------------------------------------------- checks
def _note_kind(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    _, note = _resolve(texts, s, "note", frozenset({"note"}))
    got = note.extra.get("note_class")
    return got == s.expect["note_class"], f"{note.id} note_class {got}"


def _paratext(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    span = _range(texts, s, "", kinds=None)
    kinds = [x.kind for x in span]
    text = texts[span[-1].id.split(":", 1)[0]]
    after = [x.id for x in text.segments[text.position[span[-1].id] + 1:] if x.kind in CONTENT_KINDS]
    ok = len(span) == s.expect["count"] and all(k == "paratext" for k in kinds) and not after
    return ok, f"{len(span)} segment(s) of kinds {sorted(set(kinds))}; {len(after)} content segment(s) after"


def _adjacent(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    text, first = _resolve(texts, s, "first", CONTENT_KINDS)
    other, second = _resolve(texts, s, "second", CONTENT_KINDS)
    if other is not text:
        return False, "first and second lie in different texts"
    content = [x.id for x in text.segments if x.kind in CONTENT_KINDS]
    between = content.index(second.id) - content.index(first.id) - 1
    return 0 <= between <= s.expect["max_between"], f"{between} content segment(s) between {first.id} and {second.id}"


def _ref_units(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[list[tuple[str, _Unit]], set[str] | None]:
    units = [(x.id, view.unit(x.id)) for x in _range(texts, s, "ref_")]
    wit = {x.id for x in _range(texts, s, "wit_")} if "wit_from" in s.loci else None
    return units, wit


def _failures(units: list[tuple[str, _Unit]], bad) -> tuple[bool, str]:
    wrong = [f"{uid}={u.relation or u.status}" for uid, u in units if bad(u)]
    return not wrong, f"{len(units)} unit(s); failing: {', '.join(wrong[:5]) or 'none'}"


def _relation_in(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    units, wit = _ref_units(s, texts, view)
    allowed, flags, flip = set(s.expect["relations"]), set(s.expect.get("flags", ())), s.expect.get("polarity_flip")
    return _failures(units, lambda u: u.relation not in allowed or (flip is not None and u.polarity_flip != flip)
                     or not flags <= u.flags or (wit is not None and not wit.issuperset(u.wit_ids)))


def _status_in(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    units, _ = _ref_units(s, texts, view)
    return _failures(units, lambda u: u.status not in set(s.expect["statuses"]))


def _none_status(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    units, _ = _ref_units(s, texts, view)
    return _failures(units, lambda u: u.status == s.expect["status"])


def _linked_local(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    units, wit = _ref_units(s, texts, view)
    local = {seg.id: seg.local_chapter for t in texts.values() for seg in t.segments}
    return _failures(units, lambda u: not u.wit_ids or any(local.get(w) != s.expect["local"] for w in u.wit_ids)
                     or (wit is not None and not wit.issuperset(u.wit_ids)))


def _witness_only(s: Sentinel, texts: Mapping[str, _Text], view: _View) -> tuple[bool, str]:
    segs = _range(texts, s, "wit_")
    kinds = set(s.expect["kinds"])
    wrong = [f"{x.id}={view.witness_only.get(x.id, 'linked' if x.id in view.linked else 'none')}" for x in segs
             if x.id in view.linked or view.witness_only.get(x.id) not in kinds]
    return not wrong, f"{len(segs)} segment(s); failing: {', '.join(wrong[:5]) or 'none'}"


_CHECKERS = {"note_kind": _note_kind, "paratext": _paratext, "adjacent": _adjacent, "relation_in": _relation_in,
             "status_in": _status_in, "none_status": _none_status, "linked_local": _linked_local,
             "witness_only": _witness_only}
_NEEDS_VIEW = frozenset(CHECKS[3:])


def check(sentinels: Sequence[Sentinel], stage: str, segments: Iterable[Segment],
          alignment: Alignment | None = None, cells: Sequence[Cell] | None = None) -> list[SentinelResult]:
    """Evaluate every non-retired sentinel due at ``stage`` (see the module docstring).

    ``segments`` are all segments of the reference and the witness (every kind, document
    order); ``alignment`` is read at stage proposal, ``cells`` at stage final.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    texts = _texts(segments)
    source = {"proposal": alignment, "final": cells}.get(stage)
    view = _View(alignment if stage == "proposal" else None, cells if stage == "final" else None)
    out = []
    for s in sentinels:
        if s.status == "retired" or STAGES.index(s.stage) > STAGES.index(stage):
            continue
        if s.check in _NEEDS_VIEW and source is None:
            passed, detail = False, f"no {'alignment' if stage == 'proposal' else 'cells'} given at stage {stage}"
        else:
            try:
                passed, detail = _CHECKERS[s.check](s, texts, view)
            except _Unresolved as exc:
                passed, detail = False, f"unresolved locus: {exc}"
        out.append(SentinelResult(s.id, s.check, stage, s.stage, s.status, s.blocking, passed, detail))
    return out


def text_prefixes(segments: Iterable[Segment]) -> frozenset[str]:
    """The locus prefixes of a text ("D417", "D418", "T0892"); Sanskrit unit ids have none."""
    return frozenset(s.id.partition(":")[0] for s in segments if ":" in s.id)


def applicable(sentinels: Sequence[Sentinel], reference: frozenset[str], witness: frozenset[str]) -> list[Sentinel]:
    """The sentinels that apply to one (reference, witness) pair, given the locus prefixes of
    each side (``text_prefixes``). A check on the alignment or the matrix applies only when its
    ``ref_*`` loci lie in the reference and its ``wit_*`` loci in the witness: a fact stated
    for Derge -> Chinese says nothing about Sanskrit -> Chinese. Segment-only checks (stage
    ingest kinds) always apply and read every ingested text."""
    def fits(s: Sentinel) -> bool:
        if s.check not in _NEEDS_VIEW:
            return True
        return all(locus.partition(":")[0] in (reference if key.startswith("ref_") else witness)
                   for key, locus in s.loci.items())
    return [s for s in sentinels if fits(s)]


def blocking_failures(results: Iterable[SentinelResult]) -> list[SentinelResult]:
    """Verified sentinels that did not pass (what a gate reads)."""
    return [r for r in results if r.blocking and not r.passed]
