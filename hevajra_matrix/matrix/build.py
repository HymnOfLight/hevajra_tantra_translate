"""Matrix cells for one (reference, witness) pair: precedence, grades and descriptive dimensions.

``build_cells`` is a pure function run on every ``build`` (so human verdicts always flow
back into the matrix; fixes review defect #2).

Precedence per reference unit (synthesis 4.4)
    1. gold (dev or test) whose fingerprint matches the unit's current text   grade A
    2. otherwise the latest final human verdict with a matching fingerprint     grade A
    3. otherwise the Claude consensus link                                      its grade
    4. otherwise UNALIGNED with the consensus reason ("unassessed" if none)     grade X
    Baselines never populate the matrix: callers pass the consensus alignment only.
    A gold or verdict row whose fingerprint no longer matches (or whose unit vanished) is
    never applied; it is returned as a ``StaleVerdict`` with its short quotes as a
    re-resolution hint.

Status comes from the relation alone (``matrix.status``); length never enters. The human
decision values ``lacuna`` and ``unresolved`` give LACUNA and UNALIGNED("unresolved_by_human").

Descriptive dimensions (synthesis 4.5; never inputs to status)
    d_len  log(witness length / reference length) of the linked segments (``textnorm.length``)
    d_lit  transliteration density of the linked witness text (``core.translit``)
    d_ord  signed rank displacement of the unit's first linked witness segment within its
           chapter GROUP (the concordance group: e.g. Chinese pin11 = I.11 + II.1), computed
           once per group over the group's linked units (fixes defect #6, which computed
           positions per reference chapter inside a merged witness chapter). 0 for every unit
           of an in-order alignment, whatever is omitted or added; None when the first linked
           segment lies outside the group (a relocation) or the unit is in no group.

Orphan rows (witness-only material; E5)
    One row per witness segment, id ``+<segment id>`` (unique: two orphans on one line give
    two rows; fixes #14), status NA, relation = the ``WitnessOnlyKind``. Claims come from the
    consensus and from human verdicts on orphan rows; a segment that some reference cell
    links is never an orphan, and ``has_counterpart`` removes a claim. Each row is attributed
    to a reference chapter (critique B15): the group's only chapter, else the chapter of the
    nearest linked witness segment in the group (the preceding one on a tie), else of the
    nearest linked segment anywhere, else "".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Collection, Iterable, Mapping, Sequence

from ..core.ids import orphan_row_id, orphan_source
from ..core.textnorm import fingerprint, length
from ..core.translit import transliteration_charset, transliteration_density
from ..core.types import (
    REASON_UNASSESSED,
    Alignment,
    Cell,
    Grade,
    Link,
    Relation,
    Segment,
    Status,
    Verdict,
    WitnessOnlyKind,
)
from ..review.verdicts import HAS_COUNTERPART, LACUNA, UNRESOLVED, decision, decision_date, is_orphan
from .status import RELATION_STATUS

ChapterGroup = tuple[Collection[str], Collection[str]]   # (reference chapters, witness local chapters)

REASON_HUMAN_UNRESOLVED = "unresolved_by_human"
STALE_FINGERPRINT = "fingerprint_changed"
STALE_MISSING = "unit_missing"
CONTROL_SOURCE_PREFIXES = ("dp:", "placebo:")    # baselines and placebos: never matrix input


@dataclass(frozen=True)
class StaleVerdict:
    """A gold row or verdict that was not applied because its unit's text changed or vanished."""

    unit_id: str
    batch_id: str
    item_id: str
    task: str
    stored_fingerprint: str
    current_fingerprint: str
    reason: str
    quote_ref: str = ""
    quote_zh: str = ""


@dataclass(frozen=True)
class _Human:
    """The human decision applied to one row."""

    value: str
    wit_ids: tuple[str, ...]
    polarity_flip: bool
    flags: frozenset[str]
    source: str


def segment_fingerprint(seg: Segment) -> str:
    """The stored fingerprint, or one computed from the text when the ingester left it empty."""
    return seg.fingerprint or fingerprint(seg.text, seg.lang)


# --------------------------------------------------------------------------- entry point
def build_cells(
    reference_segments: Sequence[Segment],
    consensus: Alignment,
    grades: Mapping[str, Grade],
    reasons: Mapping[str, str],
    gold: Iterable[Verdict],
    verdicts: Iterable[Verdict],
    witness_segments: Sequence[Segment],
    groups: Sequence[ChapterGroup],
) -> tuple[list[Cell], list[Cell], list[StaleVerdict]]:
    """(reference cells in reference order, orphan cells in witness order, stale verdicts).

    ``reference_segments``  the reference units (matrix rows), in reference order
    ``consensus``, ``grades``, ``reasons``  the output of ``collate.consensus``
    ``gold``                blind gold verdicts (task gold) of the primary annotator, not the
                            second annotator's (those only measure human-human agreement)
    ``verdicts``            committed review verdicts (verify, audit, resolve) of THIS witness
                            (``review.verdicts.load(verdict_dir / witness)``); a verdict whose
                            row is unknown here is listed as stale ("unit_missing")
    ``witness_segments``    the whole witness text in document order (all kinds)
    ``groups``              chapter groups from the concordance, as for ``align.dp.align_groups``
    """
    if consensus.source.startswith(CONTROL_SOURCE_PREFIXES):
        raise ValueError(f"{consensus.source} is a control alignment; baselines never populate the matrix")
    _check_groups(groups)
    ref_fp = {s.id: segment_fingerprint(s) for s in reference_segments}
    wit_by_id = {s.id: s for s in witness_segments}
    row_fp = {**ref_fp, **{orphan_row_id(s.id): segment_fingerprint(s) for s in witness_segments}}
    gold_rows, stale_gold = _current(gold, row_fp)
    review_rows, stale_review = _current(verdicts, row_fp)
    human = {**_decisions(review_rows, "verdict"), **_decisions(gold_rows, "gold")}   # gold wins
    witness = consensus.witness
    links = consensus.by_ref()
    charset = transliteration_charset(witness_segments)
    cells = [_unit_cell(seg, witness, human.get(seg.id), links.get(seg.id), grades, reasons, wit_by_id, charset,
                        consensus.source) for seg in reference_segments]
    displacement = group_displacements(reference_segments, witness_segments, cells, groups)
    cells = [replace(c, d_ord=displacement.get(c.unit_id)) for c in cells]
    orphans = _orphan_cells(consensus.witness_only(), human, grades, cells, witness, witness_segments,
                            groups, charset)
    return cells, orphans, stale_gold + stale_review


# --------------------------------------------------------------------------- human rows
def _current(verdicts: Iterable[Verdict], row_fp: Mapping[str, str]) -> tuple[list[Verdict], list[StaleVerdict]]:
    """Split verdicts into those whose fingerprint matches the current text and the stale rest."""
    current: list[Verdict] = []
    stale: list[StaleVerdict] = []
    for v in verdicts:
        now = row_fp.get(v.unit_id)
        if now == v.fingerprint:
            current.append(v)
            continue
        stale.append(StaleVerdict(
            unit_id=v.unit_id, batch_id=v.batch_id, item_id=v.item_id, task=v.task,
            stored_fingerprint=v.fingerprint, current_fingerprint=now or "",
            reason=STALE_MISSING if now is None else STALE_FINGERPRINT,
            quote_ref=getattr(v, "quote_ref", ""), quote_zh=getattr(v, "quote_zh", ""),
        ))
    return current, stale


def _decisions(verdicts: Sequence[Verdict], kind: str) -> dict[str, _Human]:
    """Per row id, the latest decided verdict (by decision date, then input order)."""
    best: dict[str, tuple[str, int, Verdict]] = {}
    for i, v in enumerate(verdicts):
        if decision(v) is None:
            continue
        rank = (decision_date(v), i)
        if v.unit_id not in best or rank >= best[v.unit_id][:2]:
            best[v.unit_id] = (*rank, v)
    out = {}
    for uid, (_, _, v) in best.items():
        value, wit_ids = decision(v)  # type: ignore[misc]
        out[uid] = _Human(value, tuple(wit_ids), v.polarity_flip, v.flags, f"{kind}:{v.batch_id}")
    return out


# --------------------------------------------------------------------------- unit cells
def _unit_cell(seg: Segment, witness: str, human: _Human | None, link: Link | None, grades: Mapping[str, Grade],
               reasons: Mapping[str, str], wit_by_id: Mapping[str, Segment], charset: frozenset[str],
               source: str) -> Cell:
    chapter = seg.chapter or ""
    if human is not None:
        return _human_cell(seg, witness, chapter, human, wit_by_id, charset)
    if link is not None:
        if not isinstance(link.relation, Relation):
            raise ValueError(f"{seg.id}: consensus link carries a witness-only kind {link.relation}")
        if seg.id not in grades:
            raise ValueError(f"{seg.id}: consensus link without a grade")
        return Cell(unit_id=seg.id, witness=witness, status=RELATION_STATUS[link.relation], grade=grades[seg.id],
                    chapter=chapter, relation=link.relation.value, polarity_flip=link.polarity_flip,
                    wit_ids=link.wit_ids, flags=link.flags, source=link.source,
                    **_dims(seg, link.wit_ids, wit_by_id, charset))
    return Cell(unit_id=seg.id, witness=witness, status=Status.UNALIGNED, grade=Grade.X, chapter=chapter,
                reason=reasons.get(seg.id) or REASON_UNASSESSED, source=source)


def _human_cell(seg: Segment, witness: str, chapter: str, h: _Human, wit_by_id: Mapping[str, Segment],
                charset: frozenset[str]) -> Cell:
    common = dict(unit_id=seg.id, witness=witness, grade=Grade.A, chapter=chapter, flags=h.flags, source=h.source)
    if h.value == LACUNA:
        return Cell(status=Status.LACUNA, **common)  # type: ignore[arg-type]
    if h.value == UNRESOLVED:
        return Cell(status=Status.UNALIGNED, reason=REASON_HUMAN_UNRESOLVED, **common)  # type: ignore[arg-type]
    try:
        relation = Relation(h.value)
    except ValueError:
        raise ValueError(f"{seg.id}: human decision {h.value!r} is not a relation, lacuna or unresolved "
                         f"(from {h.source})") from None
    return Cell(status=RELATION_STATUS[relation], relation=relation.value, polarity_flip=h.polarity_flip,
                wit_ids=h.wit_ids, **common, **_dims(seg, h.wit_ids, wit_by_id, charset))  # type: ignore[arg-type]


def _dims(seg: Segment, wit_ids: Sequence[str], wit_by_id: Mapping[str, Segment],
          charset: frozenset[str]) -> dict[str, float | None]:
    missing = [i for i in wit_ids if i not in wit_by_id]
    if missing:
        raise ValueError(f"{seg.id}: linked witness segment(s) not in the witness text: {missing}")
    linked = [wit_by_id[i] for i in wit_ids]
    return {"d_len": length_ratio(seg, linked), "d_lit": literal_density(linked, charset)}


def length_ratio(ref: Segment, linked: Sequence[Segment]) -> float | None:
    """log(witness length / reference length); None without a counterpart or a zero length."""
    r = length(ref.text, ref.lang)
    w = sum(length(s.text, s.lang) for s in linked)
    return math.log(w / r) if r > 0 and w > 0 else None


def literal_density(linked: Sequence[Segment], charset: frozenset[str]) -> float | None:
    """Transliteration density of the linked witness text; None without a counterpart."""
    if not linked:
        return None
    return transliteration_density(" ".join(s.text for s in linked), linked[0].lang, charset)


# --------------------------------------------------------------------------- d_ord
def order_displacement(unit_ids: Sequence[str], first_position: Mapping[str, int]) -> dict[str, float]:
    """Signed rank displacement in [-1, 1] of each unit that has a first linked position.

    The linked units are ranked once in reference order and once by the witness position of
    their first linked segment (ties keep reference order, so n:1 links do not count as
    disorder). d = (witness rank - reference rank) / (n - 1).
    """
    linked = [u for u in unit_ids if u in first_position]
    n = len(linked)
    if n < 2:
        return {u: 0.0 for u in linked}
    by_witness = sorted(range(n), key=lambda i: (first_position[linked[i]], i))
    witness_rank = {linked[i]: rank for rank, i in enumerate(by_witness)}
    return {u: (witness_rank[u] - i) / (n - 1) for i, u in enumerate(linked)}


def group_displacements(reference_segments: Sequence[Segment], witness_segments: Sequence[Segment],
                        cells: Sequence[Cell], groups: Sequence[ChapterGroup]) -> dict[str, float]:
    """d_ord for every unit whose first linked segment lies in the unit's own chapter group."""
    wit_ids_of = {c.unit_id: c.wit_ids for c in cells}
    out: dict[str, float] = {}
    for ref_keys, wit_keys in groups:
        ref_keys, wit_keys = frozenset(ref_keys), frozenset(wit_keys)
        units = [s.id for s in reference_segments if s.chapter in ref_keys]
        position = {s.id: i for i, s in enumerate(s for s in witness_segments if s.local_chapter in wit_keys)}
        first = {}
        for uid in units:
            inside = [position[w] for w in wit_ids_of.get(uid, ()) if w in position]
            if inside:
                first[uid] = min(inside)
        out.update(order_displacement(units, first))
    return out


# --------------------------------------------------------------------------- orphans
def _orphan_cells(claims: Sequence[Link], human: Mapping[str, _Human], grades: Mapping[str, Grade],
                  cells: Sequence[Cell], witness: str, witness_segments: Sequence[Segment],
                  groups: Sequence[ChapterGroup], charset: frozenset[str]) -> list[Cell]:
    wit_by_id = {s.id: s for s in witness_segments}
    position = {s.id: i for i, s in enumerate(witness_segments)}
    chapter_of_segment = _chapter_of_segment(cells)
    machine: dict[str, Link] = {}
    for link in claims:
        for sid in link.wit_ids:
            machine.setdefault(sid, link)
    human_rows = {orphan_source(uid): h for uid, h in human.items() if is_orphan(uid)}
    rows: list[Cell] = []
    for sid in [s.id for s in witness_segments if s.id in machine or s.id in human_rows]:
        if sid in chapter_of_segment:          # linked by a reference cell: not witness-only
            continue
        row_id = orphan_row_id(sid)
        h = human_rows.get(sid)
        if h is not None and h.value == HAS_COUNTERPART:
            continue
        if h is not None:
            try:
                kind = WitnessOnlyKind(h.value)
            except ValueError:
                raise ValueError(f"{row_id}: human decision {h.value!r} is not a witness-only kind "
                                 f"or {HAS_COUNTERPART} (from {h.source})") from None
            grade, flags, source = Grade.A, h.flags, h.source
        else:
            link = machine[sid]
            kind, flags, source = WitnessOnlyKind(link.relation), link.flags, link.source
            if row_id not in grades:
                raise ValueError(f"{row_id}: consensus witness-only record without a grade")
            grade = grades[row_id]
        rows.append(Cell(unit_id=row_id, witness=witness, status=Status.NA, grade=grade,
                         chapter=_attribute(position[sid], witness_segments, chapter_of_segment, groups),
                         relation=kind.value, wit_ids=(sid,), flags=flags, source=source,
                         d_lit=literal_density([wit_by_id[sid]], charset)))
    return rows


def attribute_chapter(segment_id: str, witness_segments: Sequence[Segment], cells: Sequence[Cell],
                      groups: Sequence[ChapterGroup]) -> str:
    """Reference chapter of a witness-only segment (critique B15; rule in the module docstring)."""
    index = next(i for i, s in enumerate(witness_segments) if s.id == segment_id)
    return _attribute(index, witness_segments, _chapter_of_segment(cells), groups)


def _chapter_of_segment(cells: Sequence[Cell]) -> dict[str, str]:
    """Witness segment id -> chapter of the first reference cell (in reference order) linking it."""
    out: dict[str, str] = {}
    for c in cells:
        for w in c.wit_ids:
            out.setdefault(w, c.chapter)
    return out


def _attribute(index: int, order: Sequence[Segment], chapter_of_segment: Mapping[str, str],
               groups: Sequence[ChapterGroup]) -> str:
    local = order[index].local_chapter
    group = next(((frozenset(r), frozenset(w)) for r, w in groups if local in frozenset(w)), None)
    if group is not None:
        refs, locals_ = group
        if len(refs) == 1:
            return next(iter(refs))
        found = _nearest(index, order, chapter_of_segment, lambda s, ch: s.local_chapter in locals_ and ch in refs)
        if found:
            return found
    return _nearest(index, order, chapter_of_segment, lambda s, ch: bool(ch))


def _nearest(index: int, order: Sequence[Segment], chapter_of_segment: Mapping[str, str],
             accept: Callable[[Segment, str], bool]) -> str:
    """Chapter of the nearest accepted linked segment (the preceding one on a tie), or ""."""
    for distance in range(1, len(order)):
        for j in (index - distance, index + distance):
            if 0 <= j < len(order) and order[j].id in chapter_of_segment:
                chapter = chapter_of_segment[order[j].id]
                if accept(order[j], chapter):
                    return chapter
    return ""


# --------------------------------------------------------------------------- helpers
def _check_groups(groups: Sequence[ChapterGroup]) -> None:
    for side in (0, 1):
        seen: set[str] = set()
        for group in groups:
            keys = group[side]
            if isinstance(keys, str):
                raise TypeError(f"chapter keys must be a collection of strings, got the string {keys!r}")
            repeated = seen & set(keys)
            if repeated:
                raise ValueError(f"chapter key(s) {sorted(repeated)} occur in more than one group")
            seen.update(keys)
