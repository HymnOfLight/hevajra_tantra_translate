"""Checks that need more than one window: V9, V10, and merging chunks into a replicate.

    compare_overlaps  V10: the units two adjacent chunks of a chapter both read are
                      compared; the agreement statistic is ``OverlapReport.rate``
    merge_chapter     one chapter's chunks as one ``Collation``; overlap units read
                      differently get the flag ``overlap_disagreement``; V9: every
                      core-window witness segment must be linked or witness-only in some
                      chunk, else an ``unaccounted`` diagnostic
    merge_text        the chapters as one text-level ``Collation``: witness-only records
                      reconciled across chapters, and V9 for segments only ever declared
                      ``belongs_elsewhere``
    verify_all        one replicate end to end (verify, merge chapters, merge text)

A chunk's model sees the chapter's whole core window but only part of the chapter's
reference units, so it declares the witness segments that render the other chunks'
units ``belongs_elsewhere``. Those declarations are resolved here: a segment linked by
any reference unit is never witness-only (``verify.reconcile_witness_only``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from ..core.types import REASON_UNASSESSED, Alignment, Diagnostic, Link, WitnessOnlyKind
from ..matrix.status import outcome_class
from .collator import Parsed
from .verify import FLAG_OVERLAP_DISAGREEMENT, CheckLexicon, Collation, frozen_mapping, reconcile_witness_only, verify
from .windows import Window


@dataclass(frozen=True)
class OverlapReport:
    """V10: units read by two adjacent chunks and resolved in both."""

    compared: int = 0
    agreed: int = 0
    disagreeing: tuple[str, ...] = ()

    @property
    def rate(self) -> float | None:
        return self.agreed / self.compared if self.compared else None

    def __add__(self, other: "OverlapReport") -> "OverlapReport":
        return OverlapReport(self.compared + other.compared, self.agreed + other.agreed,
                             self.disagreeing + other.disagreeing)


def compare_overlaps(windows: Sequence[Window], results: Sequence[Collation]) -> OverlapReport:
    """Two readings agree when they give the same outcome class and the same segments."""
    _check_chapter(windows, results)
    report = OverlapReport()
    for i, window in enumerate(windows[:-1]):
        first, second = results[i].alignment.by_ref(), results[i + 1].alignment.by_ref()
        for uid in window.overlap_ids:
            if uid in first and uid in second:
                kind = window.segment[uid].kind
                a, b = first[uid], second[uid]
                same = (set(a.wit_ids) == set(b.wit_ids) and outcome_class(a.relation, a.polarity_flip, kind)
                        == outcome_class(b.relation, b.polarity_flip, kind))
                report += OverlapReport(1, int(same), () if same else (uid,))
    return report


def merge_chapter(windows: Sequence[Window], results: Sequence[Collation]) -> Collation:
    """One chapter's chunks as one collation, with V10 flags and the V9 check.

    An overlap unit's link comes from the chunk that holds it farthest from the chunk's
    edges (the earlier chunk on a tie), so the first half of an overlap is read from the
    earlier chunk and the second half from the later one; when that chunk left the unit
    unresolved, a resolved reading from the other chunk is used.
    """
    _check_chapter(windows, results)
    held: dict[str, list[tuple[int, int]]] = {}          # unit id -> [(-distance from edge, window)]
    for i, window in enumerate(windows):
        n = len(window.units)
        for j, unit in enumerate(window.units):
            held.setdefault(unit.id, []).append((-min(j, n - 1 - j), i))
    overlap = compare_overlaps(windows, results)
    disagreeing = set(overlap.disagreeing)
    by_ref = [r.alignment.by_ref() for r in results]
    links: list[Link] = []
    unresolved: dict[str, str] = {}
    diagnostics = [d for r in results for d in r.diagnostics]
    for uid, entries in held.items():
        order = [i for _, i in sorted(entries)]
        chosen = next((by_ref[i][uid] for i in order if uid in by_ref[i]), None)
        if chosen is None:
            unresolved[uid] = results[order[0]].unresolved.get(uid, REASON_UNASSESSED)
        elif uid in disagreeing:
            links.append(replace(chosen, flags=chosen.flags | {FLAG_OVERLAP_DISAGREEMENT}))
            diagnostics.append(Diagnostic(FLAG_OVERLAP_DISAGREEMENT, (uid,),
                                          detail=f"{windows[0].chapter}: adjacent chunks read this unit differently"))
        else:
            links.append(chosen)
    witness_only = reconcile_witness_only(links, [w for r in results for w in r.alignment.witness_only()])
    accounted = {i for link in (*links, *witness_only) for i in link.wit_ids}
    window = windows[0]
    diagnostics += [Diagnostic("unaccounted", wit_ids=run,
                               detail=f"{window.chapter}: {len(run)} core-window segment(s) neither linked nor "
                                      "witness-only")
                    for run in _unaccounted_runs(window, accounted)]
    first = results[0].alignment
    return Collation(Alignment(first.source, first.reference, first.witness, tuple(links + witness_only)),
                     frozen_mapping(unresolved), tuple(diagnostics), tuple(h for r in results for h in r.hints))


def merge_text(chapters: Sequence[Collation]) -> Collation:
    """Chapter collations as one text-level collation (the replicate's alignment).

    Witness-only records are reconciled across chapters (a neighbour chapter's
    ``belongs_elsewhere`` record disappears once the segment is linked where it
    belongs). A segment still only declared ``belongs_elsewhere`` gets an
    ``unaccounted`` diagnostic (V9 over the whole text) and keeps its record.
    """
    if not chapters:
        raise ValueError("merge_text: no chapters")
    first = chapters[0].alignment
    if any((c.alignment.source, c.alignment.reference, c.alignment.witness) !=
           (first.source, first.reference, first.witness) for c in chapters):
        raise ValueError("merge_text: chapters from different sources or text pairs")
    unit_links = [link for c in chapters for link in c.alignment.links if link.ref_id is not None]
    ids = [link.ref_id for link in unit_links] + [u for c in chapters for u in c.unresolved]
    if len(ids) != len(set(ids)):
        raise ValueError("merge_text: a reference unit occurs in two chapters")
    witness_only = reconcile_witness_only(unit_links, [w for c in chapters for w in c.alignment.witness_only()])
    elsewhere = tuple(i for w in witness_only if w.relation is WitnessOnlyKind.BELONGS_ELSEWHERE
                      for i in w.wit_ids)
    diagnostics = [d for c in chapters for d in c.diagnostics]
    if elsewhere:
        diagnostics.append(Diagnostic("unaccounted", wit_ids=elsewhere,
                                      detail="declared belongs_elsewhere but linked by no reference unit "
                                             "(meaningful when every chapter was collated)"))
    return Collation(Alignment(first.source, first.reference, first.witness, tuple(unit_links + witness_only)),
                     frozen_mapping({u: r for c in chapters for u, r in c.unresolved.items()}), tuple(diagnostics),
                     tuple(h for c in chapters for h in c.hints))


def verify_all(windows: Sequence[Window], parsed: Sequence[Parsed], lexicon: CheckLexicon,
               source: str) -> tuple[Collation, OverlapReport]:
    """One replicate end to end: verify every window, merge chapters, merge the text."""
    if len(windows) != len(parsed):
        raise ValueError("verify_all: one parse result per window is required")
    per_window = [verify(p, w, lexicon, source) for w, p in zip(windows, parsed)]
    groups: dict[str, list[int]] = {}
    for i, window in enumerate(windows):
        groups.setdefault(window.chapter, []).append(i)
    chapters, overlap = [], OverlapReport()
    for idx in groups.values():
        ws, rs = [windows[i] for i in idx], [per_window[i] for i in idx]
        chapters.append(merge_chapter(ws, rs))
        overlap += compare_overlaps(ws, rs)
    return merge_text(chapters), overlap


def _check_chapter(windows: Sequence[Window], results: Sequence[Collation]) -> None:
    if not windows or len(windows) != len(results):
        raise ValueError("need one result per window, and at least one window")
    first = windows[0]
    if not all(w.chapter == first.chapter and w.core_locals == first.core_locals and w.text == first.text
               for w in windows):
        raise ValueError("windows of one chapter must share chapter, core window and witness text")
    for a, b in zip(windows, windows[1:]):
        if a.overlap_ids != tuple(u.id for u in b.units[:a.overlap_next]):
            raise ValueError(f"windows {a.key} and {b.key} are not consecutive chunks")
    own = [u.id for w in windows[:-1] for u in w.units[:len(w.units) - w.overlap_next]]
    if len(set(own + [u.id for u in windows[-1].units])) != len(own) + len(windows[-1].units):
        raise ValueError("chunks of one chapter must be given once each, in order")


def _unaccounted_runs(window: Window, accounted: set[str]) -> list[tuple[str, ...]]:
    """Maximal runs of consecutive core-window segments that are not in ``accounted``."""
    runs: list[tuple[str, ...]] = []
    current: list[str] = []
    for seg in window.text:
        if seg.id in window.core and seg.id not in accounted:
            current.append(seg.id)
        elif current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return runs
