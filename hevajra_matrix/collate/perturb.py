"""Perturbation placebos (synthesis 5.3): does the collator read the text it is given?

Each builder returns the perturbed window and the truth needed to score the answer to it;
the answer is produced and verified exactly like a normal one. The perturbations detect
filling-in from memory of T0892 and position or length heuristics:

    wrong_window     the core window is moved to a witness chapter that is not in the
                     original core window. The full text is still supplied, so a reader of
                     the text links the true counterparts outside the window (relocation)
                     or says no_counterpart; links INTO the wrong window are false.
                     G2: false-link rate <= 0.10.
    delete_segments  a fraction of the core window's content segments (with the witness
                     notes attached to them) is removed from the witness text. Handles are
                     renumbered, so the gap leaves no trace. The reference units whose
                     whole counterpart was removed must come out PARTIAL or ABSENT.
                     G2: recall >= 0.70.
    remove_negators  the negator is removed from the witness segment of gold "equivalent"
                     pairs in which both sides are negated; a reader must now report a
                     reversal or a polarity flip. Optional; polarity recall is reported.

Scores are ``Rate`` values (hits / n); an unresolved unit counts as a miss, never as a hit.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, replace
from typing import Sequence

from ..core.lexicon import Negators
from ..core.textnorm import TSHEG, TSHEG_NB
from ..core.types import Alignment, Relation
from ..ingest import CONTENT_KINDS
from .windows import Window

COVERAGE_LOSS = frozenset({Relation.ABRIDGED, Relation.NO_COUNTERPART})


@dataclass(frozen=True)
class Rate:
    hits: int
    n: int

    @property
    def value(self) -> float | None:
        return self.hits / self.n if self.n else None


@dataclass(frozen=True)
class WrongWindowTruth:
    core: frozenset[str]              # witness segment ids of the wrong core window


@dataclass(frozen=True)
class DeletionTruth:
    deleted: tuple[str, ...]          # witness segment ids removed from the text (content and notes)
    units: tuple[str, ...] = ()       # reference units scored for this window (empty: not restricted);
                                      # the units shared with the next chunk are left to that chunk


@dataclass(frozen=True)
class NegationTruth:
    pairs: tuple[tuple[str, str], ...]   # (reference unit id, witness segment id) now differing in polarity


# --------------------------------------------------------------------------- builders
def wrong_window(window: Window, other_local_keys: Sequence[str]) -> tuple[Window, WrongWindowTruth]:
    """Point the core window at ``other_local_keys`` (disjoint from the original window)."""
    others = frozenset(other_local_keys)
    present = {s.local_chapter for s in window.text}
    if not others or others & window.core_locals or not others <= present:
        raise ValueError(f"window {window.key}: the wrong window must be non-empty witness chapters outside "
                         f"{sorted(window.core_locals)}, got {sorted(others)}")
    perturbed = replace(window, key=f"{window.key}+wrong", core_locals=others)
    return perturbed, WrongWindowTruth(perturbed.core)


def delete_segments(window: Window, fraction: float, seed: int) -> tuple[Window, DeletionTruth]:
    """Remove round(fraction * n) (at least one) of the n core-window content segments."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    candidates = [s.id for s in window.text if s.id in window.core and s.kind in CONTENT_KINDS]
    if not candidates:
        raise ValueError(f"window {window.key}: no content segment in the core window")
    chosen = set(random.Random(seed).sample(candidates, max(1, round(fraction * len(candidates)))))
    chosen |= {s.id for s in window.text if s.kind == "note" and s.extra.get("host") in chosen}
    kept = tuple(s for s in window.text if s.id not in chosen)
    deleted = tuple(s.id for s in window.text if s.id in chosen)
    # The last ``overlap_next`` units open the next chunk too; they are scored there only, so
    # deletion recall counts every unit once.
    shared = set(window.overlap_ids)
    return (replace(window, key=f"{window.key}-del{seed}", text=kept),
            DeletionTruth(deleted, tuple(u.id for u in window.units if u.id not in shared)))


def remove_negators(window: Window, gold_pairs: Sequence[tuple[str, str]], negators: Negators,
                    n: int) -> tuple[Window, NegationTruth]:
    """Remove one negator from the witness segment of up to ``n`` gold pairs.

    ``gold_pairs`` are (reference unit id, witness segment id) of gold ``equivalent`` links
    in which both sides are negated, in the order to use them. Pairs outside the window
    or whose witness segment holds no removable negator are skipped.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    units = {u.id for u in window.units}
    texts = {s.id: s.text for s in window.text}
    changed: dict[str, str] = {}
    used: list[tuple[str, str]] = []
    for ref_id, wit_id in gold_pairs:
        if len(used) == n:
            break
        if ref_id not in units or wit_id not in texts:
            continue
        if wit_id not in changed:
            stripped = strip_negator(texts[wit_id], window.wit_lang, negators)
            if stripped is None:
                continue
            changed[wit_id] = stripped
        used.append((ref_id, wit_id))
    text = tuple(replace(s, text=changed[s.id]) if s.id in changed else s for s in window.text)
    return replace(window, key=f"{window.key}-neg", text=text), NegationTruth(tuple(used))


def strip_negator(text: str, lang: str, negators: Negators) -> str | None:
    """``text`` without its first negator (outside the exclusion contexts), or None.

    Chinese negators are single characters; in other languages a form must stand as a
    whole token (Tibetan syllable, word), and one adjacent tsheg or space goes with it.
    """
    blocked = [(i, i + len(x)) for x in negators.exclusions_for(lang) for i in _find_all(text, x)]
    hits = sorted((i, form) for form in negators.for_lang(lang) for i in _find_all(text, form)
                  if not any(a <= i and i + len(form) <= b for a, b in blocked)
                  and (lang == "zh" or (_edge(text, i - 1) and _edge(text, i + len(form)))))
    if not hits:
        return None
    i, form = hits[0]
    j = i + len(form)
    if lang != "zh":
        if j < len(text) and (text[j] in (TSHEG, TSHEG_NB) or text[j].isspace()):
            j += 1
        elif i > 0 and (text[i - 1] in (TSHEG, TSHEG_NB) or text[i - 1].isspace()):
            i -= 1
    return text[:i] + text[j:]


def _find_all(text: str, sub: str) -> list[int]:
    out, i = [], text.find(sub)
    while sub and i != -1:
        out.append(i)
        i = text.find(sub, i + 1)
    return out


def _edge(text: str, i: int) -> bool:
    """True at a token edge: outside the text, a tsheg, a space, punctuation or a symbol."""
    if i < 0 or i >= len(text):
        return True
    ch = text[i]
    return ch in (TSHEG, TSHEG_NB) or ch.isspace() or unicodedata.category(ch)[0] in "PS"


# --------------------------------------------------------------------------- scoring
def false_link_rate(truth: WrongWindowTruth, result: Alignment) -> Rate:
    """Resolved reference units linked into the wrong window / resolved reference units."""
    links = result.by_ref().values()
    return Rate(sum(1 for link in links if truth.core.intersection(link.wit_ids)), len(links))


def deletion_recall(truth: DeletionTruth, expected: Alignment, result: Alignment) -> Rate:
    """Units whose whole ``expected`` counterpart was deleted, found PARTIAL or ABSENT.

    ``expected`` is gold, or else the unperturbed run's consensus; only its units with a
    non-empty counterpart entirely inside ``truth.deleted`` are scored, and only those of the
    perturbed window (``truth.units``): a deleted segment may be the counterpart of a unit in
    another chunk or a neighbouring chapter, which this answer never assesses.
    """
    deleted = set(truth.deleted)
    window = set(truth.units)
    affected = [uid for uid, link in expected.by_ref().items() if link.wit_ids and deleted.issuperset(link.wit_ids)
                and (not window or uid in window)]
    found = result.by_ref()
    return Rate(sum(1 for uid in affected if uid in found and found[uid].relation in COVERAGE_LOSS), len(affected))


def negation_recall(truth: NegationTruth, result: Alignment) -> Rate:
    """Perturbed units reported as a reversal or with a polarity flip."""
    found = result.by_ref()
    units = sorted({ref for ref, _ in truth.pairs})
    hits = sum(1 for uid in units if uid in found
               and (found[uid].relation is Relation.REVERSAL or found[uid].polarity_flip))
    return Rate(hits, len(units))
