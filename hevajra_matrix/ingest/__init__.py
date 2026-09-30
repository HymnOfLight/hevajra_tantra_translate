"""Ingest: turn the raw witness files into coordinate-keyed segments and evidence records.

Every parser returns an ``IngestResult``. The parsers live in submodules and are
imported explicitly (``from hevajra_matrix.ingest import cbeta``), which keeps this
module free of import cycles:

    cbeta.parse(path, witness, data_dir)            CBETA TEI P5 (T0892)
    derge.parse(path, toh, witness, data_dir)       Esukhia Derge Kangyur volume text
    sanskrit.load_reference(path, witness)          researcher-supplied Sanskrit TSV
    sanskrit.load_readings(directory)               per-manuscript readings (decomposition)
    notes.classify(text, rules)                     class of a CBETA inline note

What an ingester guarantees
    * Segment ids come from ``core.ids.assign_ids`` (witness coordinates, never a running
      counter) and every segment carries ``core.textnorm.fingerprint(text, lang)``.
    * Nothing is silently dropped: translator notes become ``kind="note"`` segments,
      Taisho footnotes become ``Footnote`` records, every reading the ingester resolved
      (Derge ``{a,b}`` marks, the CBETA apparatus) becomes a ``Variant``, and material
      outside the work proper is kept as ``kind="paratext"``.
    * ``report`` holds the counts that the G0 integrity gate and the ingest report read.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..core.types import Segment

# Segment kinds that carry the text of the work and can be aligned. The other kinds are
# head (chapter titles), meta (front matter, fascicle headings, bylines), colophon
# (chapter and text-end colophons, structural anchors), paratext (outside the work) and
# note (translator notes, kept as evidence).
CONTENT_KINDS = frozenset({"prose", "verse_line", "verse", "mantra"})


@dataclass(frozen=True)
class Footnote:
    """A Taisho editors' footnote (T0892 has 181): an indirect Sanskrit witness.

    Footnotes are evidence for the decomposition and for human reviewers; they never
    reach the collator, whose Chinese text is the translation alone.

    ``n``           Taisho note number, page + running number (e.g. "0594008")
    ``locus``       Taisho line of the note's anchor (e.g. "0594b28")
    ``segment_id``  segment holding the anchor, i.e. the first character of the lemma
    ``zh_lemma``    the Chinese lemma the note is attached to ("" when it gives none)
    ``sa_text``     the rest of the note, verbatim (normally romanised Sanskrit)
    ``source``      "tokyo335" when the note carries the manuscript marker (a reading of
                    Tokyo Imperial University Sanskrit manuscript no. 335), else "taisho"
                    (a Sanskrit equivalent supplied by the Taisho editors)
    ``text``        the whole note, verbatim
    """

    n: str
    locus: str
    segment_id: str
    zh_lemma: str
    sa_text: str
    source: str
    text: str


@dataclass(frozen=True)
class Variant:
    """A reading the ingester chose between, logged so that the other one is not lost.

    ``reading`` is the form kept in the segment text; ``alternative`` the form set aside.
    ``kind`` is "orthographic_variant" for a Derge ``{a,b}`` mark (b is kept) and
    "apparatus" for a CBETA apparatus entry (the lemma, already in the text, is kept).
    """

    segment_id: str
    locus: str
    reading: str
    alternative: str
    kind: str


@dataclass(frozen=True)
class IngestResult:
    """Everything one parser read from one witness file.

    ``metadata`` is JSON-ready witness information (chapters, glosses, translators'
    colophon, ...) and ``report`` JSON-ready counts; treat both as read-only.
    """

    segments: tuple[Segment, ...]
    footnotes: tuple[Footnote, ...] = ()
    variants: tuple[Variant, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    report: Mapping[str, Any] = field(default_factory=dict)


def kind_counts(segments: Iterable[Segment]) -> dict[str, int]:
    """Number of segments per kind, sorted by kind (for ingest reports)."""
    return dict(sorted(Counter(s.kind for s in segments).items()))


def duplicate_ids(segments: Iterable[Segment]) -> list[str]:
    """Ids that occur more than once (must be empty; the G0 gate checks it)."""
    return sorted(i for i, n in Counter(s.id for s in segments).items() if n > 1)
