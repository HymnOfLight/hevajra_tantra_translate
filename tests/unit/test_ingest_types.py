"""Unit tests of the shared ingest records and helpers (hevajra_matrix.ingest)."""

from __future__ import annotations

import dataclasses

import pytest

from hevajra_matrix.core.types import Segment
from hevajra_matrix.ingest import CONTENT_KINDS, Footnote, IngestResult, Variant, duplicate_ids, kind_counts


def seg(sid: str, kind: str = "prose") -> Segment:
    return Segment(id=sid, witness="w", lang="zh", text="x", start="0001a01", end="0001a01", kind=kind)


def test_records_are_frozen():
    note = Footnote(n="1", locus="0001a01", segment_id="T1:0001a01.1", zh_lemma="", sa_text="a", source="taisho",
                    text="a")
    variant = Variant(segment_id="D1:1a.1.1", locus="1a.1", reading="b", alternative="a", kind="orthographic_variant")
    result = IngestResult(segments=(seg("T1:0001a01.1"),))
    for record in (note, variant, result):
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.segments = ()     # type: ignore[misc]
    assert (result.footnotes, result.variants, dict(result.metadata), dict(result.report)) == ((), (), {}, {})


def test_kind_counts_and_duplicate_ids():
    segments = [seg("T1:0001a01.1"), seg("T1:0001a01.2", "note"), seg("T1:0001a01.1", "head")]
    assert kind_counts(segments) == {"head": 1, "note": 1, "prose": 1}
    assert duplicate_ids(segments) == ["T1:0001a01.1"]
    assert duplicate_ids(segments[:2]) == []


def test_content_kinds_exclude_structure_and_evidence():
    assert CONTENT_KINDS == {"prose", "verse_line", "verse", "mantra"}
    assert not CONTENT_KINDS & {"head", "meta", "colophon", "paratext", "note"}
