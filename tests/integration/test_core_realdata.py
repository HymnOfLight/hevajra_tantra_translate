"""Real-data check of the coordinate id scheme (core.ids) on T0892 and Derge D417/D418.

Every ingested segment gets the id ``<prefix>:<start line>.<ordinal>`` (notes
``.n<ordinal>``) via ``core.ids.assign_ids``. This test shows on the real texts that the
ids are unique, parse back, sort in document order, and that a line holds only a
handful of segments (a degenerate coordinate stream, e.g. every segment on a
placeholder line, would still give unique ids, so the per-line maximum is bounded too).

It deliberately uses only ``Segment.start`` and ``Segment.kind`` from the ingesters, so
it exercises the rewritten ingesters end to end.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from hevajra_matrix.core.ids import assign_ids, orphan_row_id, parse_segment_id, sort_key

pytestmark = pytest.mark.realdata

CBETA_PREFIX = "T0892"
DERGE_TOH = ("D417", "D418")
MAX_SEGMENTS_PER_LINE = 50      # a Taisho line holds 17 characters, a Derge line about 57 syllables
                                # (median; max 87). Observed maxima: 9 (T0892), 16 (D417), 10 (D418).


def _segment_starts(raw_dir: Path) -> dict[str, list[tuple[str, bool]]]:
    """(start line, is_note) of every segment in document order, per id prefix."""
    from hevajra_matrix.ingest import cbeta, derge

    data = Path(__file__).resolve().parents[2] / "data"
    zh = cbeta.parse(raw_dir / "T18n0892.xml", "zh_T0892_song", data)
    bo = derge.parse(raw_dir / "derge_rgyud_bum_nga.txt", list(DERGE_TOH), "bo_derge_D417_418", data)
    starts = {CBETA_PREFIX: [(s.start, s.kind == "note") for s in zh.segments]}
    for toh in DERGE_TOH:
        starts[toh] = [(s.start, False) for s in bo.segments if s.id.startswith(toh + ":")]
    return starts


def id_report(starts: Mapping[str, Sequence[tuple[str, bool]]]) -> dict[str, dict]:
    """Assign ids per prefix, assert the invariants, and return per-prefix statistics."""
    report: dict[str, dict] = {}
    all_ids: list[str] = []
    for prefix, entries in starts.items():
        ids = assign_ids(prefix, entries)
        all_ids.extend(ids)
        refs = [parse_segment_id(i) for i in ids]
        assert [str(r) for r in refs] == ids
        content = [i for i, r in zip(ids, refs) if not r.is_note]
        assert sorted(content, key=sort_key) == content, f"{prefix}: ids do not sort in document order"
        per_line = Counter(line for line, _ in entries)
        busiest, count = per_line.most_common(1)[0]
        report[prefix] = {"segments": len(ids), "notes": len(ids) - len(content), "lines": len(per_line),
                          "max_per_line": count, "busiest_line": busiest}
    duplicates = [i for i, n in Counter(all_ids).items() if n > 1]
    assert duplicates == []
    assert len({orphan_row_id(i) for i in all_ids}) == len(all_ids)
    return report


def test_coordinate_ids_are_unique_and_ordered(raw_dir):
    report = id_report(_segment_starts(raw_dir))
    assert set(report) == {CBETA_PREFIX, *DERGE_TOH}
    for prefix, stats in report.items():
        print(f"{prefix}: {stats}")
        assert stats["segments"] > 100, prefix
        assert stats["max_per_line"] <= MAX_SEGMENTS_PER_LINE, (prefix, stats)
