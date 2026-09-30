"""matrix.export: the four matrix files, their exact English headers and contents."""

from __future__ import annotations

import csv
from pathlib import Path

from hevajra_matrix.core.types import Cell, Grade, Segment, Status
from hevajra_matrix.matrix.build import STALE_FINGERPRINT, StaleVerdict
from hevajra_matrix.matrix.export import CELL_COLUMNS, STALE_COLUMNS, UNIT_COLUMNS, write_matrix


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or ()), list(reader)


CELLS = [
    Cell("D417:1a.1.1", "zh_wit", Status.PRESENT, Grade.B, chapter="I.1", relation="equivalent",
         wit_ids=("T0892:0587a01.1", "T0892:0587a01.2"), flags=frozenset({"crossing", "corroborated"}),
         d_len=-0.123456, d_ord=0.0, d_lit=0.25, source="claude:consensus"),
    Cell("D417:1a.1.2", "zh_wit", Status.UNALIGNED, Grade.X, chapter="I.1", reason="refused:bio",
         source="claude:consensus"),
]
ORPHANS = [Cell("+T0892:0587a02.1", "zh_wit", Status.NA, Grade.B, chapter="I.1", relation="addition",
                wit_ids=("T0892:0587a02.1",), d_lit=0.0, source="claude:consensus")]
STALE = [StaleVerdict("D417:1a.1.3", "b1", "verify:D417:1a.1.3", "verify", "aaaaaaaaaaaa", "bbbbbbbbbbbb",
                      STALE_FINGERPRINT, quote_ref="q")]


def test_write_matrix_files_and_headers(tmp_path: Path) -> None:
    ref = Segment("D417:1a.1.1", "bo_ref", "bo", "ka kha ga", "1a.1", "1a.1", "verse", chapter="I.1")
    paths = write_matrix(CELLS, ORPHANS, STALE, tmp_path / "matrix", reference_segments=[ref])
    assert [p.name for p in paths] == ["cells.csv", "units.csv", "wide_status.csv", "stale_verdicts.csv"]

    header, rows = _read(paths[0])
    assert tuple(header) == CELL_COLUMNS
    assert [r["row_type"] for r in rows] == ["unit", "unit", "orphan"]
    assert rows[0]["wit_ids"] == "T0892:0587a01.1 T0892:0587a01.2"
    assert rows[0]["flags"] == "corroborated;crossing"
    assert rows[0]["d_len"] == "-0.1235" and rows[1]["d_len"] == ""
    assert rows[1]["reason"] == "refused:bio"

    header, rows = _read(paths[1])
    assert tuple(header) == UNIT_COLUMNS
    assert rows[0]["kind"] == "verse" and rows[0]["length"] == "3" and rows[0]["fingerprint"]
    assert rows[2] == {"unit_id": "+T0892:0587a02.1", "row_type": "orphan", "chapter": "I.1",
                       "kind": "", "fingerprint": "", "length": ""}

    header, rows = _read(paths[2])
    assert header == ["unit_id", "chapter", "zh_wit"]
    assert [r["zh_wit"] for r in rows] == ["PRESENT", "UNALIGNED", "NA"]

    header, rows = _read(paths[3])
    assert tuple(header) == STALE_COLUMNS
    assert rows[0]["reason"] == STALE_FINGERPRINT and rows[0]["current_fingerprint"] == "bbbbbbbbbbbb"


def test_headers_are_ascii_english() -> None:
    for cols in (CELL_COLUMNS, UNIT_COLUMNS, STALE_COLUMNS):
        assert all(c.isascii() and c == c.lower() for c in cols)


def test_empty_matrix_still_writes_headers(tmp_path: Path) -> None:
    paths = write_matrix([], [], [], tmp_path)
    assert all(p.read_text(encoding="utf-8").strip() for p in paths)
