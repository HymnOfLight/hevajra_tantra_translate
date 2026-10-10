"""matrix.export: collate.v2 conventions appear in the existing ``flags`` column of cells.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from hevajra_matrix.core.types import FLAG_CONVENTION_UNSUPPORTED, Cell, Grade, Status
from hevajra_matrix.matrix.export import CELL_COLUMNS, write_matrix


def test_convention_flags_are_exported_in_the_flags_column(tmp_path: Path) -> None:
    cells = [Cell("u1", "zh", Status.PRESENT, Grade.C, chapter="I.1", relation="paraphrase",
                  flags=frozenset({"transcribed_name", FLAG_CONVENTION_UNSUPPORTED}), source="claude:consensus"),
             Cell("u2", "zh", Status.PRESENT, Grade.A, chapter="I.1", relation="equivalent",
                  flags=frozenset({"standing_rendering"}), source="verdict:b1"),
             Cell("u3", "zh", Status.PRESENT, Grade.B, chapter="I.1", relation="equivalent")]
    path = write_matrix(cells, [], [], tmp_path)[0]
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == CELL_COLUMNS, "v1 and v2 runs share the header"
    assert [r["flags"] for r in rows] == ["convention_unsupported;transcribed_name", "standing_rendering", ""]
