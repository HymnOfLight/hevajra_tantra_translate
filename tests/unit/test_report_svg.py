"""``report.svg``: well-formed figures built from ``Cell`` lists."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from hevajra_matrix.core.types import Cell, Grade, Status
from hevajra_matrix.report.svg import STATUS_COLOR, chapter_heatmap, status_strip

NS = "{http://www.w3.org/2000/svg}"
CELLS = [
    Cell("D417:1b.1.1", "w", Status.PRESENT, Grade.B, chapter="I.1", relation="equivalent"),
    Cell("D417:1b.1.2", "w", Status.ABSENT, Grade.B, chapter="I.1", relation="no_counterpart"),
    Cell("D418:1b.1.1", "w", Status.PARTIAL, Grade.C, chapter="II.1", relation="abridged"),
    Cell("D418:1b.1.2", "w", Status.UNALIGNED, Grade.X, chapter="II.1", reason="refused:other"),
    Cell("+T0892:0601c01.1", "w", Status.NA, Grade.C, chapter="II.12", relation="addition"),
]


def test_status_strip_has_one_bar_per_unit_and_chapter_ticks(tmp_path: Path) -> None:
    path = tmp_path / "strip.svg"
    status_strip(CELLS, path, caption="unvalidated instrument output")
    root = ET.parse(path).getroot()
    bars = root.findall(f"{NS}rect")
    assert [b.get("fill") for b in bars] == [STATUS_COLOR[c.status] for c in CELLS[:4]], "orphans skipped"
    texts = [t.text for t in root.iter(f"{NS}text")]
    assert "I.1" in texts and "II.1" in texts and "II.12" not in texts
    assert "unvalidated instrument output" in texts[0]


def test_chapter_heatmap_counts_cells_per_chapter_and_status(tmp_path: Path) -> None:
    path = tmp_path / "heat.svg"
    chapter_heatmap(CELLS, path)
    root = ET.parse(path).getroot()
    labels = [t.text for t in root.iter(f"{NS}text")]
    assert labels.count("1") == 4, "each of the four unit cells is alone in its chapter x status box"
    assert labels.index("I.1") < labels.index("II.1"), "chapters in reference order"
    assert len(root.findall(f"{NS}rect")) == 2 * 5


def test_empty_matrix_still_writes_a_valid_file(tmp_path: Path) -> None:
    status_strip([], tmp_path / "a.svg")
    chapter_heatmap([], tmp_path / "b.svg")
    assert ET.parse(tmp_path / "a.svg").getroot().tag == f"{NS}svg"
    assert ET.parse(tmp_path / "b.svg").getroot().tag == f"{NS}svg"
