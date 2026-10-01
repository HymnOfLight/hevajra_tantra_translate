"""Unit tests of hevajra_matrix.ingest.sanskrit (reference TSV and per-manuscript readings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hevajra_matrix.core.textnorm import fingerprint
from hevajra_matrix.ingest import sanskrit

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "fixtures"
HEADER = "unit_id\tms\tstatus\treading\tsource\tnote\n"


# --------------------------------------------------------------------------- reference units
def test_reference_units_in_file_order():
    segs = sanskrit.load_reference(FIXTURES / "sa_edition_a.tsv", "sa_a")
    assert [(s.id, s.kind, s.chapter, s.local_chapter) for s in segs] == [
        ("I.1.p01", "prose", "I.1", "I.1"),
        ("I.1.p02", "prose", "I.1", "I.1"),
        ("I.1.1", "verse", "I.1", "I.1"),
        ("I.1.2", "verse", "I.1", "I.1"),
        ("I.1.3a", "verse", "I.1", "I.1"),
        ("I.2.1", "verse", "I.2", "I.2"),
        ("II.1.1", "verse", "II.1", "II.1"),
    ]
    for s in segs:
        assert (s.witness, s.lang, s.start, s.end) == ("sa_a", "sa", s.id, s.id)
        assert s.fingerprint == fingerprint(s.text, "sa") and s.extra == {}


def test_text_is_kept_with_whitespace_collapsed():
    segs = {s.id: s for s in sanskrit.load_reference(FIXTURES / "sa_edition_a.tsv", "sa_a")}
    assert segs["I.1.p02"].text == "vajragarbha mahāsattva sādhu sādhu"
    assert segs["I.1.1"].text == "dvātriṃśan nāḍyaḥ"


def test_flags_mark_units_without_text():
    segs = {s.id: s for s in sanskrit.load_reference(FIXTURES / "sa_edition_b.tsv", "sa_b")}
    assert (segs["I.1.2"].text, segs["I.1.2"].extra) == ("", {"flag": "ABSENT"})
    assert (segs["I.2.1"].text, segs["I.2.1"].extra) == ("", {"flag": "LACUNA"})


@pytest.mark.parametrize("line, message", [
    ("I.1.p001\tevam\n", "non-canonical"),
    ("I.1.p1\tevam\n", "bad reference unit id"),
    ("I.12.1\tevam\n", "out of range"),
    ("X.1.1\tevam\n", "unknown part"),
    ("I.1.1\tevam\tMISSING\n", "unknown flag"),
    ("I.1.1\t\n", "no text"),
    ("I.1.1\tevam\n\nI.1.1\tmaya\n", "duplicate"),
    ("I.1.1\tevam\tLACUNA\textra\n", "columns"),
])
def test_malformed_reference_files_are_rejected(tmp_path, line, message):
    path = tmp_path / "sa_x.tsv"
    path.write_text("# comment\n" + line, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        sanskrit.load_reference(path, "sa_x")


# --------------------------------------------------------------------------- readings
def test_readings_are_grouped_by_unit():
    table = sanskrit.load_readings(FIXTURES / "sa_readings")
    assert [(r.ms, r.status, r.reading) for r in table.for_unit("I.1.1")] == [
        ("C", "present", ""), ("K", "variant", "dvātriṃśa nāḍyaḥ"), ("Na", "not_collated", "")]
    assert table.for_unit("I.1.p01")[0].status == "illegible"       # short row, missing columns padded
    assert table.for_unit("I.1.p01")[0].note == ""
    assert table.for_unit("I.9.9") == ()
    assert table.manuscripts() == frozenset({"C", "K", "Na", "P"})
    assert [r.ms for r in table.for_unit("II.1.1")] == ["C"]


def test_readings_statuses_are_the_documented_ones():
    assert sanskrit.READING_STATUSES == {"present", "absent", "variant", "illegible", "not_collated"}


def write_readings(directory: Path, name: str, body: str, header: str = HEADER) -> Path:
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(header + body, encoding="utf-8")
    return directory


@pytest.mark.parametrize("name, body, header, message", [
    ("I.1.tsv", "I.1.1\tC\tlost\t\t\t\n", HEADER, "status"),
    ("I.1.tsv", "I.1.1\tC\tvariant\t\t\t\n", HEADER, "needs the reading"),
    ("I.1.tsv", "I.2.1\tC\tpresent\t\t\t\n", HEADER, "belongs to I.2"),
    ("I.1.tsv", "I.1.1\t\tpresent\t\t\t\n", HEADER, "manuscript"),
    ("I.1.tsv", "I.1.1\tC\n", HEADER, "columns"),
    ("I.1.tsv", "I.1.1\tC\tpresent\t\t\t\nI.1.1\tC\tabsent\t\t\t\n", HEADER, "second reading"),
    ("I.1.tsv", "I.1.1\tC\tpresent\t\t\t\n", "unit\tms\tstatus\n", "header"),
    ("chapter1.tsv", "I.1.1\tC\tpresent\t\t\t\n", HEADER, "file name"),
])
def test_malformed_readings_are_rejected(tmp_path, name, body, header, message):
    directory = write_readings(tmp_path / "readings", name, body, header)
    with pytest.raises(ValueError, match=message):
        sanskrit.load_readings(directory)


def test_missing_readings_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        sanskrit.load_readings(tmp_path / "absent")


def test_readings_may_be_keyed_by_derge_segment_ids(tmp_path):
    """While the Derge is the provisional reference, readings use its segment ids; the file's
    chapter is not checked against them (the stats stage reports ids that are not units)."""
    directory = write_readings(tmp_path / "readings", "I.1.tsv",
                               "D417:1b.2.1\tC\tpresent\t\t\t\nD417:1b.2.2\tK\tabsent\t\t\t\n")
    table = sanskrit.load_readings(directory)
    assert [r.status for r in table.for_unit("D417:1b.2.2")] == ["absent"]
    assert table.manuscripts() == {"C", "K"}
    bad = write_readings(tmp_path / "bad", "I.1.tsv", "D417:1b\tC\tpresent\t\t\t\n")
    with pytest.raises(ValueError, match="bad segment id"):
        sanskrit.load_readings(bad)
