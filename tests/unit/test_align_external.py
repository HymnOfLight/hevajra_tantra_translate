"""align.external: the TSV import path for external alignments (MITRA-E, DharmaNexus)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hevajra_matrix.align.external import ExternalAlignmentError, dump_tsv, load_tsv, write_tsv
from hevajra_matrix.core.types import Alignment, Link, Relation, WitnessOnlyKind

SOURCE = "external:test"


def _write(tmp_path: Path, text: str, name: str = "a.tsv", bom: bool = False) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")
    return path


def _load(path: Path) -> Alignment:
    return load_tsv(path, reference="bo", witness="zh", source=SOURCE)


def test_round_trip(tmp_path):
    links = (
        Link("D417:1a.1.1", ("T0892:0587c11.1",), Relation.EQUIVALENT, source=SOURCE),
        Link("D417:1a.1.2", ("T0892:0587c11.1", "T0892:0587c12.1"), Relation.PARAPHRASE, source=SOURCE),
        Link("D417:1a.2.1", (), Relation.NO_COUNTERPART, source=SOURCE),
        Link(None, ("T0892:0587c13.1",), WitnessOnlyKind.ADDITION, source=SOURCE),
        Link(None, ("T0892:0587c13.n1",), WitnessOnlyKind.TRANSLATOR_NOTE, source=SOURCE),
        Link("I.1.p03", ("T0892:0587c14.1",), Relation.SUBSTITUTION, source=SOURCE),
    )
    original = Alignment(source=SOURCE, reference="bo", witness="zh", links=links)
    path = tmp_path / "out.tsv"
    write_tsv(original, path)
    assert _load(path) == original
    assert path.read_text(encoding="utf-8").splitlines()[0] == "ref_id\twit_ids\trelation"


def test_relation_column_is_optional_and_defaults_follow_the_row(tmp_path):
    path = _write(tmp_path, "ref_id\twit_ids\nD417:1a.1.1\tT0892:0587c11.1 T0892:0587c11.2\nD417:1a.1.2\t\n"
                            "\tT0892:0587c12.1\n")
    got = [(l.ref_id, l.wit_ids, l.relation) for l in _load(path).links]
    assert got == [
        ("D417:1a.1.1", ("T0892:0587c11.1", "T0892:0587c11.2"), Relation.EQUIVALENT),
        ("D417:1a.1.2", (), Relation.NO_COUNTERPART),
        (None, ("T0892:0587c12.1",), WitnessOnlyKind.ADDITION),
    ]


def test_empty_relation_cells_take_the_default_and_blank_lines_are_skipped(tmp_path):
    path = _write(tmp_path, "ref_id\twit_ids\trelation\n\nD417:1a.1.1\tT0892:0587c11.1\t\n\t\t\n", bom=True)
    a = _load(path)
    assert [(l.ref_id, l.relation) for l in a.links] == [("D417:1a.1.1", Relation.EQUIVALENT)]
    assert (a.source, a.reference, a.witness) == (SOURCE, "bo", "zh")


@pytest.mark.parametrize("text, error", [
    ("ref\twit_ids\n", "header"),
    ("ref_id\twit_ids\tscore\n", "header"),
    ("ref_id\tref_id\twit_ids\n", "header"),
    ("ref_id\twit_ids\nD417:1a.1.1\tT0892:0587c11.1\textra\n", "more cells"),
    ("ref_id\twit_ids\nD417:1a.1.1\tT0892:0587c11.1\nD417:1a.1.1\tT0892:0587c12.1\n", "occurs twice"),
    ("ref_id\twit_ids\nD417:1a.1.1\tnot-an-id\n", "bad segment id"),
    ("ref_id\twit_ids\nD417 1a\tT0892:0587c11.1\n", "bad"),
    ("ref_id\twit_ids\n+T0892:0587c11.1\tT0892:0587c11.1\n", "matrix row id"),
    ("ref_id\twit_ids\nD417:1a.1.1\tT0892:0587c11.1 T0892:0587c11.1\n", "repeated witness id"),
    ("ref_id\twit_ids\n\t\n", None),
    ("ref_id\twit_ids\trelation\n\t\tno_counterpart\n", "witness-only row needs"),
    ("ref_id\twit_ids\trelation\nD417:1a.1.1\tT0892:0587c11.1\tomitted\n", "unknown relation"),
    ("ref_id\twit_ids\trelation\nD417:1a.1.1\t\tequivalent\n", "does not fit"),
    ("ref_id\twit_ids\trelation\nD417:1a.1.1\tT0892:0587c11.1\tno_counterpart\n", "does not fit"),
    ("ref_id\twit_ids\trelation\n\tT0892:0587c11.1\tequivalent\n", "unknown relation"),
    ("ref_id\twit_ids\trelation\nD417:1a.1.1\tT0892:0587c11.1\taddition\n", "unknown relation"),
])
def test_malformed_files_are_rejected_with_the_line(tmp_path, text, error):
    path = _write(tmp_path, text)
    if error is None:  # a row with only empty cells is a blank line
        assert _load(path).links == ()
        return
    with pytest.raises(ExternalAlignmentError, match=error) as info:
        _load(path)
    assert str(path) in str(info.value)


def test_dump_writes_explicit_relations():
    text = dump_tsv([Link("D417:1a.1.1", (), Relation.NO_COUNTERPART)])
    assert text == "ref_id\twit_ids\trelation\nD417:1a.1.1\t\tno_counterpart\n"
