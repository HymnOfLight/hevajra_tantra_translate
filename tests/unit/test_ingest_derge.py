"""Unit tests of hevajra_matrix.ingest.derge.

Two kinds of input:
    * data/fixtures/mini_derge.txt with the real markers (data/lexicon/derge_markers.yaml):
      front matter, {a,b} marks, chapter colophons of both forms, the text end of Toh 417,
      a Toh starting mid-line, a content unit that contains the text-end word, the
      translators' colophons and a following text that is not requested.
    * ASCII "volumes" built here with an ASCII marker file in a temporary data directory,
      to pin down one rule at a time. Words stand in for syllables and the shad is written
      as an escape (the language policy keeps Tibetan script out of .py files).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hevajra_matrix.core.ids import parse_segment_id
from hevajra_matrix.core.lexicon import LexiconError
from hevajra_matrix.core.textnorm import fingerprint
from hevajra_matrix.ingest import derge

DATA = Path(__file__).resolve().parents[2] / "data"
WITNESS = "bo_derge_D417_418"
S = "\u0f0d"                      # Tibetan shad: ends a unit
OM = "\u0f68\u0f7c\u0f7e"         # the syllable om with anusvara (Sanskrit-only sign)


@pytest.fixture(scope="module")
def mini():
    return derge.parse(DATA / "fixtures" / "mini_derge.txt", ["D417", "D418"], WITNESS, DATA)


# --------------------------------------------------------------------------- fixture
def test_segments_kinds_and_local_chapters(mini):
    got = [(s.id, s.kind, s.local_chapter) for s in mini.segments]
    assert got == [
        ("D417:1b.1.1", "meta", None),
        ("D417:1b.1.2", "meta", None),
        ("D417:1b.1.3", "meta", None),
        ("D417:1b.1.4", "meta", None),
        ("D417:1b.1.5", "meta", None),
        ("D417:1b.1.6", "verse_line", "D417:1"),
        ("D417:1b.2.1", "prose", "D417:1"),
        ("D417:1b.2.2", "verse_line", "D417:1"),
        ("D417:1b.2.3", "mantra", "D417:1"),
        ("D417:1b.2.4", "colophon", "D417:1"),
        ("D417:1b.3.1", "verse_line", "D417:2"),
        ("D417:1b.3.2", "verse_line", "D417:2"),
        ("D417:1b.4.1", "colophon", "D417:2"),
        ("D417:1b.4.2", "colophon", "D417:3"),
        ("D418:1b.4.1", "prose", "D418:1"),
        ("D418:1b.5.1", "verse_line", "D418:1"),     # contains the text-end word, but is not the last
        ("D418:1b.5.2", "prose", "D418:1"),
        ("D418:1b.5.3", "colophon", "D418:1"),
        ("D418:1b.5.4", "verse_line", "D418:2"),
        ("D418:1b.5.5", "colophon", "D418:2"),
        ("D418:1b.6.1", "colophon", "D418:3"),
        ("D418:1b.6.2", "paratext", None),
        ("D418:1b.6.3", "paratext", None),
        ("D418:2a.1.1", "paratext", None),
    ]


def test_ids_fingerprints_and_coordinates(mini):
    for s in mini.segments:
        assert str(parse_segment_id(s.id)) == s.id and s.id.split(":")[1].startswith(s.start)
        assert s.fingerprint == fingerprint(s.text, "bo")
        assert (s.witness, s.lang, s.chapter) == (WITNESS, "bo", None)
    by_id = {s.id: s for s in mini.segments}
    assert (by_id["D418:1b.4.1"].start, by_id["D418:1b.4.1"].end) == ("1b.4", "1b.5")   # across a line break


def test_edit_marks_resolve_to_b_and_are_logged(mini):
    got = [(v.segment_id, v.locus, v.kind) for v in mini.variants]
    assert got == [("D417:1b.2.2", "1b.2", "orthographic_variant"), ("D417:1b.3.2", "1b.3", "orthographic_variant"),
                   ("D418:1b.5.1", "1b.5", "orthographic_variant")]
    by_id = {s.id: s for s in mini.segments}
    for v in mini.variants:
        assert v.reading in by_id[v.segment_id].text and v.alternative not in by_id[v.segment_id].text
    assert mini.report["variants"] == 3 and mini.report["variant_pairs"] == 2
    assert not any("{" in s.text or "}" in s.text for s in mini.segments)


def test_chapters_metadata(mini):
    got = [(c["local"], c["ordinal"], c["text_end"]) for c in mini.metadata["chapters"]]
    assert got == [("D417:1", 1, False), ("D417:2", 2, False), ("D417:3", None, True),
                   ("D418:1", 1, False), ("D418:2", 2, False), ("D418:3", None, True)]
    assert mini.report["chapters"] == {"D417": 3, "D418": 3}
    assert mini.report["chapter_ordinal_mismatches"] == []


def test_translators_colophon_is_parsed_into_metadata(mini):
    meta = mini.metadata
    assert meta["paratext"] == ["D418:1b.6.2", "D418:1b.6.3", "D418:2a.1.1"]
    assert [(t["role"], t["segment_id"]) for t in meta["translators"]] == [
        ("indian_preceptor", "D418:1b.6.2"), ("translator", "D418:1b.6.3")]
    assert all(t["name"] for t in meta["translators"])
    assert meta["revision_statement"] == mini.segments[-1].text
    assert meta["reviser"] and meta["reviser"] in meta["revision_statement"]
    assert mini.report["colophon_roles"] == ["indian_preceptor", "translator", "reviser"]


def test_metadata_and_report_are_json_ready(mini):
    json.loads(json.dumps({"metadata": mini.metadata, "report": mini.report}, ensure_ascii=False))


def test_report(mini):
    assert mini.report["segments"] == 24
    assert mini.report["content_segments"] == 10
    assert mini.report["paratext"] == 3
    assert mini.report["kinds"] == {"colophon": 6, "mantra": 1, "meta": 5, "paratext": 3, "prose": 3,
                                    "verse_line": 6}
    assert mini.report["duplicate_ids"] == []
    assert mini.footnotes == ()


def test_only_requested_texts_and_their_marks(mini):
    only_d418 = derge.parse(DATA / "fixtures" / "mini_derge.txt", ["D418"], WITNESS, DATA)
    assert {s.id.split(":")[0] for s in only_d418.segments} == {"D418"}
    assert len(only_d418.variants) == 1
    d419 = derge.parse(DATA / "fixtures" / "mini_derge.txt", ["D419"], WITNESS, DATA)
    assert len(d419.variants) == 1 and d419.metadata["reviser"] is None


# --------------------------------------------------------------------------- ASCII volumes
ASCII_MARKERS = r"""
text_end: 'END'
chapter_colophon: 'chapter (?P<ordinal>[a-z]+) ends.?$'
ordinals:
  - {form: one, value: 1}
  - {form: two, value: 2}
  - {form: three, value: 3}
front_matter: {start: 'TITLE', end: 'THUS', start_within_units: 2, max_units: 3}
translator_colophon:
  - {role: translator, pattern: 'translated by (?P<name>[a-z]+)'}
  - {role: reviser, pattern: 'revised by (?P<name>[a-z]+)'}
"""


@pytest.fixture
def ascii_data(tmp_path: Path) -> Path:
    (tmp_path / "lexicon").mkdir()
    (tmp_path / "lexicon" / "derge_markers.yaml").write_text(ASCII_MARKERS, encoding="utf-8")
    return tmp_path


def parse_lines(tmp_path: Path, data_dir: Path, *lines: str, toh=("D1",)):
    path = tmp_path / "volume.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return derge.parse(path, list(toh), WITNESS, data_dir)


def kinds(result) -> list[tuple[str, str, str | None]]:
    return [(s.text, s.kind, s.local_chapter) for s in result.segments]


def test_units_end_after_each_shad_and_empty_units_are_dropped(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data, "[1a]", f"[1a.1]{{D1}}a b{S}{S} {S}c{S}")
    assert [(s.id, s.text) for s in r.segments] == [("D1:1a.1.1", f"a b{S}"), ("D1:1a.1.2", f"c{S}")]


def test_a_unit_broken_across_lines_keeps_both_coordinates(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data, "[1a.1]{D1}a b ", f"[1a.2]c d{S}")
    (seg,) = r.segments
    assert (seg.text, seg.start, seg.end) == (f"a b c d{S}", "1a.1", "1a.2")


def test_front_matter_ends_at_the_text_proper(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data, f"[1a.1]{{D1}}TITLE x{S}homage{S}THUS heard{S}y{S}")
    assert kinds(r) == [(f"TITLE x{S}", "meta", None), (f"homage{S}", "meta", None),
                        (f"THUS heard{S}", "prose", "D1:1"), (f"y{S}", "prose", "D1:1")]


def test_front_matter_needs_a_title_near_the_start_and_is_capped(tmp_path, ascii_data):
    late = parse_lines(tmp_path, ascii_data, f"[1a.1]{{D1}}a{S}b{S}TITLE{S}c{S}")
    assert all(s.kind == "prose" for s in late.segments)
    capped = parse_lines(tmp_path, ascii_data, f"[1a.1]{{D1}}TITLE{S}a{S}b{S}c{S}d{S}")
    assert [s.kind for s in capped.segments] == ["meta", "meta", "meta", "prose", "prose"]


def test_colophons_close_chapters_and_ordinals_are_cross_checked(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data,
                    f"[1a.1]{{D1}}a{S}chapter one ends{S}b{S}chapter three ends{S}c{S}fin END{S}")
    assert kinds(r) == [(f"a{S}", "prose", "D1:1"), (f"chapter one ends{S}", "colophon", "D1:1"),
                        (f"b{S}", "prose", "D1:2"), (f"chapter three ends{S}", "colophon", "D1:2"),
                        (f"c{S}", "prose", "D1:3"), (f"fin END{S}", "colophon", "D1:3")]
    assert [(c["ordinal"], c["text_end"]) for c in r.metadata["chapters"]] == [(1, False), (3, False), (None, True)]
    assert r.report["chapter_ordinal_mismatches"] == ["D1:2"]


def test_only_the_last_text_end_starts_the_paratext(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data,
                    f"[1a.1]{{D1}}a END b{S}c{S}d END{S}translated by pema{S}revised by dawa{S}",
                    f"[1a.2]{{D2}}next{S}")
    assert kinds(r) == [(f"a END b{S}", "prose", "D1:1"), (f"c{S}", "prose", "D1:1"),
                        (f"d END{S}", "colophon", "D1:1"), (f"translated by pema{S}", "paratext", None),
                        (f"revised by dawa{S}", "paratext", None)]
    assert r.metadata["translators"] == [{"role": "translator", "name": "pema", "segment_id": "D1:1a.1.4"}]
    assert (r.metadata["reviser"], r.metadata["revision_statement"]) == ("dawa", f"revised by dawa{S}")
    assert r.report["paratext"] == 2


def test_without_a_text_end_there_is_no_paratext(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data, f"[1a.1]{{D1}}a{S}revised by dawa{S}")
    assert [s.kind for s in r.segments] == ["prose", "prose"]
    assert r.metadata["reviser"] is None and r.metadata["paratext"] == []


def test_verse_and_mantra_kinds(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data,
                    f"[1a.1]{{D1}}a b c d e f g{S}a b c d e f g h{S}{OM} {OM} {OM}{S}{OM} x y z{S}")
    assert [s.kind for s in r.segments] == ["verse_line", "prose", "mantra", "prose"]


def test_edit_marks_and_text_starts(tmp_path, ascii_data):
    r = parse_lines(tmp_path, ascii_data, f"[1a.1]{{D1}}a {{bb,b}} c{S}{{D2}}{{xx,x}}{S}", f"[1a.2]{{D3}}z{S}",
                    toh=("D1", "D3"))
    assert [(s.id, s.text) for s in r.segments] == [("D1:1a.1.1", f"a b c{S}"), ("D3:1a.2.1", f"z{S}")]
    assert [(v.reading, v.alternative, v.segment_id) for v in r.variants] == [("b", "bb", "D1:1a.1.1")]


@pytest.mark.parametrize("lines, message", [
    ((f"[1a.1]{{D1}}a {{b{S}",), "brace"),
    ((f"[1a.1]{{D1}}a b}}{S}",), "brace"),
    ((f"[1a.1]{{D1}}a{S}", "[1b] stray text"), "bare folio marker"),
    ((f"[1a.1]{{D2}}a{S}",), "no text start"),
    ((f"[1a.1]{{D1}}a{S}{{D1}}b{S}",), "second start"),
])
def test_malformed_volumes_are_rejected(tmp_path, ascii_data, lines, message):
    with pytest.raises(ValueError, match=message):
        parse_lines(tmp_path, ascii_data, *lines)


def test_requested_texts_must_be_distinct(tmp_path, ascii_data):
    with pytest.raises(ValueError, match="distinct"):
        parse_lines(tmp_path, ascii_data, f"[1a.1]{{D1}}a{S}", toh=("D1", "D1"))


@pytest.mark.parametrize("old, new, message", [
    ("(?P<ordinal>[a-z]+)", "([a-z]+)", "ordinal"),
    ("(?P<name>[a-z]+)'}\n  - {role: reviser", "([a-z]+)'}\n  - {role: reviser", "name"),
    ("text_end: 'END'\n", "", "expected keys"),
    ("max_units: 3", "max_units: x", "malformed"),
])
def test_malformed_marker_files_are_rejected(tmp_path, old, new, message):
    (tmp_path / "lexicon").mkdir()
    assert old in ASCII_MARKERS
    (tmp_path / "lexicon" / "derge_markers.yaml").write_text(ASCII_MARKERS.replace(old, new), encoding="utf-8")
    with pytest.raises(LexiconError, match=message):
        derge.load_markers(tmp_path)
