"""Unit tests of hevajra_matrix.ingest.cbeta.

Two kinds of input:
    * data/fixtures/mini_cbeta.xml with the real note rules (data/lexicon/notes_zh.yaml):
      every structure of T0892 in miniature (notes of all seven classes, footnotes with and
      without the manuscript marker, the apparatus, glosses, dharani, verse lines).
    * small ASCII documents built here with an ASCII rule file in a temporary data
      directory, to pin down one rule at a time. Clause punctuation is written as escapes
      (the language policy keeps CJK characters out of .py files).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hevajra_matrix.core.ids import parse_segment_id
from hevajra_matrix.core.textnorm import fingerprint
from hevajra_matrix.ingest import CONTENT_KINDS, cbeta
from hevajra_matrix.ingest import notes as note_rules

COMMA = "\uff0c"        # fullwidth comma (clause end)
STOP = "\u3002"         # ideographic full stop (clause end)
WITNESS = "zh_T0892_song"
DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def mini():
    return cbeta.parse(DATA / "fixtures" / "mini_cbeta.xml", WITNESS, DATA)


def by_id(result):
    return {s.id: s for s in result.segments}


# --------------------------------------------------------------------------- fixture: segments
def test_segments_in_document_order_with_kinds_and_chapters(mini):
    got = [(s.id, s.kind, s.local_chapter) for s in mini.segments]
    assert got == [
        ("T0892:0587c04.1", "meta", None),
        ("T0892:0587c05.1", "meta", None),
        ("T0892:0587c06.n1", "note", None),
        ("T0892:0587c08.1", "meta", None),
        ("T0892:0587c10.1", "head", "pin1"),
        ("T0892:0587c11.1", "prose", "pin1"),
        ("T0892:0587c11.2", "prose", "pin1"),
        ("T0892:0587c13.1", "prose", "pin1"),
        ("T0892:0587c13.2", "prose", "pin1"),
        ("T0892:0587c13.3", "prose", "pin1"),
        ("T0892:0587c13.4", "prose", "pin1"),
        ("T0892:0587c14.1", "prose", "pin1"),
        ("T0892:0587c14.n1", "note", "pin1"),
        ("T0892:0587c14.2", "prose", "pin1"),
        ("T0892:0587c15.1", "verse_line", "pin1"),
        ("T0892:0587c15.2", "verse_line", "pin1"),
        ("T0892:0587c16.1", "verse_line", "pin1"),
        ("T0892:0587c16.2", "verse_line", "pin1"),
        ("T0892:0587c17.1", "head", "pin2"),
        ("T0892:0587c18.1", "mantra", "pin2"),
        ("T0892:0587c18.n1", "note", "pin2"),
        ("T0892:0587c18.n2", "note", "pin2"),
        ("T0892:0587c18.n3", "note", "pin2"),
        ("T0892:0587c19.n1", "note", "pin2"),
        ("T0892:0587c19.n2", "note", "pin2"),
        ("T0892:0587c20.1", "prose", "pin2"),
        ("T0892:0587c20.2", "prose", "pin2"),
        ("T0892:0587c20.n1", "note", "pin2"),
        ("T0892:0587c20.3", "prose", "pin2"),
        ("T0892:0587c21.1", "meta", "pin2"),
        ("T0892:0587c22.1", "meta", "pin2"),
        ("T0892:0587c22.n1", "note", "pin2"),
        ("T0892:0587c23.1", "meta", "pin2"),
        ("T0892:0587c24.1", "prose", "pin2"),
    ]


def test_every_segment_has_a_valid_id_fingerprint_and_witness(mini):
    for s in mini.segments:
        assert str(parse_segment_id(s.id)) == s.id
        assert s.fingerprint == fingerprint(s.text, "zh")
        assert (s.witness, s.lang, s.chapter) == (WITNESS, "zh", None)
        assert s.text == s.text.strip() and s.text


def test_segments_spanning_lines_keep_start_and_end(mini):
    seg = by_id(mini)
    assert (seg["T0892:0587c11.2"].start, seg["T0892:0587c11.2"].end) == ("0587c11", "0587c13")
    assert (seg["T0892:0587c05.1"].start, seg["T0892:0587c05.1"].end) == ("0587c05", "0587c06")


def test_dharani_is_one_mantra_segment_and_rare_characters_are_kept(mini):
    seg = by_id(mini)
    assert seg["T0892:0587c18.1"].end == "0587c19"
    assert "\U00021060" in seg["T0892:0587c14.2"].text      # CB00768 given by a <g> element


# --------------------------------------------------------------------------- fixture: notes
def test_notes_have_class_and_host_and_follow_it(mini):
    notes = {s.id: (s.extra["note_class"], s.extra["host"]) for s in mini.segments if s.kind == "note"}
    assert notes == {
        "T0892:0587c06.n1": ("title", "T0892:0587c05.1"),
        "T0892:0587c14.n1": ("substitution_instruction", "T0892:0587c14.1"),
        "T0892:0587c18.n1": ("phonetic", "T0892:0587c18.1"),
        "T0892:0587c18.n2": ("mantra_numbering", "T0892:0587c18.1"),
        "T0892:0587c18.n3": ("phonetic", "T0892:0587c18.1"),
        "T0892:0587c19.n1": ("mantra_numbering", "T0892:0587c18.1"),
        "T0892:0587c19.n2": ("fanqie", "T0892:0587c18.1"),
        "T0892:0587c20.n1": ("vorlage_statement", "T0892:0587c20.2"),
        "T0892:0587c22.n1": ("fascicle", "T0892:0587c22.1"),
    }
    order = [s.id for s in mini.segments]
    for s in mini.segments:
        if s.kind == "note":
            between = order[order.index(s.extra["host"]) + 1:order.index(s.id)]
            assert all(by_id(mini)[i].kind == "note" for i in between), s.id


def test_note_text_never_enters_reading_text(mini):
    content = [s.text for s in mini.segments if s.kind != "note"]
    for note in (s for s in mini.segments if s.kind == "note"):
        assert not any(note.text in text for text in content if len(note.text) > 1), note.id


def test_line_break_inside_a_note_advances_the_line(mini):
    seg = by_id(mini)
    split_note = seg["T0892:0587c18.n3"]
    assert (split_note.start, split_note.end) == ("0587c18", "0587c19")
    assert "\n" not in split_note.text and len(split_note.text) == 2
    assert seg["T0892:0587c19.n1"].start == "0587c19"


def test_note_report(mini):
    assert mini.report["notes"] == 9
    assert mini.report["note_classes"] == {"fanqie": 1, "fascicle": 1, "mantra_numbering": 2, "phonetic": 2,
                                           "substitution_instruction": 1, "title": 1, "vorlage_statement": 1}
    assert mini.report["unclassified_notes"] == []


# --------------------------------------------------------------------------- fixture: back matter
def test_all_taisho_footnotes_are_kept_at_their_anchor(mini):
    got = [(f.n, f.locus, f.segment_id, f.source) for f in mini.footnotes]
    assert got == [
        ("0587005", "0587c05", "T0892:0587c05.1", "taisho"),
        ("0587006", "0587c11", "T0892:0587c11.2", "taisho"),
        ("0587007", "0587c16", "T0892:0587c16.1", "tokyo335"),
        ("0587008", "0587c17", "T0892:0587c17.1", "taisho"),
        ("0587009", "0587c18", "T0892:0587c18.1", "taisho"),
    ]
    assert mini.report["footnote_sources"] == {"taisho": 4, "tokyo335": 1}


def test_footnote_lemma_and_sanskrit_are_split_verbatim(mini):
    notes = {f.n: f for f in mini.footnotes}
    assert notes["0587006"].sa_text == "Sarvatathā-gata-kāya-vāk-citta-vajra-yogirbhageṣu."
    assert notes["0587006"].zh_lemma and notes["0587006"].text.startswith(notes["0587006"].zh_lemma)
    assert notes["0587005"].sa_text == "Hevajra-tantra" and not notes["0587005"].zh_lemma.endswith(",")
    assert notes["0587008"].sa_text == "" and notes["0587008"].zh_lemma == notes["0587008"].text
    assert notes["0587009"].zh_lemma == "" and notes["0587009"].sa_text == "Oṁ ā hūṁ phaṭ svāhā."
    assert notes["0587007"].sa_text.startswith("viyoga.")          # the marker stays in the text


def test_cbeta_modifications_and_foreign_notes_are_not_footnotes(mini):
    assert len(mini.footnotes) == 5
    assert not any("modification" in f.text for f in mini.footnotes)


def test_apparatus_becomes_variants_without_nested_notes(mini):
    (variant,) = mini.variants
    assert variant.kind == "apparatus"
    assert (variant.segment_id, variant.locus) == ("T0892:0587c14.2", "0587c14")
    assert len(variant.reading) == 1 and len(variant.alternative) == 1
    assert variant.reading in by_id(mini)[variant.segment_id].text


def test_glosses_and_chapters_in_metadata(mini):
    (gloss,) = mini.metadata["glosses"]
    assert gloss["sa"] == "Sarvatathā-gata-kāya-vāk-citta-vajra-yogirbhageṣu"
    assert (gloss["locus"], gloss["segment_id"]) == ("0587c11", "T0892:0587c11.2")
    assert [c["local"] for c in mini.metadata["chapters"]] == ["pin1", "pin2"]
    assert mini.metadata["prefix"] == "T0892"


def test_metadata_and_report_are_json_ready(mini):
    json.loads(json.dumps({"metadata": mini.metadata, "report": mini.report}, ensure_ascii=False))


def test_report_counts(mini):
    assert mini.report["segments"] == len(mini.segments) == 34
    assert mini.report["content_segments"] == sum(s.kind in CONTENT_KINDS for s in mini.segments) == 17
    assert mini.report["kinds"] == {"head": 2, "mantra": 1, "meta": 6, "note": 9, "prose": 12, "verse_line": 4}
    assert (mini.report["chapters"], mini.report["variants"], mini.report["footnotes"]) == (2, 1, 5)
    assert mini.report["duplicate_ids"] == []


# --------------------------------------------------------------------------- synthetic ASCII documents
ASCII_RULES = """\
rules:
  - {class: vorlage_statement, pattern: ".*VORLAGE.*"}
  - {class: phonetic, pattern: "y+"}
  - {class: mantra_numbering, pattern: "[0-9]+"}
tokyo335_marker: "(MS)"
"""


@pytest.fixture
def ascii_data(tmp_path: Path) -> Path:
    (tmp_path / "lexicon").mkdir()
    (tmp_path / "lexicon" / "notes_zh.yaml").write_text(ASCII_RULES, encoding="utf-8")
    return tmp_path


def tei(body: str, back: str = "", text_id: str = "T18n0892") -> str:
    return (f'<TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:cb="http://www.cbeta.org/ns/1.0" '
            f'xml:id="{text_id}"><text><body>{body}</body><back>{back}</back></text></TEI>')


def parse_ascii(tmp_path: Path, data_dir: Path, body: str, back: str = "", text_id: str = "T18n0892"):
    path = tmp_path / "doc.xml"
    path.write_text(tei(body, back, text_id), encoding="utf-8")
    return cbeta.parse(path, WITNESS, data_dir)


def chapter(n: int, content: str) -> str:
    return f'<cb:div type="pin"><cb:mulu n="{n}" level="1">x</cb:mulu>{content}</cb:div>'


def test_prose_is_cut_after_clause_punctuation(tmp_path, ascii_data):
    body = chapter(1, f'<lb n="0001a01"/><p>aa{COMMA}bb{STOP}{COMMA}cc</p>')
    r = parse_ascii(tmp_path, ascii_data, body)
    assert [s.text for s in r.segments] == [f"aa{COMMA}", f"bb{STOP}{COMMA}", "cc"]
    assert [s.id for s in r.segments] == ["T0892:0001a01.1", "T0892:0001a01.2", "T0892:0001a01.3"]


def test_punctuation_only_pieces_are_dropped(tmp_path, ascii_data):
    body = chapter(1, f'<lb n="0001a01"/><p>aa{COMMA} {STOP}</p><p>{COMMA}</p>')
    r = parse_ascii(tmp_path, ascii_data, body)
    assert [s.text for s in r.segments] == [f"aa{COMMA}"]


def test_note_hosts_the_segment_before_it_across_blocks(tmp_path, ascii_data):
    body = chapter(1, f'<lb n="0001a01"/><p>aa{COMMA}<note place="inline">yy</note>bb</p>'
                      f'<lb n="0001a02"/><p><note place="inline">7</note>cc</p>')
    r = parse_ascii(tmp_path, ascii_data, body)
    got = [(s.id, s.extra.get("host")) for s in r.segments]
    assert got == [("T0892:0001a01.1", None), ("T0892:0001a01.n1", "T0892:0001a01.1"),
                   ("T0892:0001a01.2", None), ("T0892:0001a02.n1", "T0892:0001a01.2"),
                   ("T0892:0001a02.1", None)]


def test_note_before_any_text_hosts_the_first_segment(tmp_path, ascii_data):
    body = chapter(1, '<lb n="0001a01"/><note place="inline">yy</note><p>aa</p>')
    r = parse_ascii(tmp_path, ascii_data, body)
    assert [(s.id, s.extra.get("host")) for s in r.segments] == [
        ("T0892:0001a01.1", None), ("T0892:0001a01.n1", "T0892:0001a01.1")]


def test_notes_do_not_renumber_content_segments(tmp_path, ascii_data):
    plain = parse_ascii(tmp_path, ascii_data, chapter(1, f'<lb n="0001a01"/><p>aa{COMMA}bb</p>'))
    noted = parse_ascii(tmp_path, ascii_data,
                        chapter(1, f'<lb n="0001a01"/><p>aa<note place="inline">yy</note>{COMMA}bb</p>'))
    assert [s.id for s in plain.segments] == [s.id for s in noted.segments if s.kind != "note"]
    assert [s.text for s in plain.segments] == [s.text for s in noted.segments if s.kind != "note"]


def test_unknown_note_is_reported_unclassified(tmp_path, ascii_data):
    body = chapter(1, '<lb n="0001a01"/><p>aa<note place="inline">zzz</note></p>')
    r = parse_ascii(tmp_path, ascii_data, body)
    assert r.report["unclassified_notes"] == ["T0892:0001a01.n1"]
    assert r.segments[1].extra["note_class"] == note_rules.UNCLASSIFIED


def test_outside_chapter_divisions_local_chapter_is_none(tmp_path, ascii_data):
    body = ('<lb n="0001a01"/><cb:juan><cb:jhead>title</cb:jhead></cb:juan>'
            + chapter(3, '<lb n="0001a02"/><head>three</head>') + '<lb n="0001a03"/><byline>by</byline>')
    r = parse_ascii(tmp_path, ascii_data, body)
    assert [(s.kind, s.local_chapter) for s in r.segments] == [("meta", None), ("head", "pin3"), ("meta", None)]


def test_footnote_source_and_anchor_host(tmp_path, ascii_data):
    body = chapter(1, f'<lb n="0001a01"/><p>aa{COMMA}<anchor xml:id="nkr_note_orig_0001001"/>bb</p>')
    back = ('<note n="0001001" type="orig" place="foot text" target="#nkr_note_orig_0001001">bb vana (MS)</note>'
            '<note n="0001001" type="mod" target="#nkr_note_mod_0001001">ignored</note>')
    r = parse_ascii(tmp_path, ascii_data, body, back)
    (note,) = r.footnotes
    assert (note.locus, note.segment_id, note.zh_lemma, note.sa_text, note.source) == (
        "0001a01", "T0892:0001a01.2", "", "bb vana (MS)", "tokyo335")


def test_footnote_without_anchor_is_an_error(tmp_path, ascii_data):
    back = '<note n="0001001" type="orig" target="#nkr_note_orig_0001001">x</note>'
    with pytest.raises(ValueError, match="not an anchor"):
        parse_ascii(tmp_path, ascii_data, chapter(1, '<lb n="0001a01"/><p>aa</p>'), back)


def test_bad_text_id_or_chapter_number_is_an_error(tmp_path, ascii_data):
    with pytest.raises(ValueError, match="xml:id"):
        parse_ascii(tmp_path, ascii_data, chapter(1, '<lb n="0001a01"/><p>aa</p>'), text_id="nonsense")
    with pytest.raises(ValueError, match="cb:mulu"):
        parse_ascii(tmp_path, ascii_data, '<cb:div type="pin"><lb n="0001a01"/><p>aa</p></cb:div>')
    with pytest.raises(ValueError, match="no text"):
        parse_ascii(tmp_path, ascii_data, '<lb n="0001a01"/>')
