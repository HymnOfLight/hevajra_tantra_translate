from hevajra_matrix.anchors import anchor_similarity, load_lexicon
from hevajra_matrix.ingest.cbeta import parse_cbeta_tei
from hevajra_matrix.ingest.derge import parse_derge_volume
from hevajra_matrix.ingest.sanskrit import load_reference_tsv


def test_cbeta_segments_coordinates_and_chapters(fixtures):
    doc = parse_cbeta_tei(fixtures / "mini_cbeta.xml")
    assert doc.title_note == "大幻化普通儀軌三十一分中略出二無我法"
    assert [c["n"] for c in doc.chapters] == [1, 2]
    kinds = [s.kind for s in doc.segments]
    assert "head" in kinds and "verse_line" in kinds and "prose" in kinds
    prose = [s for s in doc.segments if s.kind == "prose"]
    assert prose[0].text == "如是我聞：" and prose[0].start == "0587c11"
    # a clause spanning two Taishō lines keeps both coordinates
    span = next(s for s in prose if "喻施婆倪數" in s.text)
    assert span.start == "0587c11" and span.end == "0587c13"
    # inline phonetic gloss is stripped from the text but counted
    n32 = next(s for s in prose if "三十二" in s.text)
    assert "引" not in n32.text and n32.extra.get("inline_notes") == 1
    # verse lines split at the caesura comma → hemistichs
    verse = [s.text for s in doc.segments if s.kind == "verse_line"]
    assert verse == ["「最初觀於慈，", "次即觀於悲，", "第三當觀喜，", "一切處學捨。」"]
    assert doc.segments[-1].local_chapter == 2
    assert doc.glosses[0]["sa"].startswith("Sarvatathāgata")
    assert doc.apparatus[0]["lem"] == "晝" and doc.apparatus[0]["rdg"] == ["畫"]


def test_derge_slicing_chapters_and_front_matter(fixtures):
    texts = parse_derge_volume(fixtures / "mini_derge.txt", ["D417", "D418"])
    assert [t.toh for t in texts] == ["D417", "D418"]
    d417 = texts[0]
    metas = [s for s in d417.segments if s.kind == "meta"]
    assert metas and all("ཐོས་པ" not in s.text for s in metas)
    nidana = next(s for s in d417.segments if "འདི་སྐད་བདག་གིས་ཐོས་པ" in s.text)
    assert nidana.kind != "meta" and nidana.start == "1b.1" and nidana.end == "1b.2"
    assert [c["ordinal"] for c in d417.chapters] == [1, 2, None]
    assert d417.chapters[-1]["text_end"] is True
    assert any(s.kind == "mantra" for s in d417.segments)
    assert "D419" not in "".join(s.text for s in texts[1].segments)
    assert texts[1].chapters[0]["ordinal"] == 1


def test_sanskrit_tsv_flags(fixtures):
    a = load_reference_tsv(fixtures / "sa_edition_a.tsv")
    b = load_reference_tsv(fixtures / "sa_edition_b.tsv")
    assert [s.seg_id for s in a] == ["I.1.p01", "I.1.p02", "I.1.1", "I.1.2", "I.2.1", "II.1.1"]
    assert a[0].chapter == "I.1" and a[-1].chapter == "II.1"
    flags = {s.seg_id: s.extra.get("flag") for s in b}
    assert flags["I.1.2"] == "ABSENT" and flags["I.2.1"] == "LACUNA"


def test_anchor_lexicon_cross_lingual():
    lex = load_lexicon()
    zh = lex.extract("金剛藏菩薩摩訶薩！彼血脈相有三十二種", "zh")
    bo = lex.extract("རྡོ་རྗེ་སྙིང་པོ་སྙིང་རྗེ་ཆེན་པོ་ལེགས་སོ། རྩ་ནི་སུམ་ཅུ་རྩ་གཉིས་ཏེ", "bo")
    sa = lex.extract("vajragarbha dvātriṃśan nāḍyaḥ", "sa")
    assert "name:vajragarbha" in zh and "name:vajragarbha" in bo and "name:vajragarbha" in sa
    assert "num:32" in zh and "num:32" in bo and "num:32" in sa
    assert anchor_similarity(zh, bo) > 0.5
    assert anchor_similarity(set(), set()) == 0.0
    # transliteration density: transliterated name vs translated clause
    assert lex.transliteration_density("拏吉尼羅羅拏辣娑拏阿嚩底", "zh") > 0.8
    assert lex.transliteration_density("佛告金剛藏菩薩", "zh") < 0.2
