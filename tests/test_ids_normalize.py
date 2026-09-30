import pytest

from hevajra_matrix import normalize
from hevajra_matrix.ids import REF_CHAPTERS, make_orphan_id, parse_unit_id, sort_key


def test_ref_chapters_count():
    assert len(REF_CHAPTERS) == 23
    assert REF_CHAPTERS[0] == "I.1" and REF_CHAPTERS[-1] == "II.12"


@pytest.mark.parametrize("uid,kind,chapter", [
    ("I.5.12", "verse", "I.5"), ("I.5.12a", "verse", "I.5"), ("II.3.p07", "prose", "II.3"),
    ("I.1.t0012", "provisional", "I.1"),
])
def test_parse_unit_id(uid, kind, chapter):
    u = parse_unit_id(uid)
    assert u.kind == kind and u.chapter_key == chapter
    assert str(u) == uid


def test_parse_orphan_and_sort():
    o = parse_unit_id(make_orphan_id("zh_T0892_song", "0594c15", "0594c17"))
    assert o.kind == "orphan" and o.witness == "zh_T0892_song" and o.coords == "0594c15-0594c17"
    ids = ["II.1.1", "I.2.p01", "I.2.3", "+zh_T0892_song:0594c15", "I.1.1"]
    assert sorted(ids, key=sort_key) == ["I.1.1", "I.2.3", "I.2.p01", "II.1.1", "+zh_T0892_song:0594c15"]


def test_bad_ids():
    with pytest.raises(ValueError):
        parse_unit_id("I.12.1")  # part I has 11 chapters
    with pytest.raises(ValueError):
        parse_unit_id("III.1.1")


def test_zh_numerals_and_length():
    assert normalize.zh_parse_numeral("三十二") == 32
    assert normalize.zh_parse_numeral("十") == 10
    assert normalize.zh_parse_numeral("一百八") == 108
    assert 32 in normalize.zh_numerals("彼血脈相有三十二種")
    assert 1 not in normalize.zh_numerals("一切如來一時")  # lone 一 is not an anchor
    assert normalize.zh_length("「善哉，善哉。」") == 4


def test_bo_numerals_syllables_translit():
    assert normalize.bo_numerals("རྩ་ནི་སུམ་ཅུ་རྩ་གཉིས་ཏེ") == {32}
    assert normalize.bo_numerals("བཅོ་ལྔ་དང་བརྒྱ") == {15, 100}
    assert normalize.bo_length("རྡོ་རྗེ་སྙིང་པོས་གསོལ་བ།") == 6
    assert normalize.bo_sanskrit_syllable_ratio("ཨོཾ་ཧཱུྃ་ཕཊ") > 0.9
    assert normalize.bo_sanskrit_syllable_ratio("རྡོ་རྗེ་སྙིང་པོ") == 0.0


def test_sa_length_numerals():
    assert normalize.sa_length("evaṃ mayā śrutam") == 6
    assert 32 in normalize.sa_numerals("dvātriṃśan nāḍyaḥ")
