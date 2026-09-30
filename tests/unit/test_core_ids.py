"""core.ids: coordinate-derived ids, their stability, parsing and ordering."""

from __future__ import annotations

import random

import pytest

from hevajra_matrix.core import ids
from hevajra_matrix.core.ids import (
    REF_CHAPTERS,
    SegmentRef,
    assign_ids,
    chapter_key,
    id_kind,
    note_id,
    orphan_row_id,
    orphan_source,
    parse_segment_id,
    parse_snellgrove_id,
    segment_id,
    sort_key,
)


# --------------------------------------------------------------------------- construction
def test_spec_examples():
    assert segment_id("T0892", "0592a29", 1) == "T0892:0592a29.1"
    assert segment_id("D417", "8a.4", 3) == "D417:8a.4.3"
    assert note_id("T0892", "0592a27", 1) == "T0892:0592a27.n1"
    assert orphan_row_id("T0892:0601c01.2") == "+T0892:0601c01.2"


@pytest.mark.parametrize("prefix,line,ordinal", [
    ("", "0592a29", 1), ("T:0892", "0592a29", 1), ("1T", "0592a29", 1), ("+T0892", "0592a29", 1),
    ("T0892", "", 1), ("T0892", "0592 a29", 1), ("T0892", "8a..4", 1), ("T0892", ".8a", 1),
    ("T0892", "8a.", 1), ("T0892", "0592a29", 0), ("T0892", "0592a29", -1), ("T0892", "0592a29", True),
    ("T0892", "0592a29", 1.0),
])
def test_bad_components_are_rejected(prefix, line, ordinal):
    with pytest.raises(ValueError):
        segment_id(prefix, line, ordinal)
    with pytest.raises(ValueError):
        note_id(prefix, line, ordinal)


@pytest.mark.parametrize("sid", [
    "T0892:0592a29.1", "D417:8a.4.3", "T0892:0592a27.n1", "D418:13b.5.n12", "sa_ms_C:12v3.2",
])
def test_segment_ids_round_trip(sid):
    ref = parse_segment_id(sid)
    assert str(ref) == sid
    assert (note_id if ref.is_note else segment_id)(ref.prefix, ref.line, ref.ordinal) == sid


def test_parse_segment_id_fields():
    assert parse_segment_id("D417:8a.4.3") == SegmentRef("D417", "8a.4", 3, False)
    assert parse_segment_id("T0892:0592a27.n2") == SegmentRef("T0892", "0592a27", 2, True)


@pytest.mark.parametrize("bad", [
    "T0892:0592a29", "T0892:0592a29.01", "T0892:0592a29.n0", "T0892:0592a29.0", "+T0892:0592a29.1",
    "T0892:0592a29.x1", "T0892 :0592a29.1", "0592a29.1", "I.5.12", "",
])
def test_parse_segment_id_rejects(bad):
    with pytest.raises(ValueError):
        parse_segment_id(bad)


def test_orphan_rows_on_one_line_are_distinct():
    """Defect #14: two witness-only blocks on one line used to collapse into one row."""
    first, second = assign_ids("T0892", [("0601c01", False), ("0601c01", False)])
    assert orphan_row_id(first) != orphan_row_id(second)
    assert orphan_source(orphan_row_id(second)) == second


@pytest.mark.parametrize("bad", ["I.5.12", "+T0892:0601c01.2", "T0892:0601c01"])
def test_orphan_row_id_needs_a_segment_id(bad):
    with pytest.raises(ValueError):
        orphan_row_id(bad)


@pytest.mark.parametrize("bad", ["T0892:0601c01.2", "+I.5.12", "+", "++T0892:0601c01.2"])
def test_orphan_source_rejects(bad):
    with pytest.raises(ValueError):
        orphan_source(bad)


# --------------------------------------------------------------------------- assign_ids
def test_assign_ids_numbers_content_and_notes_separately_per_line():
    entries = [("0592a27", False), ("0592a27", True), ("0592a27", False), ("0592a28", False), ("0592a27", True)]
    assert assign_ids("T0892", entries) == [
        "T0892:0592a27.1", "T0892:0592a27.n1", "T0892:0592a27.2", "T0892:0592a28.1", "T0892:0592a27.n2",
    ]


def test_inserting_a_note_changes_no_content_id_and_nothing_on_other_lines():
    before = [("1b.1", False), ("1b.2", False), ("1b.2", True), ("1b.2", False), ("1b.3", False)]
    after = before[:1] + [("1b.2", False), ("1b.2", True), ("1b.2", True), ("1b.2", False), ("1b.3", False)]
    old, new = assign_ids("D417", before), assign_ids("D417", after)
    content_old = [i for i, (_, note) in zip(old, before) if not note]
    content_new = [i for i, (_, note) in zip(new, after) if not note]
    assert content_old == content_new
    assert set(old) - set(new) == set()
    assert set(new) - set(old) == {"D417:1b.2.n2"}


def _random_document(rng: random.Random, n: int) -> list[tuple[str, bool]]:
    lines = [f"{p}{side}.{k}" for p in range(1, 4) for side in "ab" for k in range(1, 8)]
    starts = sorted(rng.choices(range(len(lines)), k=n))
    return [(lines[i], rng.random() < 0.2) for i in starts]


@pytest.mark.parametrize("seed", range(30))
def test_an_insertion_only_renames_ids_on_its_own_line(seed):
    rng = random.Random(seed)
    doc = _random_document(rng, 40)
    pos = rng.randrange(len(doc) + 1)
    line = doc[pos - 1][0] if pos else doc[0][0]
    inserted = (line, rng.random() < 0.5)
    new_doc = doc[:pos] + [inserted] + doc[pos:]
    old, new = assign_ids("D417", doc), assign_ids("D417", new_doc)
    assert len(set(new)) == len(new)                      # unique by construction
    kept_old = [i for i, (ln, _) in zip(old, doc) if ln != line]
    kept_new = [i for i, (ln, _) in zip(new, new_doc) if ln != line]
    assert kept_old == kept_new
    same_kind_old = [i for i, e in zip(old, doc) if e[0] == line and e[1] != inserted[1]]
    same_kind_new = [i for i, e in zip(new, new_doc) if e[0] == line and e[1] != inserted[1]]
    assert same_kind_old == same_kind_new                 # the other kind on that line is untouched


def test_assign_ids_validates_every_entry():
    with pytest.raises(ValueError):
        assign_ids("T0892", [("0592a27", False), ("bad line", False)])


# --------------------------------------------------------------------------- Snellgrove ids
def test_ref_chapters():
    assert len(REF_CHAPTERS) == 23
    assert REF_CHAPTERS[0] == "I.1" and REF_CHAPTERS[10] == "I.11" and REF_CHAPTERS[-1] == "II.12"
    assert isinstance(REF_CHAPTERS, tuple)


def test_no_global_scheme_registry():
    assert not hasattr(ids, "register_scheme")
    with pytest.raises(TypeError):
        ids.PART_CHAPTERS["L"] = 81  # read-only: no module-level state can be changed


@pytest.mark.parametrize("uid,chapter,prose,index,half", [
    ("I.5.12", "I.5", False, 12, ""), ("I.5.12a", "I.5", False, 12, "a"), ("II.3.p07", "II.3", True, 7, ""),
    ("I.1.p03", "I.1", True, 3, ""), ("II.12.p100", "II.12", True, 100, ""), ("I.11.999b", "I.11", False, 999, "b"),
])
def test_parse_snellgrove_id(uid, chapter, prose, index, half):
    u = parse_snellgrove_id(uid)
    assert (u.chapter_key, u.prose, u.index, u.half) == (chapter, prose, index, half)
    assert str(u) == uid
    assert chapter_key(uid) == chapter
    assert id_kind(uid) == "snellgrove"


@pytest.mark.parametrize("bad", [
    "I.12.1", "II.13.1", "III.1.1", "I.0.1", "I.1.0", "I.1.p00", "I.1.p3", "I.1.p003", "I.1.012", "I.01.1",
    "I.1.t0012", "I.1.p03a", "I.1.12c", "i.1.1", "I.1", "",
])
def test_parse_snellgrove_id_rejects(bad):
    with pytest.raises(ValueError):
        parse_snellgrove_id(bad)


def test_chapter_key_refuses_coordinate_ids():
    with pytest.raises(ValueError, match="Segment.chapter"):
        chapter_key("D417:8a.4.3")


@pytest.mark.parametrize("uid,kind", [
    ("T0892:0592a29.1", "segment"), ("T0892:0592a27.n1", "note"), ("+T0892:0601c01.2", "orphan"),
    ("+T0892:0592a27.n1", "orphan"), ("I.5.12a", "snellgrove"),
])
def test_id_kind(uid, kind):
    assert id_kind(uid) == kind


@pytest.mark.parametrize("bad", ["", "+", "T0892:", "I.5", "hello", "+I.5.12"])
def test_id_kind_rejects(bad):
    with pytest.raises(ValueError):
        id_kind(bad)


# --------------------------------------------------------------------------- ordering
def test_sort_key_orders_all_kinds():
    expected = [
        "I.1.1", "I.2.3", "I.2.p01", "I.5.12", "I.5.12a", "I.5.12b", "I.5.13", "I.11.1", "II.1.1", "II.12.p02",
        "D417:1b.7.1", "D417:1b.7.2", "D417:1b.7.n1", "D417:1b.10.1", "D417:2a.1.1", "D417:10a.1.1",
        "D418:1a.1.1",
        "T0892:0592a27.1", "T0892:0592a27.2", "T0892:0592a27.10", "T0892:0592a27.n1", "T0892:0592a28.1",
        "T0892:0592b01.1", "T0892:0593a01.1",
        "+D417:1b.7.2", "+T0892:0592a27.n1", "+T0892:0601c01.2",
    ]
    shuffled = expected[:]
    random.Random(7).shuffle(shuffled)
    assert sorted(shuffled, key=sort_key) == expected


def test_sort_key_rejects_invalid_ids():
    with pytest.raises(ValueError):
        sort_key("not an id")
