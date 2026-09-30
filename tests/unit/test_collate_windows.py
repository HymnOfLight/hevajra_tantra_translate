"""collate.windows: chunking, core window, handles and the Window invariants."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hevajra_matrix.collate.windows import Window, WindowParams, chunk_bounds, plan, prompt_kind
from hevajra_matrix.config import ConfigError

from test_collate_support import concordance, reference, window, windows, witness


# --------------------------------------------------------------------------- chunk_bounds
@pytest.mark.parametrize("n, size, overlap, expected", [
    (0, 150, 10, []),
    (5, 150, 10, [(0, 5)]),
    (150, 150, 10, [(0, 150)]),
    (10, 4, 1, [(0, 4), (3, 7), (6, 10)]),
    (11, 4, 1, [(0, 4), (3, 7), (6, 9), (8, 11)]),
    (310, 150, 10, [(0, 110), (100, 210), (200, 310)]),
])
def test_chunk_bounds_examples(n: int, size: int, overlap: int, expected: list[tuple[int, int]]) -> None:
    assert chunk_bounds(n, size, overlap) == expected


@pytest.mark.parametrize("n", range(1, 400, 7))
@pytest.mark.parametrize("size, overlap", [(150, 10), (40, 5), (7, 0), (3, 2)])
def test_chunk_bounds_invariants(n: int, size: int, overlap: int) -> None:
    bounds = chunk_bounds(n, size, overlap)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(b - a <= size for a, b in bounds)
    for (a1, b1), (a2, b2) in zip(bounds, bounds[1:]):
        assert b1 - a2 == overlap, "consecutive chunks share exactly `overlap` units"
        assert a2 > a1
    # fewest chunks: one chunk fewer could not cover n units
    k = len(bounds)
    assert k == 1 or (k - 1) * size - (k - 2) * overlap < n


    sizes = [b - a for a, b in bounds]
    assert max(sizes) - min(sizes) <= 1, "chunks are balanced"


def test_chunk_bounds_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_bounds(10, 5, 5)


# --------------------------------------------------------------------------- params
def test_window_params_from_config_and_validation() -> None:
    assert WindowParams.from_config({"windows": {"max_ref_units": 150, "overlap": 10, "neighbours": 1}}) \
        == WindowParams(150, 10, 1)
    with pytest.raises(ConfigError):
        WindowParams.from_config({"windows": {"max_units": 3}})
    with pytest.raises(ConfigError):
        WindowParams(10, 10, 0)
    with pytest.raises(ConfigError):
        WindowParams(10, -1, 0)
    with pytest.raises(ConfigError):
        WindowParams(True, 0, 0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- plan
def test_plan_one_window_per_small_chapter_in_reference_order() -> None:
    ws = windows()
    assert [w.key for w in ws] == ["I.1.w1", "I.2.w1"]
    assert [u.id for u in ws[0].units] == [s.id for s in reference() if s.chapter == "I.1" and s.kind != "colophon"]
    assert all(w.overlap_next == 0 for w in ws)


def test_plan_chunks_with_overlap() -> None:
    ws = [w for w in windows(max_ref_units=4, overlap=2) if w.chapter == "I.1"]
    assert [len(w.units) for w in ws] == [4, 4]
    assert ws[0].overlap_next == 2 and ws[1].overlap_next == 0
    assert ws[0].overlap_ids == tuple(u.id for u in ws[1].units[:2])


def test_witness_text_is_the_whole_text_with_notes_but_no_front_matter() -> None:
    w = window()
    kinds = {s.kind for s in w.text}
    assert "note" in kinds and "head" in kinds
    assert not kinds & {"meta", "paratext", "colophon"}
    assert len(w.text) == sum(1 for s in witness() if s.kind not in {"meta", "paratext"})
    assert all(ww.text == w.text for ww in windows()), "every window shows the same witness text"


def test_handles_are_sequential_over_the_whole_text() -> None:
    w1, w2 = windows()
    assert list(w1.wit_handles) == [f"z{i:04d}" for i in range(1, len(w1.text) + 1)]
    assert w1.wit_handles == w2.wit_handles, "a handle names the same segment in every window"
    assert list(w1.ref_handles) == [f"r{i:03d}" for i in range(1, len(w1.units) + 1)]
    assert list(w2.ref_handles)[0] == "r001", "reference handles restart in every chunk"
    for handle, sid in (*w1.wit_handles.items(), *w1.ref_handles.items()):
        assert w1.handle_of[sid] == handle


def test_core_window_follows_the_concordance() -> None:
    w = window()
    assert w.core_locals == frozenset({"pin1"})
    assert w.core == frozenset(s.id for s in w.text if s.local_chapter == "pin1")
    assert w.core_ranges() == [("z0001", "z0009")]
    wide = window(neighbours=1)
    assert wide.core_locals == frozenset({"pin1", "pin2"})


def test_core_ranges_of_a_split_window() -> None:
    conc = concordance({"I.1": ["pin1", "pin3"], "I.2": ["pin2"]})
    w = window(conc=conc)
    ranges = w.core_ranges()
    assert len(ranges) == 2 and ranges[0] == ("z0001", "z0009")


def test_plan_requires_reference_chapters() -> None:
    units = [replace(s, chapter=None) for s in reference()]
    with pytest.raises(ValueError, match="reference chapter"):
        plan(units, witness(), concordance(), WindowParams())


def test_plan_requires_both_texts() -> None:
    with pytest.raises(ValueError):
        plan([], witness(), concordance(), WindowParams())


# --------------------------------------------------------------------------- Window invariants
def test_window_rejects_a_second_witness_or_language() -> None:
    w = window()
    stranger = replace(w.text[1], id="XX:0001a01.1", witness="other")
    with pytest.raises(ValueError, match="one witness"):
        replace(w, text=(*w.text, stranger))
    sanskrit = replace(w.units[0], id="BT:9a.1.1", lang="sa")
    with pytest.raises(ValueError, match="one language"):
        replace(w, units=(*w.units, sanskrit))


def test_window_rejects_non_alignable_units_and_foreign_kinds() -> None:
    w = window()
    colophon = next(s for s in reference() if s.kind == "colophon")
    with pytest.raises(ValueError, match="alignable"):
        replace(w, units=(*w.units, colophon))
    meta = next(s for s in witness() if s.kind == "meta")
    with pytest.raises(ValueError, match="witness kinds"):
        replace(w, text=(meta, *w.text))


def test_window_rejects_units_of_another_chapter_and_duplicates() -> None:
    w1, w2 = windows()
    with pytest.raises(ValueError):
        replace(w1, units=(*w1.units, w2.units[0]))
    with pytest.raises(ValueError, match="duplicate"):
        replace(w1, text=(*w1.text, w1.text[0]))
    with pytest.raises(ValueError, match="overlap_next"):
        replace(w1, overlap_next=len(w1.units))


def test_window_is_immutable_and_segment_lookup_works() -> None:
    w = window()
    with pytest.raises(AttributeError):
        w.key = "x"  # type: ignore[misc]
    assert w.segment[w.units[0].id] is w.units[0]
    assert w.position[w.text[3].id] == 3
    assert isinstance(w, Window)


def test_prompt_kind_maps_verse_lines() -> None:
    assert prompt_kind("verse_line") == "verse"
    assert prompt_kind("prose") == "prose"
