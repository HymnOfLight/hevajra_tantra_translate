"""Real-data checks of the ingest stage (G0 facts) on CBETA T0892 and Derge Toh 417-418.

Run with HEVAJRA_RAW_DIR pointing at the fetched files (T18n0892.xml,
derge_rgyud_bum_nga.txt). The expected numbers were re-checked on the raw files for the
v0.3 design (synthesis 0.2 and 3.2; critique 2026-09-30). Source-language quotes come from
data/sentinels/sentinels.yaml (the language policy keeps them out of .py files).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest
import yaml

from hevajra_matrix import registry
from hevajra_matrix.core.textnorm import quote_in
from hevajra_matrix.ingest import CONTENT_KINDS, cbeta, derge

pytestmark = pytest.mark.realdata

DATA = Path(__file__).resolve().parents[2] / "data"
ZH, BO = "zh_T0892_song", "bo_derge_D417_418"
SENTINELS = {s["id"].split("_")[0]: s for s in
             yaml.safe_load((DATA / "sentinels" / "sentinels.yaml").read_text(encoding="utf-8"))["sentinels"]}


@pytest.fixture(scope="module")
def zh():
    return cbeta.parse(Path(os.environ["HEVAJRA_RAW_DIR"]) / "T18n0892.xml", ZH, DATA)


@pytest.fixture(scope="module")
def bo():
    return derge.parse(Path(os.environ["HEVAJRA_RAW_DIR"]) / "derge_rgyud_bum_nga.txt", ["D417", "D418"], BO, DATA)


def on_line(result, locus: str, quote: str | None = None, kinds=None):
    """Segments whose id names the locus ("T0892:0592a27") and that contain the quote."""
    prefix, line = locus.split(":")
    lang = "zh" if prefix == "T0892" else "bo"
    return [s for s in result.segments
            if s.id.startswith(f"{prefix}:") and s.start == line and (kinds is None or s.kind in kinds)
            and (quote is None or quote_in(quote, s.text, lang))]


# --------------------------------------------------------------------------- CBETA
def test_521_notes_in_seven_classes(zh):
    assert zh.report["notes"] == 521
    assert zh.report["note_classes"] == {"phonetic": 344, "mantra_numbering": 162, "fanqie": 9,
                                         "substitution_instruction": 2, "fascicle": 2, "title": 1,
                                         "vorlage_statement": 1}
    assert zh.report["unclassified_notes"] == []


def test_reading_text_segmentation_is_unchanged_from_v02(zh):
    reading = [s for s in zh.segments if s.kind != "note"]
    print(f"\nT0892: {len(reading)} reading-text segments, {zh.report['content_segments']} content")
    assert len(reading) == 2613                      # v0.2 ingest output (synthesis 0.2)
    assert zh.report["kinds"]["head"] == 20


def test_vorlage_statement_0592a27(zh):
    s5a = SENTINELS["S5a"]
    (note,) = on_line(zh, s5a["loci"]["note"], s5a["quotes"]["note"], kinds={"note"})
    assert note.id == "T0892:0592a27.n1" and note.text == s5a["quotes"]["note"]
    assert note.extra == {"note_class": "vorlage_statement", "host": "T0892:0592a26.4"}


def test_181_footnotes_16_from_tokyo_335(zh):
    assert len(zh.footnotes) == 181 == len({f.n for f in zh.footnotes})
    assert Counter(f.source for f in zh.footnotes) == {"taisho": 165, "tokyo335": 16}
    heads = {s.id: s for s in zh.segments if s.kind == "head"}
    notes = {f.n: f for f in zh.footnotes}
    assert heads[notes["0594008"].segment_id].local_chapter == "pin11"
    assert "vajragarbhābhisaṁbodhi" in notes["0594008"].sa_text
    assert heads[notes["0595012"].segment_id].local_chapter == "pin12"
    assert "dvitīya kalpasya prathamaḥ" in notes["0595012"].sa_text
    assert notes["0587005"].segment_id == "T0892:0587c05.1"      # base-text note on the first fascicle heading


def test_twenty_chapters_apparatus_and_glosses(zh):
    assert [c["local"] for c in zh.metadata["chapters"]] == [f"pin{n}" for n in range(1, 21)]
    assert {s.local_chapter for s in zh.segments if s.kind in CONTENT_KINDS} == {f"pin{n}" for n in range(1, 21)}
    assert len(zh.variants) == 6 and len(zh.metadata["glosses"]) == 69


# --------------------------------------------------------------------------- Derge
def test_three_paratext_segments_after_the_final_colophon(bo):
    d418 = [s for s in bo.segments if s.id.startswith("D418:")]
    *_, final, p1, p2, p3 = d418
    assert final.kind == "colophon" and final.start == "30a.2"
    assert bo.metadata["chapters"][-1] == {"local": "D418:12", "toh": "D418", "ordinal": None, "end": "30a.2",
                                           "segment_id": final.id, "text_end": True}
    assert [(s.kind, s.start) for s in (p1, p2, p3)] == [("paratext", "30a.2"), ("paratext", "30a.3"),
                                                         ("paratext", "30a.3")]
    assert bo.report["paratext"] == 3


def test_translators_colophon(bo):
    meta = bo.metadata
    assert [t["role"] for t in meta["translators"]] == ["indian_preceptor", "translator"]
    assert meta["reviser"] and meta["revision_statement"] == bo.segments[-1].text
    assert meta["reviser"] in meta["revision_statement"]


def test_26_edit_marks_in_6_pairs_and_no_brace_left(bo, zh):
    assert bo.report["variants"] == 26 and bo.report["variant_pairs"] == 6
    assert Counter(v.kind for v in bo.variants) == {"orthographic_variant": 26}
    assert not any(ch in s.text for s in (*bo.segments, *zh.segments) for ch in "{}")


def test_23_derge_chapters_with_matching_ordinals(bo):
    assert bo.report["chapters"] == {"D417": 11, "D418": 12}
    assert bo.report["chapter_ordinal_mismatches"] == []
    assert [c["text_end"] for c in bo.metadata["chapters"]].count(True) == 2


def test_alignable_reference_units(bo):
    units = [s for s in bo.segments if s.kind in CONTENT_KINDS]
    print(f"\nD417-418: {len(units)} alignable reference units; kinds {bo.report['kinds']}")
    assert bo.report["content_segments"] == len(units) == 3042     # v0.2: 3,045 minus the 3 paratext units


def test_no_duplicate_ids_over_both_witnesses(zh, bo):
    ids = [s.id for s in (*zh.segments, *bo.segments)]
    assert len(ids) == len(set(ids))
    assert zh.report["duplicate_ids"] == bo.report["duplicate_ids"] == []


# --------------------------------------------------------------------------- ingest-stage sentinels
def test_s5_translator_notes(zh):
    for key in ("S5a", "S5b", "S5c", "S5d"):
        s = SENTINELS[key]
        (note,) = on_line(zh, s["loci"]["note"], s["quotes"]["note"], kinds={"note"})
        assert note.extra["note_class"] == s["expect"]["note_class"], key


def test_s6_colophon_is_paratext(bo):
    s = SENTINELS["S6"]
    (first,) = on_line(bo, s["loci"]["from"], s["quotes"]["from"])
    (last,) = on_line(bo, s["loci"]["to"], s["quotes"]["to"])
    ids = [x.id for x in bo.segments]
    span = bo.segments[ids.index(first.id):ids.index(last.id) + 1]
    assert [x.kind for x in span] == ["paratext"] * s["expect"]["count"]
    assert span[-1] is bo.segments[-1]


def test_s7_no_melapaka_list_between_8a6_and_8a7(bo):
    s = SENTINELS["S7"]
    (first,) = on_line(bo, s["loci"]["first"], s["quotes"]["first"])
    (second,) = on_line(bo, s["loci"]["second"], s["quotes"]["second"])
    content = [x.id for x in bo.segments if x.kind in CONTENT_KINDS]
    between = content.index(second.id) - content.index(first.id) - 1
    print(f"\nS7: {between} content segment(s) between {first.id} and {second.id}")
    assert 0 <= between <= s["expect"]["max_between"]


def test_every_sentinel_locus_and_quote_resolves(zh, bo):
    texts = {"T0892": zh, "D417": bo, "D418": bo}
    for sid, s in SENTINELS.items():
        for key, locus in s["loci"].items():
            quote = s["quotes"].get(key)
            assert on_line(texts[locus.split(":")[0]], locus, quote), (sid, key, locus)


# --------------------------------------------------------------------------- concordance
def test_concordance_declares_exactly_the_ingested_chapters(zh, bo):
    conc = registry.load_concordance(DATA / "registry" / "concordance.yaml")
    for witness, result in ((ZH, zh), (BO, bo)):
        ingested = {s.local_chapter for s in result.segments if s.local_chapter}
        assert ingested == set(conc.local_order[witness]), witness
    assigned = conc.assign_reference_chapters([s for s in bo.segments if s.kind in CONTENT_KINDS], BO)
    by_chapter = Counter(s.chapter for s in assigned)
    assert set(by_chapter) == set(conc.spans) and None not in by_chapter
    zh_assigned = conc.assign_reference_chapters([s for s in zh.segments if s.kind in CONTENT_KINDS], ZH)
    unassigned = {s.local_chapter for s in zh_assigned if s.chapter is None}
    assert unassigned == {"pin11", "pin18", "pin20"}             # the merged Chinese chapters
