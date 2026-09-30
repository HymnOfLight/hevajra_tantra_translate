"""Unit tests of hevajra_matrix.registry and of the committed registry files.

The committed files (data/registry/*.yaml) are small research data whose facts other
stages rely on, so they are checked here as well; rule tests use synthetic YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hevajra_matrix import registry
from hevajra_matrix.core.ids import REF_CHAPTERS
from hevajra_matrix.core.types import Segment
from hevajra_matrix.registry import RegistryError

REGISTRY = Path(__file__).resolve().parents[2] / "data" / "registry"
ZH, BO = "zh_T0892_song", "bo_derge_D417_418"


# --------------------------------------------------------------------------- witnesses
@pytest.fixture(scope="module")
def witnesses() -> dict[str, registry.Witness]:
    return registry.load_witnesses(REGISTRY / "witnesses.yaml")


def test_committed_witnesses_load(witnesses):
    assert {ZH, BO, "sa_snellgrove1959", "sa_conlon2022", "en_84000_2025"} <= set(witnesses)
    assert witnesses[ZH].layer == "primary" and witnesses[ZH].counts_in_statistics
    assert not witnesses["en_84000_2025"].counts_in_statistics


def test_b9_corrections_are_in_the_registry(witnesses):
    assert witnesses["sa_farrow_menon1992"].independence == "derived"
    conlon = witnesses["sa_conlon2022"].notes
    for fact in ("heta_1.4", "C, K, Na, Nb, P", "two printed editions", "II.12", "unverified", "161-186"):
        assert fact in conlon, fact
    assert witnesses[ZH].date.startswith("1054-1055") and "Szántó 2015: 1055" in witnesses[ZH].date
    derge = witnesses[BO]
    assert "1392-1481" in derge.info["reviser"] and "84000 authority file" in derge.notes
    assert "1040s" not in derge.date and "removed" in derge.notes
    ming = witnesses["zh_ming_ms"]
    assert (ming.status, ming.layer) == ("to_verify", "indirect") and "commentary" in ming.notes
    for wid in ("zh_xixia_period", "txg_tangut", "zh_ming_ms"):
        assert witnesses[wid].status == "to_verify" and "Shen 2013" in witnesses[wid].notes, wid
    assert "publisher" in witnesses["en_willemen1983"].notes


def write_yaml(path: Path, doc) -> Path:
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


WITNESS = {"id": "w1", "lang": "zh", "layer": "primary", "role": "target", "status": "available", "title": "t"}


@pytest.mark.parametrize("change, message", [
    ({"layer": "secondary"}, "layer"),
    ({"role": "hero"}, "role"),
    ({"independence": "maybe"}, "independence"),
    ({"colour": "red"}, "unknown keys"),
])
def test_witness_records_are_validated(tmp_path, change, message):
    path = write_yaml(tmp_path / "w.yaml", {"witnesses": [{**WITNESS, **change}]})
    with pytest.raises(RegistryError, match=message):
        registry.load_witnesses(path)


def test_witness_records_need_the_required_keys(tmp_path):
    rec = {k: v for k, v in WITNESS.items() if k != "title"}
    with pytest.raises(RegistryError, match="missing keys"):
        registry.load_witnesses(write_yaml(tmp_path / "w.yaml", {"witnesses": [rec]}))


def test_duplicate_witness_ids_are_rejected(tmp_path):
    path = write_yaml(tmp_path / "w.yaml", {"witnesses": [WITNESS, WITNESS]})
    with pytest.raises(RegistryError, match="duplicate"):
        registry.load_witnesses(path)


# --------------------------------------------------------------------------- manuscripts
def test_committed_manuscripts_never_invent_shelfmarks():
    mss = registry.load_manuscripts(REGISTRY / "sa_manuscripts.yaml")
    assert list(mss) == ["C", "K", "Na", "Nb", "P", "tokyo335"]
    for ms in mss.values():
        assert (ms.shelfmark, ms.group, ms.independence) == ("to_verify", "to_verify", "to_verify"), ms.id
        assert ms.source and ms.access and ms.description
    assert "0587005" in mss["tokyo335"].description


MANUSCRIPT = {"id": "X", "description": "d", "group": "g", "independence": "to_verify", "shelfmark": "to_verify",
              "source": "s", "access": "a"}


def test_manuscript_records_are_validated(tmp_path):
    bad = write_yaml(tmp_path / "m.yaml", {"schema_version": 1, "manuscripts": [{**MANUSCRIPT, "independence": "yes"}]})
    with pytest.raises(RegistryError, match="independence"):
        registry.load_manuscripts(bad)
    rec = {k: v for k, v in MANUSCRIPT.items() if k != "shelfmark"}
    short = write_yaml(tmp_path / "m2.yaml", {"schema_version": 1, "manuscripts": [rec]})
    with pytest.raises(RegistryError, match="missing keys"):
        registry.load_manuscripts(short)
    old = write_yaml(tmp_path / "m3.yaml", {"schema_version": 2, "manuscripts": [MANUSCRIPT]})
    with pytest.raises(RegistryError, match="schema_version"):
        registry.load_manuscripts(old)


# --------------------------------------------------------------------------- committed concordance
@pytest.fixture(scope="module")
def conc() -> registry.Concordance:
    return registry.load_concordance(REGISTRY / "concordance.yaml")


def test_merged_chinese_chapters_map_to_several_reference_chapters(conc):
    assert conc.refs_for_local(ZH, "pin11") == ["I.11", "II.1"]
    assert conc.refs_for_local(ZH, "pin20") == ["II.11", "II.12"]
    assert conc.refs_for_local(ZH, "pin18") == ["II.8", "II.9"]
    assert conc.refs_for_local(ZH, "pin12") == ["II.2"]


def test_derge_chapters_map_one_to_one(conc):
    for ref in REF_CHAPTERS:
        part, n = ref.split(".")
        local = f"{'D417' if part == 'I' else 'D418'}:{n}"
        assert [s.local for s in conc.locals_for(ref, BO)] == [local]
        assert conc.refs_for_local(BO, local) == [ref]


def test_ii9_is_split_with_evidence(conc):
    spans = conc.locals_for("II.9", ZH)
    assert [(s.local, s.status, s.ref_from, s.ref_to) for s in spans] == [
        ("pin18", "verified", "D418:27b.1", "D418:28a.1"),
        (None, "absent_candidate", "D418:28a.1", "D418:29a.2"),
        ("pin18", "proposed", "D418:29a.2", "D418:29a.2"),
    ]
    assert spans[0].source.startswith("code review 2026-09-30") and spans[1].is_absent
    assert "pin18" in conc.core_window("II.9", ZH, 1)


def test_pin11_evidence_cites_the_taisho_notes_and_the_parallel(conc):
    i11 = " ".join(conc.locals_for("I.11", ZH)[0].evidence)
    ii1 = " ".join(conc.locals_for("II.1", ZH)[0].evidence)
    assert "0594008" in i11
    assert "0594008" in ii1 and "0595012" in ii1 and "14a.5" in ii1 and "0595a26-27" in ii1


def test_every_committed_span_has_evidence_and_a_source(conc):
    for ref in REF_CHAPTERS:
        for witness in conc.witnesses():
            for span in conc.locals_for(ref, witness):
                assert span.evidence and all(span.evidence) and span.source, (ref, witness)


def test_assign_reference_chapters(conc):
    def seg(local: str | None, witness: str = BO) -> Segment:
        return Segment(id="D418:1a.1.1", witness=witness, lang="bo", text="x", start="1a.1", end="1a.1",
                       kind="prose", local_chapter=local)

    got = conc.assign_reference_chapters([seg("D417:7"), seg("D418:9"), seg(None)], BO)
    assert [s.chapter for s in got] == ["I.7", "II.9", None]
    zh = conc.assign_reference_chapters([seg("pin7", ZH), seg("pin11", ZH), seg("pin20", ZH)], ZH)
    assert [s.chapter for s in zh] == ["I.7", None, None]          # merged chapters are left to alignment
    with pytest.raises(KeyError, match="no chapter"):
        conc.assign_reference_chapters([seg("D418:13")], BO)


# --------------------------------------------------------------------------- synthetic concordance
LOCALS = ["c1", "c2", "c3", "c4", "c5"]


def span(local, status="proposed", **extra):
    return {"local": local, "status": status, "evidence": ["e"], "source": "s", **extra}


def concordance_doc(overrides: dict | None = None) -> dict:
    """Every reference chapter maps to c3 unless overridden; witness "w" has chapters c1..c5."""
    overrides = overrides or {}
    return {
        "schema_version": 2,
        "reference_scheme": "sa_snellgrove1959",
        "witnesses": {"w": {"locals": LOCALS}},
        "chapters": [{"ref": ref, "spans": {"w": overrides.get(ref, [span("c3")])}} for ref in REF_CHAPTERS],
    }


def load_doc(tmp_path: Path, doc: dict) -> registry.Concordance:
    return registry.load_concordance(write_yaml(tmp_path / "c.yaml", doc))


def test_many_to_many_mapping(tmp_path):
    conc = load_doc(tmp_path, concordance_doc({
        "I.1": [span("c1"), span("c2", ref_from="X:1a.1", ref_to="X:2a.1")],
        "I.2": [span("c2")],
    }))
    assert [s.local for s in conc.locals_for("I.1", "w")] == ["c1", "c2"]
    assert conc.refs_for_local("w", "c2") == ["I.1", "I.2"]
    assert conc.refs_for_local("w", "c5") == []
    assert conc.locals_for("I.1", "w")[1].ref_from == "X:1a.1"


def test_core_window_adds_neighbours_and_clips_at_the_ends(tmp_path):
    conc = load_doc(tmp_path, concordance_doc({"I.1": [span("c1")], "I.2": [span("c2"), span("c5")]}))
    assert conc.core_window("I.1", "w", 0) == {"c1"}
    assert conc.core_window("I.1", "w", 1) == {"c1", "c2"}
    assert conc.core_window("I.2", "w", 1) == {"c1", "c2", "c3", "c4", "c5"}
    assert conc.core_window("I.3", "w", 2) == {"c1", "c2", "c3", "c4", "c5"}
    with pytest.raises(ValueError):
        conc.core_window("I.1", "w", -1)


def test_core_window_of_an_absent_chapter_uses_the_neighbouring_chapters(tmp_path):
    conc = load_doc(tmp_path, concordance_doc({
        "I.1": [span("c1")], "I.2": [span(None, "absent_candidate")], "I.3": [span("c4")]}))
    assert conc.core_window("I.2", "w", 0) == {"c1", "c4"}
    assert conc.core_window("I.2", "w", 1) == {"c1", "c2", "c3", "c4", "c5"}


def test_unknown_names_raise_key_errors(tmp_path):
    conc = load_doc(tmp_path, concordance_doc())
    with pytest.raises(KeyError, match="witness"):
        conc.locals_for("I.1", "nobody")
    with pytest.raises(KeyError, match="reference chapter"):
        conc.locals_for("III.1", "w")
    with pytest.raises(KeyError, match="no chapter"):
        conc.refs_for_local("w", "c9")


def test_missing_chapters_are_rejected(tmp_path):
    doc = concordance_doc()
    doc["chapters"] = doc["chapters"][:-1]
    with pytest.raises(RegistryError, match="II.12"):
        load_doc(tmp_path, doc)


@pytest.mark.parametrize("spans, message", [
    ([span(None)], "null exactly when"),
    ([span("c1", "absent")], "null exactly when"),
    ([span("c9")], "not declared"),
    ([span("c1", "guessed")], "status"),
    ([span("c1", ref_from="X:1a.1")], "both ref_from and ref_to"),
    ([{**span("c1"), "evidence": []}], "evidence"),
    ([{**span("c1"), "source": ""}], "source"),
    ([{**span("c1"), "note": "x"}], "unknown keys"),
])
def test_malformed_spans_are_rejected(tmp_path, spans, message):
    with pytest.raises(RegistryError, match=message):
        load_doc(tmp_path, concordance_doc({"I.1": spans}))


def test_schema_version_and_witness_coverage_are_checked(tmp_path):
    doc = concordance_doc()
    doc["schema_version"] = 1
    with pytest.raises(RegistryError, match="schema_version"):
        load_doc(tmp_path, doc)
    doc = concordance_doc()
    doc["witnesses"]["v"] = {"locals": ["x1"]}
    with pytest.raises(RegistryError, match="exactly the witnesses"):
        load_doc(tmp_path, doc)
    doc = concordance_doc()
    doc["chapters"].append(doc["chapters"][0])
    with pytest.raises(RegistryError, match="repeated"):
        load_doc(tmp_path, doc)
