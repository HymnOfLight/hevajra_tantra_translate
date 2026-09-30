"""Schema checks of data/sentinels/sentinels.yaml (schema version 2).

The sentinel checker itself belongs to the evaluation stage; this test keeps the data
file well-formed in the meantime: required fields, the 8 check kinds with their loci and
expect keys, valid statuses, relations, flags and note classes, and quotes keyed like loci.
Whether the loci and quotes resolve in the real texts is checked in
tests/integration/test_ingest_realdata.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.core.types import Relation, Status, WitnessOnlyKind
from hevajra_matrix.ingest.notes import NOTE_CLASSES

PATH = Path(__file__).resolve().parents[2] / "data" / "sentinels" / "sentinels.yaml"
DOC = yaml.safe_load(PATH.read_text(encoding="utf-8"))
SENTINELS = DOC["sentinels"]

FIELDS = {"id", "status", "verified_by", "source", "stage", "check", "loci", "quotes", "expect"}
UNIT_RANGE = ({"ref_from", "ref_to"}, {"wit_from", "wit_to"})
# check kind -> (required loci keys, optional loci keys, required expect keys, optional expect keys)
KINDS = {
    "note_kind": ({"note"}, set(), {"note_class"}, set()),
    "paratext": ({"from", "to"}, set(), {"count"}, set()),
    "adjacent": ({"first", "second"}, set(), {"max_between"}, set()),
    "relation_in": (*UNIT_RANGE, {"relations"}, {"polarity_flip", "flags"}),
    "status_in": (*UNIT_RANGE, {"statuses"}, set()),
    "none_status": (*UNIT_RANGE, {"status"}, set()),
    "linked_local": (*UNIT_RANGE, {"local"}, set()),
    "witness_only": ({"wit_from", "wit_to"}, set(), {"kinds"}, set()),
}
GOLD_FLAGS = {"scope_list", "uniform_across_list", "instruction_as_mantra", "reordered", "unsure"}
LOCUS_RE = re.compile(r"(D417|D418):[0-9]+[ab]\.[0-9]+|T0892:[0-9]{4}[abc][0-9]{2}")
LOCALS_RE = re.compile(r"pin([1-9]|1[0-9]|20)|D41[78]:[0-9]+")


def test_schema_version_and_unique_ids():
    assert DOC["schema_version"] == 2
    ids = [s["id"] for s in SENTINELS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("s", SENTINELS, ids=lambda s: s["id"])
def test_sentinel_fields(s):
    assert set(s) == FIELDS
    assert s["status"] in {"proposed", "verified", "retired"}
    assert s["stage"] in {"ingest", "proposal", "final"}
    assert isinstance(s["source"], str) and s["source"]
    if s["status"] == "verified":
        assert isinstance(s["verified_by"], str) and s["verified_by"]
    else:
        assert s["verified_by"] is None


@pytest.mark.parametrize("s", SENTINELS, ids=lambda s: s["id"])
def test_sentinel_check_loci_and_expectation(s):
    required, optional, expect_required, expect_optional = KINDS[s["check"]]
    loci = set(s["loci"])
    assert required <= loci <= required | optional
    assert not loci & optional or optional <= loci            # a witness range is given whole or not at all
    assert set(s["quotes"]) <= loci
    assert all(LOCUS_RE.fullmatch(v) for v in s["loci"].values()), s["loci"]
    assert all(isinstance(q, str) and 0 < len(q) <= 60 for q in s["quotes"].values())
    assert expect_required <= set(s["expect"]) <= expect_required | expect_optional
    _check_expect_values(s["expect"])


def _check_expect_values(expect: dict) -> None:
    assert set(expect.get("relations", [])) <= {r.value for r in Relation}
    assert set(expect.get("flags", [])) <= GOLD_FLAGS
    assert set(expect.get("statuses", [])) | ({expect["status"]} if "status" in expect else set()) <= {
        s.value for s in Status}
    assert set(expect.get("kinds", [])) <= {k.value for k in WitnessOnlyKind}
    if "note_class" in expect:
        assert expect["note_class"] in NOTE_CLASSES
    if "local" in expect:
        assert LOCALS_RE.fullmatch(expect["local"])
    for key in ("count", "max_between"):
        if key in expect:
            assert isinstance(expect[key], int) and expect[key] >= 0
    if "polarity_flip" in expect:
        assert isinstance(expect["polarity_flip"], bool)


def test_every_check_kind_has_an_example():
    assert {s["check"] for s in SENTINELS} == set(KINDS)


def test_verified_and_proposed_sets():
    by_status = {st: {s["id"].split("_")[0] for s in SENTINELS if s["status"] == st} for st in ("verified", "proposed")}
    assert by_status["verified"] == {"S1a", "S2a", "S3a", "S3b", "S3c", "S4", "S5a", "S5b", "S5c", "S5d", "S6"}
    assert by_status["proposed"] == {"S1b", "S2b", "S7", "S8a", "S8b", "S9", "S10"}
    assert all(s["verified_by"] == "code review 2026-09-30" for s in SENTINELS if s["status"] == "verified")


def test_ingest_stage_sentinels_are_the_ones_the_ingester_can_decide():
    ingest = {s["id"].split("_")[0]: s["check"] for s in SENTINELS if s["stage"] == "ingest"}
    assert ingest == {"S5a": "note_kind", "S5b": "note_kind", "S5c": "note_kind", "S5d": "note_kind",
                      "S6": "paratext", "S7": "adjacent"}


def test_s9_cites_both_taisho_footnotes():
    (s9,) = [s for s in SENTINELS if s["id"].startswith("S9_")]
    assert "0594008" in s9["source"] and "0595012" in s9["source"]
