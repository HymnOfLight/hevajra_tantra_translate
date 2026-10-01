"""``pipeline.store``: every record a stage hands on survives the JSON round trip."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from hevajra_matrix.collate.verify import Collation
from hevajra_matrix.core.types import (
    Alignment,
    Cell,
    Diagnostic,
    Estimate,
    Grade,
    Link,
    Quote,
    Relation,
    Segment,
    Status,
    WitnessOnlyKind,
)
from hevajra_matrix.evaluation.gate import GateReport
from hevajra_matrix.evaluation.sentinels import SentinelResult
from hevajra_matrix.pipeline import store


def through_json(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False))


def test_segment_round_trip() -> None:
    seg = Segment("T0892:0592a27.n1", "zh_T0892_song", "zh", "text", "0592a27", "0592a27", "note", "pin7", None,
                  "abc123", {"note_class": "vorlage_statement", "host": "T0892:0592a27.1"})
    assert store.segment_from_dict(through_json(store.segment_to_dict(seg))) == seg


def test_alignment_round_trip_keeps_relation_types_quotes_and_flags() -> None:
    a = Alignment("claude:consensus", "bo", "zh", (
        Link("D418:17b.6.1", ("T0892:0596b25.1",), Relation.REVERSAL, True, "high",
             (Quote("ref", "q", ("D418:17b.6.1",)),), frozenset({"corroborated", "crossing"}), "claude:consensus"),
        Link("D418:17b.6.2", (), Relation.NO_COUNTERPART),
        Link(None, ("T0892:0601c01.1",), WitnessOnlyKind.ADDITION, source="s")))
    back = store.alignment_from_records(through_json(store.alignment_records(a)))
    assert back == a
    assert isinstance(back.links[2].relation, WitnessOnlyKind) and isinstance(back.links[0].relation, Relation)


def test_alignment_file_needs_its_header() -> None:
    with pytest.raises(ValueError, match="header"):
        store.alignment_from_records([store.link_to_dict(Link("u", (), Relation.NO_COUNTERPART))])


def test_collation_round_trip() -> None:
    a = Alignment("r1", "bo", "zh", (Link("u1", ("z1",), Relation.EQUIVALENT),))
    c = Collation(a, MappingProxyType({"u2": "refused:other"}), (Diagnostic("relocation", ("u1",), ("z1",), "d"),),
                  (Link("u3", ("z2",), Relation.ABRIDGED, flags=frozenset({"substituted_model_hint"})),))
    back = store.collation_from_dict(a, through_json(store.collation_to_dict(c)))
    assert back.alignment == a and dict(back.unresolved) == {"u2": "refused:other"}
    assert back.diagnostics == c.diagnostics and back.hints == c.hints


def test_cell_estimate_gate_and_sentinel_round_trips() -> None:
    cell = Cell("u1", "zh", Status.PARTIAL, Grade.C, "I.1", "abridged", False, None, ("z1", "z2"),
                frozenset({"relocation"}), 0.1, 0.0, None, "claude:consensus")
    assert store.cell_from_dict(through_json(store.cell_to_dict(cell))) == cell
    est = Estimate("E1_any", 0.1, 0.05, 0.2, 10, "scope", ("a", "b"), 2, None)
    assert store.estimate_from_dict(through_json(store.estimate_to_dict(est))) == est
    missing = Estimate.missing("E3", "G4: no manuscript column")
    assert store.estimate_from_dict(through_json(store.estimate_to_dict(missing))) == missing
    gate = GateReport(1, False, MappingProxyType({"G0": True}), ("G3: x",), True, MappingProxyType({"E3": "G4: y"}),
                      "exploratory")
    back = store.gate_from_dict(through_json(store.gate_to_dict(gate)))
    assert (back.level, back.confirmatory, dict(back.passed), back.reasons, back.deferred_null,
            dict(back.not_estimable), back.scope_note) == (1, False, {"G0": True}, ("G3: x",), True,
                                                           {"E3": "G4: y"}, "exploratory")
    result = SentinelResult("S1", "adjacent", "ingest", "ingest", "proposed", False, True, "ok")
    assert store.sentinel_result_from_dict(through_json(store.sentinel_result_to_dict(result))) == result
