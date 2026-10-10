"""Sentinels under collate.v2 (evaluation.sentinels): the optional ``conventions`` of
``relation_in``, conventions read from links and cells, and expectations written before v2
that keep holding whatever conventions a v2 consensus adds."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hevajra_matrix.core.types import Alignment, Cell, Grade, Link, Relation, Status
from hevajra_matrix.evaluation import sentinels as S

from test_evaluation_sentinels import DATA, run, sentinel

TN, SR, UNSUPPORTED = "transcribed_name", "standing_rendering", "convention_unsupported"
LOCI = {"ref_from": "D417:1a.1", "ref_to": "D417:1a.2"}         # units 1a.1.1, 1a.1.2, 1a.2.1

V2 = Alignment("claude:consensus", "ref", "wit", (
    Link("D417:1a.1.1", ("T0892:0001a01.1",), Relation.EQUIVALENT, flags=frozenset({TN})),
    Link("D417:1a.1.2", ("T0892:0001a02.1",), Relation.EQUIVALENT, flags=frozenset({SR, UNSUPPORTED})),
    Link("D417:1a.2.1", ("T0892:0001a03.1",), Relation.EQUIVALENT),
))


def relation_in(expect: dict, stage: str = "proposal") -> S.Sentinel:
    return sentinel("relation_in", LOCI, {"relations": ["equivalent"], **expect}, stage=stage)


def test_without_conventions_a_unit_convention_is_never_checked():
    """A pre-v2 expectation (relations and flags only) holds on v2 output: extra convention
    flags never fail ``flags``."""
    assert run(relation_in({}), alignment=V2).passed
    flagged = Alignment("x", "ref", "wit", tuple(
        Link(link.ref_id, link.wit_ids, link.relation, flags=link.flags | {"scope_list"}) for link in V2.links))
    assert run(relation_in({"flags": ["scope_list"]}), alignment=flagged).passed


def test_conventions_restrict_every_unit_and_name_the_failing_convention():
    assert run(relation_in({"conventions": ["none", TN, SR]}), alignment=V2).passed
    result = run(relation_in({"conventions": ["none"]}), alignment=V2)
    assert not result.passed
    assert "D417:1a.1.1=equivalent+transcribed_name" in result.detail
    assert "D417:1a.1.2=equivalent+standing_rendering" in result.detail
    assert "D417:1a.2.1" not in result.detail, "a unit without a convention is 'none'"
    v1 = Alignment("claude:consensus", "ref", "wit", tuple(
        Link(link.ref_id, link.wit_ids, link.relation) for link in V2.links))
    assert run(relation_in({"conventions": ["none"]}), alignment=v1).passed, "v1 output carries none"


def test_an_unresolved_unit_has_no_convention():
    partial = Alignment("claude:consensus", "ref", "wit", V2.links[:2])
    result = run(relation_in({"conventions": ["none", TN, SR]}), alignment=partial)
    assert not result.passed and "D417:1a.2.1=UNALIGNED" in result.detail


def test_final_stage_reads_the_convention_from_cells():
    cells = [
        Cell("D417:1a.1.1", "wit", Status.PRESENT, Grade.A, relation="equivalent", wit_ids=("T0892:0001a01.1",),
             flags=frozenset({TN})),
        Cell("D417:1a.1.2", "wit", Status.PRESENT, Grade.B, relation="equivalent", wit_ids=("T0892:0001a02.1",)),
        Cell("D417:1a.2.1", "wit", Status.PRESENT, Grade.B, relation="equivalent", wit_ids=("T0892:0001a03.1",)),
    ]
    assert run(relation_in({"conventions": ["none", TN]}, stage="final"), cells=cells).passed
    result = run(relation_in({"conventions": ["none"]}, stage="final"), cells=cells)
    assert not result.passed and "D417:1a.1.1=equivalent+transcribed_name" in result.detail


@pytest.mark.parametrize("expect", [
    {"relations": ["equivalent"], "conventions": ["name"]},
    {"relations": ["equivalent"], "conventions": []},
])
def test_load_refuses_unknown_or_empty_conventions(tmp_path: Path, expect: dict):
    entry = {"id": "a", "status": "proposed", "verified_by": None, "source": "s", "stage": "final",
             "check": "relation_in", "loci": LOCI, "quotes": {}, "expect": expect}
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 2, "sentinels": [entry]}), encoding="utf-8")
    with pytest.raises(S.SentinelError, match="expect.conventions"):
        S.load(path)
    entry.update(check="status_in", expect={"statuses": ["PRESENT"], "conventions": ["none"]})
    path.write_text(yaml.safe_dump({"schema_version": 2, "sentinels": [entry]}), encoding="utf-8")
    with pytest.raises(S.SentinelError, match="relation_in only"):
        S.load(path)


def test_committed_sentinels_read_under_both_codebook_versions():
    """S1b (proposed): the twelve class names linked to the collective count are
    category_name_omitted under v1 and generalised under v2; both keep D_any and status.
    The verified sentinels' expectations are untouched by v2."""
    loaded = {s.id.split("_")[0]: s for s in S.load(DATA / "sentinels" / "sentinels.yaml")}
    assert loaded["S1b"].status == "proposed"
    assert set(loaded["S1b"].expect["relations"]) == {"category_name_omitted", "generalised"}
    verified = {k: s.expect for k, s in loaded.items() if s.blocking}
    assert verified["S1a"] == {"status": "ABSENT"}
    assert verified["S2a"] == {"relations": ["reversal"], "polarity_flip": True}
    assert not any("conventions" in e for e in verified.values())
