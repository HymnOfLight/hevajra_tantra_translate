"""Gates G0-G4, report levels, the confirmatory flag and the ledger (evaluation.gate)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate.perturb import Rate
from hevajra_matrix.config import ConfigError
from hevajra_matrix.core.types import Alignment, Link, Relation
from hevajra_matrix.evaluation import gate as GT
from hevajra_matrix.evaluation import gold as G
from hevajra_matrix.evaluation.sentinels import SentinelResult

ROOT = Path(__file__).resolve().parents[2]
CLAUDE = "claude:consensus"
DIGEST = "d" * 64
PREREG_SHA = "p" * 64


def prereg(**changes) -> dict:
    doc = yaml.safe_load((ROOT / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    doc = copy.deepcopy(doc)
    doc["frozen"] = True
    doc["instrument_digests"] = {"collate": DIGEST}
    doc["gates"]["g1"]["n_boot"] = 300
    for key, value in changes.items():
        doc[key] = value
    return doc


def make_gold(windows: int = 8, per_window: int = 10, null_every: int = 5, witness_only: int = 0) -> G.GoldSet:
    rows = []
    for w in range(windows):
        for i in range(per_window):
            uid = f"D:{w}.{i}"
            null = null_every and i % null_every == 0
            rows.append(G.GoldRow("test", f"w{w}", "ref", uid, "f", () if null else (f"T:{w}.{i}",),
                                  "no_counterpart" if null else "equivalent"))
    for k in range(witness_only):
        rows.append(G.GoldRow("test", f"w{k % windows}", "witness_only", f"+T:x.{k}", "f", (f"T:x.{k}",),
                              witness_only_kind="addition"))
    return G.GoldSet("test", "zh", tuple(rows))


def kinds(gold: G.GoldSet) -> dict[str, str]:
    return {u: "verse_line" for u in gold.units()}


def claude_alignment(gold: G.GoldSet, errors: int = 2, miss_nulls: int = 0, keep_witness_only: bool = True):
    """Gold with ``errors`` wrong units and ``miss_nulls`` NULL units linked to a guess."""
    links = []
    missed = 0
    for n, link in enumerate(x for x in gold.alignment.links if x.ref_id):
        if n < errors:
            links.append(Link(link.ref_id, ("T:wrong",), Relation.ABRIDGED, source=CLAUDE))
        elif missed < miss_nulls and link.relation is Relation.NO_COUNTERPART:
            missed += 1
            links.append(Link(link.ref_id, ("T:guess",), Relation.EQUIVALENT, source=CLAUDE))
        else:
            links.append(dataclasses.replace(link, source=CLAUDE))
    if keep_witness_only:
        links += [dataclasses.replace(x, source=CLAUDE) for x in gold.alignment.witness_only()]
    return Alignment(CLAUDE, "ref", "zh", tuple(links))


def control_alignment(gold: G.GoldSet) -> Alignment:
    """A length-only-like control: every unit linked to its neighbour's segment, never NULL."""
    refs = [x for x in gold.alignment.links if x.ref_id]
    return Alignment("dp:zero", "ref", "zh", tuple(
        Link(x.ref_id, (f"T:{x.ref_id[2:]}x",), Relation.EQUIVALENT, source="dp:zero") for x in refs))


def scores(gold: G.GoldSet, claude: Alignment | None = None) -> dict[str, G.AlignmentScores]:
    k = kinds(gold)
    return {CLAUDE: G.score(claude or claude_alignment(gold), gold, ref_kinds=k),
            "dp:zero": G.score(control_alignment(gold), gold, ref_kinds=k)}


def ok_sentinel(stage: str) -> SentinelResult:
    return SentinelResult("S", "note_kind", stage, stage, "verified", True, True)


INTEGRITY = GT.Integrity(
    ingest_reports={"zh_T0892_song": {"notes": 521, "footnotes": 181, "duplicate_ids": [], "unclassified_notes": []},
                    "bo_derge_D417_418": {"variants": 26, "paratext": 3, "duplicate_ids": []}},
    sentinels={stage: (ok_sentinel(stage),) for stage in ("ingest", "proposal", "final")},
    replicate_kappa=0.9, quote_failure_rate=0.01, missing_rate=0.0, substituted_calls=0)
PERTURB = GT.Perturbations(wrong_window=Rate(2, 50), deletion=Rate(40, 50))
CALIBRATION = GT.Calibration(planned={"pos:absent": 12}, achieved={"pos:absent": 12},
                             audit={"neg:B:sensitive": 40, "neg:B:other": 41}, null_precision=0.8, null_recall=0.75,
                             unresolved_units=0, e3_not_estimable=None, topic_labels_complete=True, topic_kappa=0.8,
                             mde=0.08, scorer_kappa=0.9)
ONCE = GT.LedgerSummary(count=1, total=1, versions=1, prereg_matches=True)


def evaluate(s=None, human=0.9, integrity=INTEGRITY, perturb=PERTURB, calibration=CALIBRATION, pr=None,
             digest=DIGEST, ledger=ONCE) -> GT.GateReport:
    return GT.evaluate(s or scores(make_gold()), human, integrity, perturb, calibration, pr or prereg(), digest,
                       ledger)


# --------------------------------------------------------------------------- levels
def test_everything_passes_level_2_confirmatory():
    report = evaluate()
    assert report.passed == {"G0": True, "G1": True, "G2": True, "G3": True}, report.reasons
    assert report.level == 2 and report.confirmatory and report.scope_note == ""
    assert report.not_estimable == {}
    assert report.deferred_null                        # 16 gold NULLs < min_null_gold 20


def test_g1_failure_gives_level_0():
    gold = make_gold()
    same = {CLAUDE: G.score(control_alignment(gold), gold, ref_kinds=kinds(gold)),
            "dp:zero": G.score(control_alignment(gold), gold, ref_kinds=kinds(gold))}
    report = evaluate(same)
    assert not report.passed["G1"] and report.level == 0
    assert any("does not beat dp:zero" in r for r in report.reasons)


def test_g0_or_g2_failure_gives_level_0_and_g3_failure_level_1():
    bad_ingest = dataclasses.replace(INTEGRITY, ingest_reports={
        **INTEGRITY.ingest_reports, "zh_T0892_song": {"notes": 520, "footnotes": 181, "duplicate_ids": ["x"]}})
    report = evaluate(integrity=bad_ingest)
    assert report.level == 0 and not report.passed["G0"]
    assert any("notes = 520" in r for r in report.reasons) and any("duplicate ids" in r for r in report.reasons)
    report = evaluate(perturb=GT.Perturbations(wrong_window=Rate(10, 50), deletion=None))
    assert report.level == 0 and not report.passed["G2"]
    report = evaluate(calibration=dataclasses.replace(CALIBRATION, audit={"neg:B:sensitive": 39}))
    assert report.level == 1 and not report.passed["G3"]


# --------------------------------------------------------------------------- G1 rules
def lenient_kappa() -> dict:
    """Kappa floors at 0, so that a test isolates the NULL rule (missing NULLs also lowers kappa)."""
    pr = prereg()
    pr["gates"]["g1"].update(min_status_kappa=0.0, min_kappa_ratio_to_human=0.0)
    return pr


def test_null_metrics_deferred_to_g3_when_gold_has_few_nulls():
    gold = make_gold(null_every=5)                                 # 16 NULLs < min_null_gold 20
    poor_null = scores(gold, claude_alignment(gold, miss_nulls=8))
    assert G.null_recall(poor_null[CLAUDE]) == 7 / 16         # unit D:0.0 is also one of the 2 errors
    report = evaluate(poor_null, pr=lenient_kappa())
    assert report.deferred_null and report.passed["G1"] and report.level == 2
    report = evaluate(poor_null, pr=lenient_kappa(), calibration=dataclasses.replace(CALIBRATION, null_recall=None))
    assert report.passed["G1"] and not report.passed["G3"] and report.level == 1
    assert any("deferred NULL recall" in r for r in report.reasons)


def test_null_metrics_gate_g1_when_gold_has_enough_nulls():
    gold = make_gold(null_every=3)                                 # 32 NULLs
    report = evaluate(scores(gold, claude_alignment(gold, miss_nulls=16)), pr=lenient_kappa())
    assert not report.deferred_null and not report.passed["G1"]
    assert any("NULL recall" in r for r in report.reasons)
    assert evaluate(scores(gold)).passed["G1"]


def test_witness_only_recall_is_gated_only_with_enough_gold():
    few = make_gold(witness_only=9)
    assert evaluate(scores(few, claude_alignment(few, keep_witness_only=False))).passed["G1"]
    many = make_gold(witness_only=10)
    report = evaluate(scores(many, claude_alignment(many, keep_witness_only=False)))
    assert not report.passed["G1"] and any("witness-only recall" in r for r in report.reasons)
    assert evaluate(scores(many)).passed["G1"]


def test_human_kappa_ceiling():
    assert not evaluate(human=None).passed["G1"]
    s = scores(make_gold())
    kappa = G.status_kappa(s[CLAUDE])
    for ratio, ok in ((kappa - 0.01, True), (kappa + 0.01, False)):
        pr = prereg()
        pr["gates"]["g1"]["min_kappa_ratio_to_human"] = ratio
        report = evaluate(s, human=1.0, pr=pr)
        assert report.passed["G1"] is ok, report.reasons


def test_absolute_floor_applies_even_when_claude_beats_controls():
    gold = make_gold()
    report = evaluate(scores(gold, claude_alignment(gold, errors=25)))
    assert not report.passed["G1"] and any("link F1" in r for r in report.reasons)


def test_sources_must_share_units():
    gold = make_gold()
    s = scores(gold)
    s["dp:zero"] = G.score(control_alignment(gold), gold, list(gold.units())[:10], ref_kinds=kinds(gold))
    report = evaluate(s)
    assert not report.passed["G1"] and any("same units" in r for r in report.reasons)


def test_check_g1_reports_the_paired_comparison():
    gold = make_gold()
    specs = GT.GateSpecs.from_prereg(prereg())
    result = GT.check_g1(scores(gold), 0.9, specs.g1, specs.seed)
    control, point, lo, hi = result.comparisons["link_f1"]
    assert control == "dp:zero" and point > 0 and lo > 0 and lo <= point <= hi
    assert GT.check_g1({CLAUDE: scores(gold)[CLAUDE]}, 0.9, specs.g1, specs.seed).passed is False


# --------------------------------------------------------------------------- G2 / G3 / G4 details
def test_g2_integrity_rules():
    assert not evaluate(integrity=dataclasses.replace(INTEGRITY, substituted_calls=1)).passed["G2"]
    assert not evaluate(integrity=dataclasses.replace(INTEGRITY, substituted_calls=None)).passed["G2"]
    assert not evaluate(integrity=dataclasses.replace(INTEGRITY, replicate_kappa=0.79)).passed["G2"]
    assert not evaluate(digest="e" * 64, ledger=GT.LedgerSummary(1, 1, 1, True)).passed["G2"]
    failed = SentinelResult("S2a", "relation_in", "proposal", "proposal", "verified", True, False, "reversal missing")
    proposed = SentinelResult("S9", "linked_local", "proposal", "proposal", "proposed", False, False, "x")
    sentinels = {**INTEGRITY.sentinels, "proposal": (proposed,)}
    assert evaluate(integrity=dataclasses.replace(INTEGRITY, sentinels=sentinels)).passed["G2"]
    sentinels = {**INTEGRITY.sentinels, "proposal": (failed, proposed)}
    report = evaluate(integrity=dataclasses.replace(INTEGRITY, sentinels=sentinels))
    assert not report.passed["G2"] and any("S2a" in r for r in report.reasons)
    sentinels = {k: v for k, v in INTEGRITY.sentinels.items() if k != "proposal"}
    assert not evaluate(integrity=dataclasses.replace(INTEGRITY, sentinels=sentinels)).passed["G2"]


def test_g3_unresolved_units_need_a_narrow_manski_interval():
    c = dataclasses.replace(CALIBRATION, unresolved_units=3, manski_width=0.04)
    assert evaluate(calibration=c).passed["G3"]
    assert not evaluate(calibration=dataclasses.replace(c, manski_width=0.06)).passed["G3"]
    assert not evaluate(calibration=dataclasses.replace(CALIBRATION, achieved={"pos:absent": 11})).passed["G3"]


def test_g4_not_estimable():
    report = evaluate(calibration=GT.Calibration())
    assert set(report.not_estimable) == {"E3", "E4"}
    assert "topic labels incomplete" in report.not_estimable["E4"]
    assert any("E6" in r for r in report.reasons)


# --------------------------------------------------------------------------- confirmatory flag
def test_confirmatory_flag():
    report = evaluate(ledger=GT.LedgerSummary(count=2, total=3, versions=2, prereg_matches=True))
    assert not report.confirmatory
    assert report.scope_note == "exploratory; test set scored 3 times by 2 instrument versions"
    assert not evaluate(ledger=GT.LedgerSummary(1, 1, 1, prereg_matches=False)).confirmatory
    assert not evaluate(pr=prereg(frozen=False)).confirmatory


def test_unknown_gate_keys_are_refused():
    bad = prereg()
    bad["gates"]["g1"]["min_link_f2"] = 0.8
    with pytest.raises(ConfigError, match="min_link_f2"):
        GT.GateSpecs.from_prereg(bad)
    bad = prereg()
    bad["gates"]["g9"] = {}
    with pytest.raises(ConfigError, match="g9"):
        GT.GateSpecs.from_prereg(bad)


# --------------------------------------------------------------------------- ledger
def test_ledger_is_append_only_and_counted(tmp_path: Path):
    path = tmp_path / "ledger" / "test_evaluations.jsonl"
    assert GT.ledger_count(path, DIGEST) == 0
    assert GT.ledger_summary(path, DIGEST, PREREG_SHA) == GT.LedgerSummary(0, 0, 0, False)
    first = GT.ledger_record(DIGEST, PREREG_SHA, "abc123", {CLAUDE: {"link_f1": 0.9}}, True, ts="2026-10-01T00:00:00+00:00")
    GT.ledger_append(path, first)
    before = path.read_bytes()
    GT.ledger_append(path, GT.ledger_record("e" * 64, PREREG_SHA, None, {}, False))
    GT.ledger_append(path, GT.ledger_record(DIGEST, "q" * 64, "abc124", {}, False))
    assert path.read_bytes().startswith(before)
    assert json.loads(before) == first and first["gate"] == "pass"
    assert GT.ledger_count(path, DIGEST) == 2 and GT.ledger_count(path, "e" * 64) == 1
    assert GT.ledger_summary(path, DIGEST, PREREG_SHA) == GT.LedgerSummary(2, 3, 2, False)
    assert GT.ledger_summary(path, "e" * 64, PREREG_SHA) == GT.LedgerSummary(1, 3, 2, True)


def test_ledger_rejects_malformed_records(tmp_path: Path):
    path = tmp_path / "l.jsonl"
    record = GT.ledger_record(DIGEST, PREREG_SHA, None, {}, True)
    with pytest.raises(ValueError):
        GT.ledger_append(path, {**record, "extra": 1})
    with pytest.raises(ValueError):
        GT.ledger_append(path, {**record, "gate": "maybe"})
    with pytest.raises(ValueError):
        GT.ledger_append(path, {**record, "prereg_sha256": ""})
    assert not path.exists()
