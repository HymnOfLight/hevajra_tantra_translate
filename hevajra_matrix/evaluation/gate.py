"""Gates G0-G4, report levels, the confirmatory flag and the test-evaluation ledger (synthesis 5.5).

    G0 integrity    ingest counts as preregistered; 0 duplicate ids; 0 unclassified notes;
                    verified sentinels pass at stage ingest
    G1 validity     on the test windows, Claude against every control on the same units
                    (critique A7): link F1 >= floor AND the paired window-cluster bootstrap
                    interval of F1(Claude) - F1(best control) has CI_low > 0; status kappa
                    >= floor, >= ratio * human kappa AND the same paired difference CI_low > 0;
                    witness-only recall >= floor when gold has enough witness-only segments;
                    NULL precision and recall >= floor when gold has enough NULLs, otherwise
                    DEFERRED to G3
    G2 instrument   replicate Fleiss kappa, quote failures, missing units, every measurement
                    call served by the requested model, digest = prereg, P-wrong-window and
                    P-deletion; verified sentinels pass at stage proposal. (No drift canary:
                    the ledger lets runs be compared. P-negation is reported, never gated.)
    G3 calibration  every stratum reached its planned n_h; every stratum of the current
                    machine cells with unverified units has phase-2 sample verdicts; every machine-negative stratum
                    audited >= floor; deferred NULL metrics >= floor; unresolved units all
                    resolved or Manski width <= max; verified sentinels pass at stage final
    G4 estimands    per estimand: E3 and E4 print NOT_ESTIMABLE with the reason; E6 falls
                    back to two-phase-corrected outcomes (a note, not a block)

Level 0 when any of G0-G2 fails, 1 when G0-G2 pass, 2 when G3 passes too. The report is
confirmatory only if the preregistration is frozen, the ledger's records for this digest
carry the current prereg sha256, the digest is the preregistered one, and the ledger shows
the test set scored exactly once for this digest.

"Best control" is the non-instrument source with the highest point value of the metric
(P1, B0, imported external alignments and the shuffled placebo P2 all compete; P2 is the
chance floor and never wins in practice). Thresholds come from
``config/preregistration.yaml: gates``; unknown keys raise ``ConfigError``.

Order of use for a test scoring: ``check_g1`` -> ``ledger_append`` (with ``ledger_record``)
-> ``evaluate`` with the ``ledger_summary`` read after the append.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..collate.consensus import SOURCE_CONSENSUS
from ..collate.perturb import Rate
from ..config import ConfigError, from_mapping
from ..core.io import append_jsonl, read_jsonl
from .gold import AlignmentScores, link_f1, null_precision, null_recall, paired_difference, status_kappa, \
    witness_only_recall
from .sentinels import SentinelResult, blocking_failures

LEDGER_KEYS = ("ts", "instrument_digest", "prereg_sha256", "git_commit", "metrics", "gate")
GATE_VALUES = ("pass", "fail")

# Preregistered ingest counts (synthesis 5.5 G0), used while the preregistration has no g0 section.
DEFAULT_G0_EXPECTED: Mapping[str, Mapping[str, int]] = MappingProxyType({
    "zh_T0892_song": MappingProxyType({"notes": 521, "footnotes": 181}),
    "bo_derge_D417_418": MappingProxyType({"variants": 26, "paratext": 3}),
})


# --------------------------------------------------------------------------- thresholds
@dataclass(frozen=True)
class G0Spec:
    expected: Mapping[str, Mapping[str, int]] = field(default_factory=lambda: DEFAULT_G0_EXPECTED)
    # witness -> ingest report key -> expected count


@dataclass(frozen=True)
class G1Spec:
    min_link_f1: float = 0.80
    min_status_kappa: float = 0.70
    min_kappa_ratio_to_human: float = 0.80
    min_witness_only_gold: int = 10
    min_witness_only_recall: float = 0.70
    min_null_gold: int = 20
    min_null_precision: float = 0.70
    min_null_recall: float = 0.70
    beat_controls: bool = True          # A7: paired bootstrap CI_low of (Claude - best control) > 0
    n_boot: int = 2000


@dataclass(frozen=True)
class G2Spec:
    min_replicate_kappa: float = 0.80
    max_quote_failure_rate: float = 0.02
    max_missing_rate: float = 0.01
    max_wrong_window_false_link_rate: float = 0.10
    min_deletion_recall: float = 0.70


@dataclass(frozen=True)
class G3Spec:
    min_audit_per_stratum: int = 40
    max_manski_width: float = 0.05


@dataclass(frozen=True)
class G4Spec:
    min_manuscripts_per_unit: int = 2
    min_topic_kappa: float = 0.70
    max_mde: float = 0.10
    min_scorer_kappa: float = 0.80


@dataclass(frozen=True)
class GateSpecs:
    g0: G0Spec
    g1: G1Spec
    g2: G2Spec
    g3: G3Spec
    g4: G4Spec
    seed: int

    @classmethod
    def from_prereg(cls, prereg: Mapping[str, Any]) -> "GateSpecs":
        gates = dict(prereg.get("gates") or {})
        unknown = sorted(set(gates) - {"g0", "g1", "g2", "g3", "g4"})
        if unknown:
            raise ConfigError(f"preregistration.yaml: gates: unknown key(s) {unknown}")
        seed = (prereg.get("stats") or {}).get("seed")
        if not isinstance(seed, int):
            raise ConfigError("preregistration.yaml: stats.seed (the bootstrap seed) must be an integer")
        return cls(*(from_mapping(spec, gates.get(key), f"preregistration.yaml: gates.{key}")
                     for key, spec in (("g0", G0Spec), ("g1", G1Spec), ("g2", G2Spec), ("g3", G3Spec),
                                       ("g4", G4Spec))), seed=seed)


# --------------------------------------------------------------------------- evidence
@dataclass(frozen=True)
class Integrity:
    """Run facts for G0 and G2. None means "not measured", which fails the gate that needs it.

    ``ingest_reports``    witness -> ``IngestResult.report``
    ``sentinels``         stage the check ran at -> its results (``sentinels.check``)
    ``substituted_calls`` measurement calls served by another model than requested
    """

    ingest_reports: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    sentinels: Mapping[str, Sequence[SentinelResult]] = field(default_factory=dict)
    replicate_kappa: float | None = None
    quote_failure_rate: float | None = None
    missing_rate: float | None = None
    substituted_calls: int | None = None


@dataclass(frozen=True)
class Perturbations:
    wrong_window: Rate | None = None     # false-link rate inside the wrong window (G2)
    deletion: Rate | None = None         # recall of injected omissions (G2)
    negation: Rate | None = None         # polarity recall (reported only)


@dataclass(frozen=True)
class Calibration:
    """Two-phase facts for G3 and estimand facts for G4.

    ``planned`` / ``achieved``  stratum -> n_h planned and verified
    ``audit``                   machine-negative x topic-group stratum -> audited units
    ``uncalibrated``            stratum of the current machine cells -> unverified units, for
                                strata with no phase-2 sample verdict (``twophase.uncalibrated_strata``;
                                e.g. strata no plan covers because topic labels changed after sampling)
    ``e3_not_estimable``        reason from ``stats.decompose.not_estimable_reason`` (None: ok)
    """

    planned: Mapping[str, int] = field(default_factory=dict)
    achieved: Mapping[str, int] = field(default_factory=dict)
    audit: Mapping[str, int] = field(default_factory=dict)
    uncalibrated: Mapping[str, int] = field(default_factory=dict)
    null_precision: float | None = None
    null_recall: float | None = None
    unresolved_units: int | None = None
    manski_width: float | None = None
    e3_not_estimable: str | None = "no Sanskrit manuscript column"
    topic_labels_complete: bool = False
    topic_kappa: float | None = None
    mde: float | None = None
    scorer_kappa: float | None = None


@dataclass(frozen=True)
class LedgerSummary:
    """What the ledger says about a digest: ``count`` scorings of it, ``total`` test scorings by
    ``versions`` distinct digests, and whether every record of it carries the current prereg sha."""

    count: int = 0
    total: int = 0
    versions: int = 0
    prereg_matches: bool = False


@dataclass(frozen=True)
class G1Result:
    passed: bool
    reasons: tuple[str, ...]
    deferred_null: bool
    comparisons: Mapping[str, tuple[str, float | None, float | None, float | None]]  # metric -> (control, diff)


@dataclass(frozen=True)
class GateReport:
    level: int
    confirmatory: bool
    passed: Mapping[str, bool]
    reasons: tuple[str, ...]
    deferred_null: bool = False
    not_estimable: Mapping[str, str] = field(default_factory=dict)
    scope_note: str = ""


# --------------------------------------------------------------------------- G1
def _floor(reasons: list[str], name: str, value: float | None, floor: float) -> None:
    if value is None or value < floor:
        reasons.append(f"G1: {name} {_fmt(value)} < {_fmt(floor)}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def check_g1(scores_by_source: Mapping[str, AlignmentScores], human_kappa: float | None, spec: G1Spec, seed: int,
             instrument_source: str = SOURCE_CONSENSUS) -> G1Result:
    """G1 on test gold. Every source must have been scored on the same units."""
    reasons: list[str] = []
    claude = scores_by_source.get(instrument_source)
    controls = {k: v for k, v in scores_by_source.items() if k != instrument_source}
    if claude is None or not controls:
        return G1Result(False, (f"G1: need {instrument_source} and at least one control scored",), False, {})
    if any(s.unit_ids() != claude.unit_ids() for s in controls.values()):
        return G1Result(False, ("G1: sources were not scored on the same units",), False, {})
    comparisons = {}
    for name, fn in (("link_f1", link_f1), ("status_kappa", status_kappa)):
        values = {k: fn(v) for k, v in controls.items()}
        best = max(values, key=lambda k: float("-inf") if values[k] is None else values[k])
        diff = paired_difference(lambda s, ws, fn=fn: fn(s.on_windows(ws)), claude, controls[best], claude.windows(),
                                 spec.n_boot, seed)
        comparisons[name] = (best, *diff)
        if spec.beat_controls and (diff[1] is None or diff[1] <= 0):
            reasons.append(f"G1: {name} does not beat {best}: paired difference {_fmt(diff[0])} "
                           f"[{_fmt(diff[1])}, {_fmt(diff[2])}]")
    _floor(reasons, "link F1", link_f1(claude), spec.min_link_f1)
    kappa = status_kappa(claude)
    _floor(reasons, "status kappa", kappa, spec.min_status_kappa)
    if human_kappa is None:
        reasons.append("G1: human-human kappa is not available (second annotator)")
    else:
        _floor(reasons, "status kappa vs human", kappa, spec.min_kappa_ratio_to_human * human_kappa)
    counts = claude.counts()
    if counts["gold_witness_only"] >= spec.min_witness_only_gold:
        _floor(reasons, "witness-only recall", witness_only_recall(claude), spec.min_witness_only_recall)
    deferred = counts["gold_null"] < spec.min_null_gold
    if not deferred:
        _floor(reasons, "NULL precision", null_precision(claude), spec.min_null_precision)
        _floor(reasons, "NULL recall", null_recall(claude), spec.min_null_recall)
    return G1Result(not reasons, tuple(reasons), deferred, MappingProxyType(comparisons))


# --------------------------------------------------------------------------- G0, G2, G3, G4
def _sentinel_reasons(gate: str, stage: str, integrity: Integrity) -> list[str]:
    results = integrity.sentinels.get(stage)
    if results is None:
        return [f"{gate}: sentinels were not checked at stage {stage}"]
    return [f"{gate}: verified sentinel {r.sentinel_id} failed: {r.detail}" for r in blocking_failures(results)]


def _g0(integrity: Integrity, spec: G0Spec) -> list[str]:
    reasons = []
    for witness, expected in spec.expected.items():
        report = integrity.ingest_reports.get(witness)
        if report is None:
            reasons.append(f"G0: no ingest report for {witness}")
            continue
        reasons += [f"G0: {witness} {key} = {report.get(key)} (expected {n})"
                    for key, n in expected.items() if report.get(key) != n]
    for witness, report in integrity.ingest_reports.items():
        for key in ("duplicate_ids", "unclassified_notes"):
            if report.get(key):
                reasons.append(f"G0: {witness} has {len(report[key])} {key.replace('_', ' ')}")
    return reasons + _sentinel_reasons("G0", "ingest", integrity)


def _rate(rate: Rate | None) -> float | None:
    return None if rate is None else rate.value


def _g2(integrity: Integrity, perturbations: Perturbations, spec: G2Spec, digest: str,
        prereg: Mapping[str, Any]) -> list[str]:
    reasons = []

    def need(name: str, value: float | None, ok) -> None:
        if value is None or not ok(value):
            reasons.append(f"G2: {name} {_fmt(value)}")

    need("replicate kappa", integrity.replicate_kappa, lambda v: v >= spec.min_replicate_kappa)
    need("quote failure rate", integrity.quote_failure_rate, lambda v: v <= spec.max_quote_failure_rate)
    need("missing rate", integrity.missing_rate, lambda v: v <= spec.max_missing_rate)
    if integrity.substituted_calls is None:
        reasons.append("G2: the served model of the measurement calls was not recorded")
    elif integrity.substituted_calls != 0:
        reasons.append(f"G2: {integrity.substituted_calls} measurement call(s) not served by the requested model")
    if digest not in (prereg.get("instrument_digests") or {}).values():
        reasons.append("G2: the instrument digest is not the preregistered one")
    need("P-wrong-window false-link rate", _rate(perturbations.wrong_window),
         lambda v: v <= spec.max_wrong_window_false_link_rate)
    need("P-deletion recall", _rate(perturbations.deletion), lambda v: v >= spec.min_deletion_recall)
    return reasons + _sentinel_reasons("G2", "proposal", integrity)


def _g3(calibration: Calibration, integrity: Integrity, spec: G3Spec, g1: G1Spec, deferred_null: bool) -> list[str]:
    reasons = [f"G3: stratum {s} verified {calibration.achieved.get(s, 0)} of {n}"
               for s, n in calibration.planned.items() if calibration.achieved.get(s, 0) < n]
    if not calibration.planned:
        reasons.append("G3: no verification plan")
    reasons += [f"G3: stratum {s} has {n} unverified unit(s) and no phase-2 sample verdict (not planned or not "
                "verified; its estimate would be the prior)" for s, n in sorted(calibration.uncalibrated.items())]
    if not calibration.audit:
        reasons.append("G3: no audit of machine negatives")
    reasons += [f"G3: audit stratum {s} has {n} < {spec.min_audit_per_stratum}"
                for s, n in calibration.audit.items() if n < spec.min_audit_per_stratum]
    if deferred_null:
        for name, value, floor in (("NULL precision", calibration.null_precision, g1.min_null_precision),
                                   ("NULL recall", calibration.null_recall, g1.min_null_recall)):
            if value is None or value < floor:
                reasons.append(f"G3: deferred {name} {_fmt(value)} < {floor}")
    width = calibration.manski_width
    if calibration.unresolved_units != 0 and (width is None or width > spec.max_manski_width):
        reasons.append(f"G3: {calibration.unresolved_units} unresolved unit(s) and Manski width {_fmt(width)}")
    return reasons + _sentinel_reasons("G3", "final", integrity)


def _g4(calibration: Calibration, spec: G4Spec) -> tuple[dict[str, str], list[str]]:
    out: dict[str, str] = {}
    if calibration.e3_not_estimable:
        out["E3"] = f"G4: {calibration.e3_not_estimable}"
    e4 = []
    if not calibration.topic_labels_complete:
        e4.append("topic labels incomplete")
    if calibration.topic_kappa is None or calibration.topic_kappa < spec.min_topic_kappa:
        e4.append(f"topic kappa {_fmt(calibration.topic_kappa)} < {spec.min_topic_kappa}")
    if calibration.mde is None or calibration.mde > spec.max_mde:
        e4.append(f"MDE {_fmt(calibration.mde)} > {spec.max_mde}")
    if e4:
        out["E4"] = "G4: " + "; ".join(e4)
    notes = []
    if calibration.scorer_kappa is None:
        notes.append("G4: E6 scorer kappa not available (no human codes); E6 H1/H2 rest on unvalidated "
                     "scorer labels and are not final")
    elif calibration.scorer_kappa < spec.min_scorer_kappa:
        notes.append(f"G4: E6 scorer kappa {_fmt(calibration.scorer_kappa)} < {spec.min_scorer_kappa}; "
                     "E6 uses two-phase-corrected outcomes")
    return out, notes


# --------------------------------------------------------------------------- evaluate
def evaluate(scores_by_source: Mapping[str, AlignmentScores], human_kappa: float | None, integrity: Integrity,
             perturbations: Perturbations, calibration: Calibration, prereg: Mapping[str, Any],
             instrument_digest: str, ledger: LedgerSummary,
             instrument_source: str = SOURCE_CONSENSUS) -> GateReport:
    """All gates, the report level and the confirmatory flag (see the module docstring)."""
    specs = GateSpecs.from_prereg(prereg)
    g1 = check_g1(scores_by_source, human_kappa, specs.g1, specs.seed, instrument_source)
    found = {"G0": _g0(integrity, specs.g0), "G1": list(g1.reasons),
             "G2": _g2(integrity, perturbations, specs.g2, instrument_digest, prereg),
             "G3": _g3(calibration, integrity, specs.g3, specs.g1, g1.deferred_null)}
    passed = {gate: not r for gate, r in found.items()}
    level = 0 if not (passed["G0"] and passed["G1"] and passed["G2"]) else (2 if passed["G3"] else 1)
    not_estimable, notes = _g4(calibration, specs.g4)
    confirmatory = (prereg.get("frozen") is True and ledger.prereg_matches and ledger.count == 1
                    and instrument_digest in (prereg.get("instrument_digests") or {}).values())
    note = "" if confirmatory else (f"exploratory; test set scored {ledger.total} times by {ledger.versions} "
                                    "instrument versions")
    reasons = tuple(r for gate in found.values() for r in gate) + tuple(notes)
    return GateReport(level, confirmatory, MappingProxyType(passed), reasons, g1.deferred_null,
                      MappingProxyType(not_estimable), note)


# --------------------------------------------------------------------------- ledger
def ledger_record(instrument_digest: str, prereg_sha256: str, git_commit: str | None,
                  metrics: Mapping[str, Any], passed: bool, ts: str | None = None) -> dict[str, Any]:
    """One ledger line; ``metrics`` is JSON-ready (e.g. source -> ``AlignmentScores.metrics()``)."""
    return {"ts": ts or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "instrument_digest": instrument_digest, "prereg_sha256": prereg_sha256, "git_commit": git_commit,
            "metrics": dict(metrics), "gate": "pass" if passed else "fail"}


def ledger_append(path: Path, record: Mapping[str, Any]) -> None:
    """Append one test scoring to the ledger (never rewritten; ``data/ledger/README.md``)."""
    if set(record) != set(LEDGER_KEYS):
        raise ValueError(f"a ledger record has exactly the keys {LEDGER_KEYS}, got {sorted(record)}")
    if record["gate"] not in GATE_VALUES or not record["instrument_digest"] or not record["prereg_sha256"]:
        raise ValueError("a ledger record needs gate pass|fail, an instrument digest and the prereg sha256")
    append_jsonl(Path(path), record)


def _records(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if Path(path).is_file() else []


def ledger_summary(path: Path, digest: str, prereg_sha256: str) -> LedgerSummary:
    records = _records(path)
    mine = [r for r in records if r.get("instrument_digest") == digest]
    return LedgerSummary(count=len(mine), total=len(records),
                         versions=len({r.get("instrument_digest") for r in records}),
                         prereg_matches=bool(mine) and all(r.get("prereg_sha256") == prereg_sha256 for r in mine))
