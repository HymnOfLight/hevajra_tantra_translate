"""The run summary (``summary.md``), gated by report level (synthesis 5.5, 6.1; critique A7).

    level 0 DESCRIPTIVE  any of G0-G2 failed: counts labelled "unvalidated instrument
                         output" with the controls beside them (P1, B0, external; never
                         P2, whose counts equal Claude's by construction), sentinel
                         results, scope lines. No rate, interval or contrast is printed,
                         and numbers inside gate reasons are withheld.
    level 1 VALIDATED    G0-G2 passed: adds alignment scores with intervals and the list
                         of witness-only material (E5 list)
    level 2 CALIBRATED   G3 passed too: adds the estimates with their intervals and the
                         Manski bounds; E3/E4 only where gate G4 passes

At every level: human-verified (grade A) counts, printed as instrument-independent lower
bounds; the confirmatory/exploratory line; and for every estimand that cannot be printed
``NOT_ESTIMABLE: <gate>: <reason>``, never a zero. ``render`` is pure: the pipeline
collects ``ReportInputs`` from the run directory.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..core.types import Alignment, Cell, Estimate, Grade, OutcomeClass, Relation
from ..evaluation.gate import GateReport
from ..evaluation.sentinels import SentinelResult
from ..matrix.status import cell_outcome, outcome_class

LEVEL_NAMES = {0: "DESCRIPTIVE", 1: "VALIDATED", 2: "CALIBRATED"}
UNVALIDATED = "unvalidated instrument output"
COUNT_ROWS = (
    ("nondev", "present, no deviation"),
    ("dev_present", "present, rewritten (substitution, reversal, ...)"),
    ("partial", "partial (abridged)"),
    ("absent", "absent (no counterpart)"),
    ("unaligned", "unaligned (no decision)"),
    ("witness_only", "witness-only segments"),
)
SOURCE_LABELS = {"claude:consensus": "Claude consensus", "dp:zero": "P1 length-only DP",
                 "dp:anchor": "B0 anchor DP", "placebo:shuffled": "P2 shuffled Claude"}
PLACEBO_PREFIX = "placebo:"
ESTIMANDS = (
    ("E1_any", "E1 share of units deviating (D_any)"),
    ("E1_cov", "E1 share of units losing coverage (D_cov, partial or absent)"),
    ("E1_absent", "E1 share of units absent"),
    ("E1_partial", "E1 share of units partial"),
    ("E2_any", "E2 syllable-weighted share deviating"),
    ("E3", "E3 decomposition of deviations (Vorlage, shared, residual)"),
    ("E4", "E4 sensitive-vs-neutral contrast"),
    ("E5", "E5 witness-only material (human-verified segments)"),
)
CALIBRATED = 2                        # every estimate needs report level 2
SCORE_COLUMNS = (("link_f1", "link F1"), ("status_kappa", "status kappa"), ("null_precision", "NULL precision"),
                 ("null_recall", "NULL recall"), ("witness_only_recall", "witness-only recall"))
_DECIMAL = re.compile(r"(?<![\w.:])-?\d+\.\d+(?![\w.])")
_INTERVAL = re.compile(r"\[[^\]]*\d[^\]]*\]")
WITHHELD = "<withheld>"
_SENTINEL_REASON = ": verified sentinel "


@dataclass(frozen=True)
class WitnessOnlyRow:
    row_id: str
    chapter: str
    kind: str
    grade: str


@dataclass(frozen=True)
class ReportInputs:
    """Everything ``render`` prints, already read from the run directory.

    ``instrument_counts``  outcome -> count for the Claude consensus (None: no instrument
                           output in this run); keys as in ``COUNT_ROWS``
    ``control_counts``     control source -> the same counts (placebo sources are refused)
    ``human_verified``     outcome -> number of grade A cells (orphans as witness_only)
    ``scores``             source -> metric -> ``Estimate`` with its bootstrap interval
    ``estimates``          estimand name -> ``Estimate`` (from the stats stage)
    ``manski``             outcome -> (lo, hi) share bounds over unresolved units
    ``notes``              translator notes relevant to the decomposition (id, class, host)
    """

    run_id: str
    reference: str
    witness: str
    scope: tuple[str, ...]
    instrument_counts: Mapping[str, int] | None = None
    control_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    human_verified: Mapping[str, int] = field(default_factory=dict)
    sentinels: tuple[SentinelResult, ...] = ()
    gold_set: str = ""
    scores: Mapping[str, Mapping[str, Estimate]] = field(default_factory=dict)
    witness_only: tuple[WitnessOnlyRow, ...] = ()
    estimates: Mapping[str, Estimate] = field(default_factory=dict)
    manski: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    max_manski_width: float = 0.05
    notes: tuple[tuple[str, str, str], ...] = ()


# --------------------------------------------------------------------------- counting (pure)
def alignment_counts(alignment: Alignment, ref_kinds: Mapping[str, str]) -> dict[str, int]:
    """Outcome counts of an aligner over the units of ``ref_kinds`` (missing units: unaligned)."""
    links = alignment.by_ref()
    counts = Counter({key: 0 for key, _ in COUNT_ROWS})
    for uid, kind in ref_kinds.items():
        link = links.get(uid)
        if link is None:
            counts["unaligned"] += 1
        else:
            counts[outcome_class(Relation(link.relation), link.polarity_flip, kind).value] += 1
    counts["witness_only"] = sum(len(link.wit_ids) for link in alignment.witness_only())
    return dict(counts)


def human_verified_counts(cells: Iterable[Cell], ref_kinds: Mapping[str, str]) -> dict[str, int]:
    """Grade A cells by outcome; orphan rows count as witness-only segments."""
    counts = Counter({key: 0 for key, _ in COUNT_ROWS})
    for c in cells:
        if c.grade is not Grade.A:
            continue
        if c.unit_id.startswith("+"):
            counts["witness_only"] += 1
            continue
        outcome = cell_outcome(c, ref_kinds.get(c.unit_id, ""))
        counts[outcome.value if isinstance(outcome, OutcomeClass) else "unaligned"] += 1
    return dict(counts)


# --------------------------------------------------------------------------- rendering
def render(inputs: ReportInputs, gate: GateReport) -> str:
    """The Markdown summary for ``inputs`` at ``gate.level`` (English)."""
    if any(src.startswith(PLACEBO_PREFIX) for src in inputs.control_counts):
        raise ValueError("P2 (placebo) counts equal the instrument's by construction; never pass them as counts")
    level = gate.level
    out = ["# Hevajratantra collation: run summary", "",
           f"Run `{inputs.run_id}`: reference `{inputs.reference}`, witness `{inputs.witness}`.", "",
           f"**Report level {level} ({LEVEL_NAMES.get(level, '?')})**: " + _gate_line(gate), "",
           _confirmatory_line(gate), "", "## Scope", "", *[f"- {s}" for s in inputs.scope], ""]
    out += _gate_findings(gate)
    out += _counts(inputs, level)
    out += _human_verified(inputs)
    out += _sentinels(inputs.sentinels, level)
    if level >= 1:
        out += _scores(inputs) + _witness_only(inputs)
    out += _estimates(inputs, gate)
    out += _notes(inputs)
    return "\n".join(out).rstrip() + "\n"


def _gate_line(gate: GateReport) -> str:
    if not gate.passed:
        return "the gates have not been evaluated in this run."
    return ", ".join(f"{g} {'passed' if ok else 'failed'}" for g, ok in sorted(gate.passed.items())) + "."


def _confirmatory_line(gate: GateReport) -> str:
    if gate.confirmatory:
        return ("**Confirmatory**: the preregistration matches and the test set was scored exactly once by "
                "this instrument.")
    note = (gate.scope_note or "the gates have not been evaluated").removeprefix("exploratory; ")
    return f"**Exploratory**: {note}."


def redact(text: str, level: int) -> str:
    """At level 0, numbers with a decimal point and bracketed intervals are withheld."""
    if level >= 1:
        return text
    return _DECIMAL.sub(WITHHELD, _INTERVAL.sub(WITHHELD, text))


def _gate_findings(gate: GateReport) -> list[str]:
    """Gate reasons; failed sentinels are counted per gate (they are listed under Sentinels)."""
    if not gate.reasons:
        return []
    out = ["## Gate findings", ""]
    if gate.level == 0:
        out += ["Values are withheld at level 0; see `evaluation/gate.json`.", ""]
    sentinel_failures = Counter(r.split(":", 1)[0] for r in gate.reasons if _SENTINEL_REASON in r)
    out += [f"- {redact(r, gate.level)}" for r in gate.reasons if _SENTINEL_REASON not in r]
    out += [f"- {g}: {n} verified sentinel(s) failed (see Sentinels)" for g, n in sorted(sentinel_failures.items())]
    return out + [""]


def _label(source: str) -> str:
    if source.startswith("external:"):
        return f"external {source.split(':', 1)[1]}"
    return SOURCE_LABELS.get(source, source)


def _counts(inputs: ReportInputs, level: int) -> list[str]:
    title = f"## Counts ({UNVALIDATED})" if level == 0 else "## Counts (instrument output)"
    columns = ([("Claude consensus", inputs.instrument_counts)] if inputs.instrument_counts is not None else []) + [
        (_label(src), counts) for src, counts in inputs.control_counts.items()]
    out = [title, ""]
    if inputs.instrument_counts is None:
        out += ["No instrument output in this run (collate was not run or was skipped).", ""]
    if not columns:
        return out + ["No alignment in this run.", ""]
    out += ["| outcome | " + " | ".join(name for name, _ in columns) + " |",
            "|---|" + "---|" * len(columns)]
    out += [f"| {label} | " + " | ".join(str(counts.get(key, 0)) for _, counts in columns) + " |"
            for key, label in COUNT_ROWS]
    return out + ["", "Controls are content-free (P1) or lexical (B0) aligners computed on the same units. "
                  "P2 (shuffled Claude) is not shown: its counts equal Claude's by construction.", ""]


def _human_verified(inputs: ReportInputs) -> list[str]:
    out = ["## Human-verified counts (grade A)", "",
           "Instrument-independent lower bounds: units decided by blind gold or a final review verdict.", "",
           "| outcome | units |", "|---|---|"]
    return out + [f"| {label} | {inputs.human_verified.get(key, 0)} |" for key, label in COUNT_ROWS] + [""]


def own_stage_results(results: Sequence[SentinelResult]) -> list[SentinelResult]:
    """One result per sentinel: the check at its own stage (later stages repeat earlier checks)."""
    return [r for r in results if r.stage == r.sentinel_stage]


def _sentinels(results: Sequence[SentinelResult], level: int) -> list[str]:
    out = ["## Sentinels", ""]
    results = own_stage_results(results)
    if not results:
        return out + ["No sentinel was checked in this run.", ""]
    out += ["| sentinel | stage | status | result |", "|---|---|---|---|"]
    for r in results:
        result = "pass" if r.passed else f"FAIL: {redact(r.detail, level)}".replace("|", "/")
        out.append(f"| {r.sentinel_id} | {r.stage} | {r.status}{' (blocking)' if r.blocking else ''} | {result} |")
    return out + [""]


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _interval(e: Estimate | None) -> str:
    if e is None or e.point is None:
        return "n/a"
    return f"{_num(e.point)} [{_num(e.lo)}, {_num(e.hi)}]"


def _scores(inputs: ReportInputs) -> list[str]:
    out = [f"## Alignment scores (gold: {inputs.gold_set or 'none'})", ""]
    if not inputs.scores:
        return out + ["No gold scored in this run.", ""]
    out += ["95% window-cluster bootstrap intervals, all aligners on the same units.", "",
            "| aligner | " + " | ".join(label for _, label in SCORE_COLUMNS) + " |",
            "|---|" + "---|" * len(SCORE_COLUMNS)]
    out += [f"| {_label(src)} | " + " | ".join(_interval(m.get(key)) for key, _ in SCORE_COLUMNS) + " |"
            for src, m in inputs.scores.items()]
    return out + [""]


def _witness_only(inputs: ReportInputs) -> list[str]:
    out = ["## Witness-only material (E5 list)", ""]
    if not inputs.witness_only:
        return out + ["No witness-only row.", ""]
    out += ["| row | chapter | kind | grade |", "|---|---|---|---|"]
    return out + [f"| {r.row_id} | {r.chapter} | {r.kind} | {r.grade} |" for r in inputs.witness_only] + [""]


def blocking_reason(gate: GateReport, needed: int) -> str | None:
    """Why the report level is below ``needed``: the first reason of the first failing gate."""
    if gate.level >= needed:
        return None
    gates = ("G0", "G1", "G2") if gate.level < 1 else ("G3",)
    for g in gates:
        if not gate.passed.get(g, False):
            return next((r for r in gate.reasons if r.startswith(f"{g}:")), f"{g}: not passed")
    return f"{gates[-1]}: not passed"


def not_estimable_reason(name: str, estimate: Estimate | None, gate: GateReport) -> str | None:
    """Why estimand ``name`` cannot be printed, or None.

    E3's G4 blocker (no manuscript column) is structural and shown at every level; for the
    others the report level comes first, then gate G4, then the stats stage's own reason.
    """
    g4 = gate.not_estimable.get(name.split("_")[0])
    if name == "E3" and g4:
        return g4
    if estimate is None:
        return blocking_reason(gate, CALIBRATED) or g4 or "stats: not computed in this run"
    return blocking_reason(gate, CALIBRATED) or g4 or estimate.not_estimable


def _estimates(inputs: ReportInputs, gate: GateReport) -> list[str]:
    out = ["## Estimates", "", "Every number is relative to the scope stated above.", ""]
    for name, label in ESTIMANDS:
        estimate = inputs.estimates.get(name)
        reason = not_estimable_reason(name, estimate, gate)
        if reason or estimate is None:
            out.append(f"- {label}: NOT_ESTIMABLE: {redact(reason or '', gate.level)}")
        elif name == "E5":
            out.append(f"- {label}: {int(estimate.point or 0)} segments, {estimate.n} characters "
                       "(a census of human-verified claims; no interval)")
        else:
            out.append(f"- {label}: {_interval(estimate)} (n = {estimate.n}; covers: {', '.join(estimate.sources)})")
            blind = inputs.estimates.get(f"{name}_blind")
            if blind is not None and blind.point is not None:
                out.append(f"  - automation-bias sensitivity (blind verdicts): {_interval(blind)}")
    if gate.level >= CALIBRATED and inputs.manski:
        out += ["", "Manski bounds over units still unresolved after review:"]
        for outcome, (lo, hi) in inputs.manski.items():
            wide = " (wider than the gate allows)" if hi - lo > inputs.max_manski_width else ""
            out.append(f"- {outcome}: [{_num(lo)}, {_num(hi)}]{wide}")
    return out + [""]


def _notes(inputs: ReportInputs) -> list[str]:
    if not inputs.notes:
        return []
    out = ["## Translator notes bearing on the decomposition (descriptive)", "",
           "Under the Derge reference no deviating unit carries such a note (critique A3); they are listed, "
           "not counted.", "", "| note | class | host segment |", "|---|---|---|"]
    return out + [f"| {n} | {cls} | {host} |" for n, cls, host in inputs.notes] + [""]
