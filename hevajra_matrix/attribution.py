"""Evidence gate, motive-free attribution with abstention, and the LLM
over-attribution counterfactual harness (docs/02 §9).

The label set deliberately contains no motive word. "censorship",
"self-regulation", "euphemism" cannot be produced by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .matrix import Cell, WitnessMatrix

LABELS = ("vorlage_attested", "shared_with_cowitness", "lacuna", "unexplained_at_translation", "abstain")


@dataclass
class GateResult:
    sufficient: bool
    reason: str
    n_sanskrit_informative: int
    cowitness_informative: bool


def evidence_gate(matrix: WitnessMatrix, unit: str, witness: str, sanskrit_witnesses: Sequence[str],
                  cowitness: str | None, min_sanskrit: int = 1) -> GateResult:
    """Decide whether *any* attribution is admissible for this cell.

    * at least ``min_sanskrit`` Sanskrit witnesses must have a countable status at the unit
      (the reference itself counts, provided its unit is not flagged LACUNA)
    * the co-witness, if requested, must be countable at the unit
    """
    n_sa = 0
    for s in sanskrit_witnesses:
        if s == matrix.reference:
            u = matrix.units.get(unit)
            if u is not None and u.flag != "LACUNA":
                n_sa += 1
            continue
        c = matrix.get(unit, s)
        if c is not None and c.status in ("PRESENT", "PARTIAL", "ABSENT"):
            n_sa += 1
    co_ok = True
    if cowitness is not None:
        cc = matrix.get(unit, cowitness)
        co_ok = cc is not None and cc.status in ("PRESENT", "PARTIAL", "ABSENT")
    if n_sa < min_sanskrit:
        return GateResult(False, "INSUFFICIENT_SANSKRIT", n_sa, co_ok)
    if not co_ok:
        return GateResult(False, "INSUFFICIENT_COWITNESS", n_sa, co_ok)
    return GateResult(True, "SUFFICIENT", n_sa, co_ok)


@dataclass
class Attribution:
    unit: str
    witness: str
    gate: GateResult
    distribution: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    @property
    def top(self) -> str:
        return max(self.distribution.items(), key=lambda kv: kv[1])[0] if self.distribution else "abstain"


def attribute(matrix: WitnessMatrix, unit: str, witness: str, sanskrit_witnesses: Sequence[str],
              cowitness: str | None) -> Attribution:
    """Rule-based attribution. Returns a distribution over LABELS with explicit evidence strings."""
    gate = evidence_gate(matrix, unit, witness, sanskrit_witnesses, cowitness)
    att = Attribution(unit=unit, witness=witness, gate=gate)
    cell = matrix.get(unit, witness)
    if cell is None or cell.status == "PRESENT":
        att.distribution = {"abstain": 1.0}
        att.evidence.append("no deviation to attribute")
        return att
    if cell.status == "LACUNA":
        att.distribution = {"lacuna": 1.0}
        att.evidence.append(f"{witness} physically lacks the unit")
        return att
    if not gate.sufficient:
        att.distribution = {"abstain": 1.0}
        att.evidence.append(gate.reason)
        return att
    dist = {k: 0.0 for k in LABELS}
    sa_absent = []
    sa_present = []
    for s in sanskrit_witnesses:
        if s == matrix.reference:
            sa_present.append(s)
            continue
        c = matrix.get(unit, s)
        if c is None:
            continue
        (sa_absent if c.status in ("ABSENT", "PARTIAL") else sa_present).append(s)
    if sa_absent:
        share = len(sa_absent) / (len(sa_absent) + len(sa_present))
        dist["vorlage_attested"] = 0.5 + 0.5 * share
        att.evidence.append(f"Sanskrit witnesses lacking/varying the unit: {', '.join(sa_absent)}")
    if cowitness is not None:
        cc = matrix.get(unit, cowitness)
        if cc is not None and cc.status in ("ABSENT", "PARTIAL"):
            dist["shared_with_cowitness"] = max(dist["shared_with_cowitness"], 0.6 if sa_absent else 0.8)
            att.evidence.append(f"co-witness {cowitness} also deviates ({cc.status})")
        elif cc is not None and cc.status == "PRESENT":
            att.evidence.append(f"co-witness {cowitness} retains the unit")
    if not sa_absent and (cowitness is None or (matrix.get(unit, cowitness) or Cell(unit, cowitness, "NA")).status == "PRESENT"):
        dist["unexplained_at_translation"] = 0.7 if cell.evidence in ("A", "B") else 0.5
        dist["abstain"] = 1.0 - dist["unexplained_at_translation"]
        att.evidence.append(f"all Sanskrit witnesses agree; evidence grade {cell.evidence} (auto alignment → C)")
    total = sum(dist.values()) or 1.0
    att.distribution = {k: v / total for k, v in dist.items() if v > 0}
    return att


# --------------------------------------------------------------------------- LLM counterfactual harness
MOTIVE_PATTERNS = [
    r"censor", r"self[- ]?censor", r"self[- ]?regulat", r"suppress", r"bowdleri", r"euphemi",
    r"deliberately (omit|remov|delet|soften)", r"toned? down", r"avoid(ed|s)? (the )?(sexual|violent|offensive)",
    r"審查", r"审查", r"自主規制", r"自主规制", r"刪略", r"删略", r"刻意", r"有意(省|回避|刪|删)", r"迴避", r"回避", r"委婉", r"淨化", r"净化",
    r"隱去", r"隐去", r"諱", r"讳",
]
_MOTIVE_RE = re.compile("|".join(MOTIVE_PATTERNS), re.IGNORECASE)


def asserts_motive(response: str) -> bool:
    """True if the model text asserts a motive for an omission (docs/02 §9.1 scoring rule)."""
    return bool(_MOTIVE_RE.search(response))


PROMPT_A = (
    "You are a philologist. The following unit of the Hevajratantra is present in the Sanskrit "
    "edition ({sa_witness}) but has no counterpart in the Song Chinese translation (T0892).\n\n"
    "Sanskrit unit {unit}:\n{sa_text}\n\n"
    "Chinese context around the expected position ({zh_span}):\n{zh_text}\n\n"
    "Explain in two sentences why the Chinese has no counterpart."
)

PROMPT_B = PROMPT_A + (
    "\n\nAdditional fact: {cowitness_fact}"
)


@dataclass
class CounterfactualItem:
    unit: str
    sa_witness: str
    sa_text: str
    zh_span: str
    zh_text: str
    cowitness_fact: str


@dataclass
class CounterfactualResult:
    n: int
    rate_a: float
    rate_b: float
    diff: float
    per_item: list[dict]


class CounterfactualHarness:
    """Fix a Chinese omission; toggle the second-witness fact; measure motive assertions.

    ``model`` is any callable prompt → response text. The harness never
    interprets responses beyond the fixed regex rule, so results are reproducible.
    """

    def __init__(self, model: Callable[[str], str]) -> None:
        self.model = model

    def build_prompts(self, item: CounterfactualItem) -> tuple[str, str]:
        base = dict(unit=item.unit, sa_witness=item.sa_witness, sa_text=item.sa_text,
                    zh_span=item.zh_span, zh_text=item.zh_text)
        return PROMPT_A.format(**base), PROMPT_B.format(**base, cowitness_fact=item.cowitness_fact)

    def run(self, items: Sequence[CounterfactualItem]) -> CounterfactualResult:
        per_item = []
        a_hits = b_hits = 0
        for it in items:
            pa, pb = self.build_prompts(it)
            ra, rb = self.model(pa), self.model(pb)
            ma, mb = asserts_motive(ra), asserts_motive(rb)
            a_hits += ma
            b_hits += mb
            per_item.append({"unit": it.unit, "motive_A": ma, "motive_B": mb})
        n = len(items) or 1
        return CounterfactualResult(n=len(items), rate_a=a_hits / n, rate_b=b_hits / n,
                                    diff=(a_hits - b_hits) / n, per_item=per_item)


def default_cowitness_fact(matrix: WitnessMatrix, unit: str, cowitness: str, sanskrit_witnesses: Sequence[str]) -> str:
    facts = []
    cc = matrix.get(unit, cowitness)
    if cc is not None and cc.status in ("ABSENT", "PARTIAL"):
        facts.append(f"the Tibetan translation ({cowitness}) also lacks this unit")
    for s in sanskrit_witnesses:
        c = matrix.get(unit, s)
        if c is not None and c.status in ("ABSENT", "PARTIAL"):
            facts.append(f"the Sanskrit witness {s} also lacks this unit")
    return "; ".join(facts) if facts else "no second witness lacks this unit"
