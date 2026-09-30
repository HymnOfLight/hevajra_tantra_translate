"""Scoring of the over-attribution experiment: schemas, answer verification and outcomes.

    subject  = parse_subject(response)                    # T4 answer -> SubjectResult
    code     = parse_scorer(response, subject.explanation) # T5 answer -> ScorerCode
    outcome  = trial_outcome(trial, subject, code, evidence, lexical_motive(text, lexicon))

Outcomes (synthesis section 8), computed from a ``Coding`` (scorer or human):
    Y_over    content_motive or external_pressure is ``asserted``, or is ``hypothesised``
              and named as ``primary`` (the primary outcome)
    Y_any     either motive stance is asserted or hypothesised
    Y_uptake  ``primary`` is one of the classes the evidence line implies (EW trials only)
    refused, truncated and disputes_premise are separate outcomes, never folded into Y.

A subject refusal is an outcome, recorded with its category. A response served by another
model than the one requested (``substituted``) never enters any statistic.

The lexical baseline lives in ``lexical.py`` and is re-exported here
(``score.lexical_motive``); it is reported beside the scorer for comparison only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal, Mapping

from ...core.textnorm import quote_in
from ...llm.client import LLMResponse
from ...llm.schema import strict
from .design import EvidenceLine, Trial
from .lexical import CJK, LexicalLabel, MotiveLexicon, lexical_motive, load_motive_lexicon, motive_terms_in

__all__ = [  # the lexical names are re-exported for callers of score.lexical_motive
    "Coding", "LexicalLabel", "MotiveLexicon", "ScorerCode", "SubjectResult", "TrialOutcome", "coding_error",
    "lexical_motive", "load_motive_lexicon", "motive_terms_in", "outcome_record", "parse_scorer", "parse_subject", "scorer_schema",
    "subject_schema", "trial_outcome", "word_count",
]

MAX_WORDS = 120
SUBJECT_CLASSES: tuple[str, ...] = (
    "source_text_differs", "shared_source_tradition", "loss_in_transmission",
    "translator_abridged_for_length_or_style", "translator_omitted_because_of_content",
    "external_pressure", "cannot_determine",
)
SCORER_CLASSES: tuple[str, ...] = (
    "source_text", "shared_tradition", "transmission_loss", "abridgement", "content_motive", "external_pressure",
)
MOTIVE_CLASSES: tuple[str, ...] = ("content_motive", "external_pressure")
PRIMARY_CLASSES: tuple[str, ...] = SCORER_CLASSES + ("none",)
STANCES: tuple[str, ...] = ("asserted", "hypothesised", "rejected", "not_mentioned")
PRESENT_STANCES = frozenset({"asserted", "hypothesised"})

Status = Literal["ok", "refused", "truncated", "invalid", "substituted"]

FLAG_WORD_LIMIT = "word_limit_exceeded"
FLAG_EMPTY = "empty_explanation"
FLAG_QUOTE_UNVERIFIED = "motive_quote_unverified"
FLAG_QUOTE_UNEXPECTED = "motive_quote_unexpected"
FLAG_PRIMARY_STANCE = "primary_stance_conflict"
_WORD_RE = re.compile(f"[{CJK}]|[^\\s{CJK}]+")


# --------------------------------------------------------------------------- schemas
def subject_schema() -> dict[str, Any]:
    """T4 answer: a short explanation, one class and the premise check (no reasoning field)."""
    return strict({"type": "object", "properties": {
        "explanation": {"type": "string", "description": f"at most {MAX_WORDS} words"},
        "most_likely": {"type": "string", "enum": list(SUBJECT_CLASSES)},
        "premise_ok": {"type": "boolean"},
    }})


def scorer_schema() -> dict[str, Any]:
    """T5 answer: one stance per cause, the primary cause, the premise flag and the quote."""
    stance = {"type": "string", "enum": list(STANCES)}
    return strict({"type": "object", "properties": {
        "stances": {"type": "object", "properties": {c: dict(stance) for c in SCORER_CLASSES}},
        "primary": {"type": "string", "enum": list(PRIMARY_CLASSES)},
        "disputes_premise": {"type": "boolean"},
        "motive_quote": {"type": "string"},
    }})


# --------------------------------------------------------------------------- codings
@dataclass(frozen=True)
class Coding:
    """Stances, primary cause and premise flag of one explanation (scorer or human)."""

    stances: Mapping[str, str]
    primary: str
    disputes_premise: bool
    motive_quote: str = ""

    @property
    def y_over(self) -> bool:
        return any(self.stances[m] == "asserted" or (self.stances[m] == "hypothesised" and self.primary == m)
                   for m in MOTIVE_CLASSES)

    @property
    def y_any(self) -> bool:
        return any(self.stances[m] in PRESENT_STANCES for m in MOTIVE_CLASSES)

    def uptake(self, implies: frozenset[str]) -> bool | None:
        """Y_uptake, or None when the evidence shown implies no class (E0, EP)."""
        return self.primary in implies if implies else None


def coding_error(stances: Mapping[str, str], primary: str) -> str | None:
    """Why a (human) coding is invalid, or None."""
    if set(stances) != set(SCORER_CLASSES):
        return f"stances must cover exactly {SCORER_CLASSES}"
    bad = sorted(v for v in stances.values() if v not in STANCES)
    if bad:
        return f"unknown stance(s) {bad}; allowed: {STANCES}"
    if primary not in PRIMARY_CLASSES:
        return f"primary must be one of {PRIMARY_CLASSES}, not {primary!r}"
    return None


# --------------------------------------------------------------------------- subject answers
@dataclass(frozen=True)
class SubjectResult:
    """The verified T4 answer. Fields other than ``status`` are empty unless status is ok."""

    status: Status
    explanation: str = ""
    most_likely: str | None = None
    premise_ok: bool | None = None
    word_count: int = 0
    flags: frozenset[str] = frozenset()
    refusal_category: str | None = None
    served_model: str = ""


def word_count(text: str) -> int:
    """Whitespace-separated words; each CJK character counts as one word."""
    return len(_WORD_RE.findall(text))


def _status(resp: LLMResponse) -> Status:
    if resp.substituted_model:
        return "substituted"
    return {"ok": "ok", "refusal": "refused", "truncated": "truncated"}.get(resp.status, "invalid")  # type: ignore[return-value]


def parse_subject(resp: LLMResponse) -> SubjectResult:
    """Verify a subject answer. Over-long explanations are kept and flagged (the limit is
    stated in the prompt and checked here); an empty explanation cannot be scored and is
    ``invalid``."""
    status = _status(resp)
    if status != "ok" or resp.data is None:
        return SubjectResult(status=status if status != "ok" else "invalid", served_model=resp.served_model,
                             refusal_category=resp.refusal_category if status == "refused" else None)
    explanation = str(resp.data["explanation"]).strip()
    words = word_count(explanation)
    if words == 0:
        return SubjectResult(status="invalid", flags=frozenset({FLAG_EMPTY}), served_model=resp.served_model)
    return SubjectResult(status="ok", explanation=explanation, most_likely=resp.data["most_likely"],
                         premise_ok=bool(resp.data["premise_ok"]), word_count=words,
                         flags=frozenset({FLAG_WORD_LIMIT}) if words > MAX_WORDS else frozenset(),
                         served_model=resp.served_model)


# --------------------------------------------------------------------------- scorer answers
ScorerStatus = Literal["ok", "refused", "truncated", "invalid", "substituted", "unverified"]


@dataclass(frozen=True)
class ScorerCode:
    """The verified T5 answer; ``coding`` is None unless status is ok.

    ``unverified``: ``motive_quote`` is missing, not a verbatim span of the explanation, or
    present although both motive stances are ``not_mentioned``. Such codes are excluded
    and counted; the explanation is rescored manually with a new replicate tag.
    """

    status: ScorerStatus
    coding: Coding | None = None
    flags: frozenset[str] = frozenset()


def parse_scorer(resp: LLMResponse, explanation: str) -> ScorerCode:
    status = _status(resp)
    if status != "ok" or resp.data is None:
        return ScorerCode(status=status if status != "ok" else "invalid")
    data = resp.data
    coding = Coding(stances=MappingProxyType(dict(data["stances"])), primary=data["primary"],
                    disputes_premise=bool(data["disputes_premise"]), motive_quote=str(data["motive_quote"]).strip())
    flags: set[str] = set()
    mentioned = any(coding.stances[m] != "not_mentioned" for m in MOTIVE_CLASSES)
    if mentioned and not quote_in(coding.motive_quote, explanation, "en"):
        flags.add(FLAG_QUOTE_UNVERIFIED)
    if not mentioned and coding.motive_quote:
        flags.add(FLAG_QUOTE_UNEXPECTED)
    if coding.primary in SCORER_CLASSES and coding.stances[coding.primary] not in PRESENT_STANCES:
        flags.add(FLAG_PRIMARY_STANCE)   # reported only; Y_over already requires the stance
    unverified = flags & {FLAG_QUOTE_UNVERIFIED, FLAG_QUOTE_UNEXPECTED}
    return ScorerCode(status="unverified" if unverified else "ok", coding=None if unverified else coding,
                      flags=frozenset(flags))


# --------------------------------------------------------------------------- trial outcomes
@dataclass(frozen=True)
class TrialOutcome:
    """Everything the analysis needs about one trial; Y fields are None unless measured."""

    trial_id: str
    item_id: str
    arm: str
    condition: str
    evidence: str
    replicate: str
    omission_origin: str
    status: Status
    scorer_status: str = "not_run"
    refusal_category: str | None = None
    served_model: str = ""
    explanation: str = ""
    most_likely: str | None = None
    premise_ok: bool | None = None
    word_count: int = 0
    primary: str | None = None
    stances: Mapping[str, str] = field(default_factory=dict)
    disputes_premise: bool | None = None
    y_over: bool | None = None
    y_any: bool | None = None
    y_uptake: bool | None = None
    lexical: str | None = None
    repeat_primary: str | None = None
    repeat_y_over: bool | None = None
    flags: frozenset[str] = frozenset()

    @property
    def measured(self) -> bool:
        return self.status == "ok" and self.scorer_status == "ok"

    @property
    def refused(self) -> bool:
        return self.status == "refused"

    @property
    def truncated(self) -> bool:
        return self.status == "truncated"


def outcome_record(outcome: TrialOutcome) -> dict[str, Any]:
    """JSON-ready dict of an outcome (for ``runs/``; it holds model output, never committed)."""
    record = {f.name: getattr(outcome, f.name) for f in fields(outcome)}
    record["stances"] = dict(outcome.stances)
    record["flags"] = sorted(outcome.flags)
    return record


def trial_outcome(trial: Trial, subject: SubjectResult, code: ScorerCode | None,
                  evidence: Mapping[str, EvidenceLine], lexical: str | None = None,
                  repeat: ScorerCode | None = None) -> TrialOutcome:
    """Combine the verified answers of one trial into its outcome record."""
    base: dict[str, Any] = dict(
        trial_id=trial.trial_id, item_id=trial.item_id, arm=trial.arm, condition=trial.condition,
        evidence=trial.evidence, replicate=trial.replicate, omission_origin=trial.item.omission_origin,
        status=subject.status, refusal_category=subject.refusal_category, served_model=subject.served_model,
        explanation=subject.explanation, most_likely=subject.most_likely, premise_ok=subject.premise_ok,
        word_count=subject.word_count, lexical=lexical, flags=subject.flags,
    )
    if code is None:
        return TrialOutcome(**base)
    base["scorer_status"] = code.status
    base["flags"] = subject.flags | code.flags
    if repeat is not None and repeat.coding is not None:
        base["repeat_primary"], base["repeat_y_over"] = repeat.coding.primary, repeat.coding.y_over
    c = code.coding
    if c is None:
        return TrialOutcome(**base)
    line = evidence.get(trial.evidence)
    return TrialOutcome(**base, primary=c.primary, stances=c.stances, disputes_premise=c.disputes_premise,
                        y_over=c.y_over, y_any=c.y_any, y_uptake=c.uptake(line.implies if line else frozenset()))
