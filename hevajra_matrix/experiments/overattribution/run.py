"""Requests of the over-attribution experiment and the loop that runs a schedule.

    resolver = segment_resolver(reference_segments, witness_segments)   # texts from raw
    request  = subject_request(trial, resolver, subject_settings, evidence)   # T4
    request  = scorer_request(explanation, scorer_settings)                   # T5
    outcomes = run_trials(schedule, resolver, client, ...)               # calls + scoring

T4 (subject) sees the fixed system prompt, an identity line, the Tibetan passage, the
Chinese context with the passage absent (3 clauses either side), the evidence line of its
condition (none under E0) and one fixed question. The prompts never ask for reasoning.

T5 (scorer) sees ONLY the subject's explanation: no item, arm, condition, evidence line or
coordinates, so its coding is blind by construction. Identical explanations therefore
share one cache entry and one coding, which is the intended behaviour.

Server-side fallback is OFF for both tasks (researcher decision): a substituted response
would put replicates on another model. ``ExperimentTaskSettings`` refuses ``fallback:
true``. Results are specific to the requested model (claude-opus-5-5) at the campaign date;
the served model is recorded on every outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Callable, Iterable, Mapping, Sequence, get_args

from ...config import ConfigError, from_mapping
from ...core.types import Segment
from ...ingest import CONTENT_KINDS
from ...llm.client import DEFAULT_MODEL, Effort, LLMClient, LLMRequest, canonical_json, sha256_text
from .design import EvidenceLine, Item, Trial
from .score import (
    MAX_WORDS,
    MotiveLexicon,
    TrialOutcome,
    lexical_motive,
    parse_scorer,
    parse_subject,
    scorer_schema,
    subject_schema,
    trial_outcome,
)

SUBJECT_TASK, SCORER_TASK = "subject", "scorer"
PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
SUBJECT_PROMPT = PROMPTS / "subject.v1.md"
SCORER_PROMPT = PROMPTS / "scorer.v1.md"
USER_SEPARATOR = "=====USER====="
EVIDENCE_PREFIX = "Additional information: "
EXPLANATION_HEADER = "Explanation to code:"
CONTEXT_CLAUSES = 3


@dataclass(frozen=True)
class ExperimentTaskSettings:
    """``config/llm.yaml: tasks.subject`` or ``tasks.scorer`` plus the top-level ``model``."""

    effort: Effort = "medium"
    max_tokens: int = 16000
    replicates: int = 1
    fallback: bool = False
    double_score_fraction: float = 0.0
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        if self.effort not in get_args(Effort):
            raise ConfigError(f"experiment effort must be one of {get_args(Effort)}, not {self.effort!r}")
        if self.fallback is not False:
            raise ConfigError("experiment tasks must run with fallback: false (a substituted model "
                              "would contaminate replicates)")
        for name in ("max_tokens", "replicates"):
            value = getattr(self, name)
            if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
                raise ConfigError(f"{name} must be a positive integer, not {value!r}")
        if not 0.0 <= float(self.double_score_fraction) <= 1.0:
            raise ConfigError("double_score_fraction must be between 0 and 1")

    @classmethod
    def from_config(cls, llm: Mapping[str, Any], task: str) -> "ExperimentTaskSettings":
        if task not in (SUBJECT_TASK, SCORER_TASK):
            raise ConfigError(f"task must be {SUBJECT_TASK!r} or {SCORER_TASK!r}, not {task!r}")
        section = dict((llm.get("tasks") or {}).get(task) or {})
        if "model" in llm:
            section.setdefault("model", llm["model"])
        return from_mapping(cls, section, f"llm.yaml: tasks.{task}")


# --------------------------------------------------------------------------- texts
@dataclass(frozen=True)
class ItemText:
    """What the subject is shown of one item. ``removed`` counts the Chinese content
    segments left out between the two context blocks (0 for a real omission)."""

    tibetan: str
    zh_before: str
    zh_after: str
    removed: int = 0


TextResolver = Callable[[Item], ItemText]


def segment_resolver(reference: Sequence[Segment], witness: Sequence[Segment],
                     clauses: int = CONTEXT_CLAUSES) -> TextResolver:
    """A ``TextResolver`` over ingested segments (texts are rebuilt from ``data/raw``).

    The Tibetan passage is the item's reference units joined by a space. The Chinese context
    is ``clauses`` content segments ending at ``zh_context_from`` and ``clauses`` starting at
    ``zh_context_to``, in document order; content segments between them are left out.
    Raises ``ValueError`` for unknown ids, a reversed gap, or a gap that contradicts the
    item's ``omission_origin`` (real: nothing between; constructed: something removed).
    """
    units = {s.id: s for s in reference}
    chinese = [s for s in witness if s.kind in CONTENT_KINDS]
    position = {s.id: i for i, s in enumerate(chinese)}

    def resolve(item: Item) -> ItemText:
        missing = [u for u in item.unit_ids if u not in units]
        missing += [c for c in (item.zh_context_from, item.zh_context_to) if c not in position]
        if missing:
            raise ValueError(f"item {item.item_id}: unknown or non-content segment id(s) {missing}")
        i, j = position[item.zh_context_from], position[item.zh_context_to]
        if i >= j:
            raise ValueError(f"item {item.item_id}: zh_context_from must precede zh_context_to")
        removed = j - i - 1
        if (item.omission_origin == "real") != (removed == 0):
            raise ValueError(f"item {item.item_id}: {item.omission_origin} omission with {removed} "
                             "Chinese segment(s) between the context blocks")
        return ItemText(tibetan=" ".join(units[u].text.strip() for u in item.unit_ids),
                        zh_before="".join(s.text.strip() for s in chinese[max(0, i - clauses + 1):i + 1]),
                        zh_after="".join(s.text.strip() for s in chinese[j:j + clauses]),
                        removed=removed)

    return resolve


# --------------------------------------------------------------------------- requests
@lru_cache(maxsize=4)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def subject_template() -> tuple[str, Template]:
    """(system prompt, user-message template) of ``subject.v1.md``."""
    system, sep, user = _read(SUBJECT_PROMPT).partition(USER_SEPARATOR)
    if not sep:
        raise ValueError(f"{SUBJECT_PROMPT} lacks the {USER_SEPARATOR} line")
    return system.strip(), Template(user.strip("\n"))


def render_subject(trial: Trial, text: ItemText, evidence: Mapping[str, EvidenceLine]) -> tuple[str, str]:
    """(system, body) of the T4 request for ``trial``."""
    system, template = subject_template()
    line = "" if trial.evidence == "none" else f"\n{EVIDENCE_PREFIX}{evidence[trial.evidence].text}\n"
    body = template.substitute(chapter=trial.item.chapter, tibetan=_one_line(text.tibetan),
                               zh_before=_one_line(text.zh_before), zh_after=_one_line(text.zh_after),
                               evidence=line, max_words=MAX_WORDS)
    return system, body + "\n"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def subject_request(trial: Trial, text_resolver: TextResolver, task_settings: ExperimentTaskSettings,
                    evidence: Mapping[str, EvidenceLine]) -> LLMRequest:
    """The T4 request of one trial; ``trial.replicate`` is the cache replicate tag."""
    system, body = render_subject(trial, text_resolver(trial.item), evidence)
    prompt_sha = sha256_text(_read(SUBJECT_PROMPT) + canonical_json({k: v.text for k, v in evidence.items()}))
    return LLMRequest(task=SUBJECT_TASK, prompt_sha=prompt_sha, system=system, body=body,
                      schema=subject_schema(), effort=task_settings.effort, max_tokens=task_settings.max_tokens,
                      replicate=trial.replicate, allow_fallback=False, model=task_settings.model)


def scorer_request(explanation: str, task_settings: ExperimentTaskSettings, replicate: str = "r1") -> LLMRequest:
    """The T5 request: the scorer template and the explanation, and nothing else."""
    system = _read(SCORER_PROMPT).strip()
    return LLMRequest(task=SCORER_TASK, prompt_sha=sha256_text(system), system=system,
                      body=f"{EXPLANATION_HEADER}\n{explanation.strip()}\n", schema=scorer_schema(),
                      effort=task_settings.effort, max_tokens=task_settings.max_tokens, replicate=replicate,
                      allow_fallback=False, model=task_settings.model)


# --------------------------------------------------------------------------- running
def run_trials(schedule: Iterable[Trial], text_resolver: TextResolver, client: LLMClient,
               subject_settings: ExperimentTaskSettings, scorer_settings: ExperimentTaskSettings,
               evidence: Mapping[str, EvidenceLine], lexicon: MotiveLexicon,
               double_score: frozenset[str] = frozenset()) -> list[TrialOutcome]:
    """Run every trial in schedule order: subject call, then (for a usable answer) the
    scorer call, plus a second scorer call (replicate "r2") for trials in ``double_score``.

    Refusals, truncations and invalid answers are outcomes, not errors; transport errors,
    budget exhaustion and offline cache misses propagate (rerunning replays the cache).
    """
    outcomes = []
    for trial in sorted(schedule, key=lambda t: t.order):
        subject = parse_subject(client.complete(subject_request(trial, text_resolver, subject_settings, evidence)))
        if subject.status != "ok":
            outcomes.append(trial_outcome(trial, subject, None, evidence))
            continue
        code = parse_scorer(client.complete(scorer_request(subject.explanation, scorer_settings)),
                            subject.explanation)
        repeat = None
        if trial.trial_id in double_score:
            repeat = parse_scorer(client.complete(scorer_request(subject.explanation, scorer_settings, "r2")),
                                  subject.explanation)
        outcomes.append(trial_outcome(trial, subject, code, evidence,
                                      lexical_motive(subject.explanation, lexicon), repeat))
    return outcomes
