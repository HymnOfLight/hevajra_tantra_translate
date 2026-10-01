"""Shared domain vocabulary: statuses, relations, grades and the immutable records
that flow between pipeline stages.

Everything here is a plain value object with no I/O and no dependency on other
package modules. Behaviour lives in the module that owns it: status derivation in
``matrix.status``, id construction in ``core.ids``, verification in
``collate.verify``, and so on. Keeping this module dependency-free lets every other
module import it without cycles.

Key invariants
    * A ``Segment`` id is derived from witness-native coordinates (see ``core.ids``)
      and never from a running counter, so it survives re-ingest.
    * One ``Link`` describes exactly one reference unit (``ref_id``) or one block of
      witness-only material (``ref_id is None``). Every aligner (Claude, the DP
      baselines, the placebos) and the human gold produce the same ``Alignment``
      type, which is what makes their evaluation comparable by construction.
    * Status is never stored on a ``Link``; it is derived from the relation by
      ``matrix.status`` and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping


class StrEnum(str, Enum):
    """``str`` enum that prints as its value (``enum.StrEnum`` needs Python 3.11)."""

    def __str__(self) -> str:
        return str(self.value)


class Status(StrEnum):
    """Cell status. Only PRESENT, PARTIAL and ABSENT enter statistical denominators."""

    PRESENT = "PRESENT"
    PARTIAL = "PARTIAL"
    ABSENT = "ABSENT"
    UNALIGNED = "UNALIGNED"   # not decided; always carries a reason
    LACUNA = "LACUNA"         # the witness is physically damaged or lost here
    NA = "NA"                 # the witness does not cover this unit at all


COUNTED_STATUSES = frozenset({Status.PRESENT, Status.PARTIAL, Status.ABSENT})


class Relation(StrEnum):
    """Textual relation between a reference unit and its witness counterpart.

    The relation is a description, never a cause: no member names a motive.
    """

    EQUIVALENT = "equivalent"
    PARAPHRASE = "paraphrase"
    GENERALISED = "generalised"
    ABRIDGED = "abridged"
    EXPANDED = "expanded"
    SUBSTITUTION = "substitution"
    REVERSAL = "reversal"
    CATEGORY_NAME_OMITTED = "category_name_omitted"
    TRANSLITERATED = "transliterated"
    NO_COUNTERPART = "no_counterpart"


class WitnessOnlyKind(StrEnum):
    """Kind of witness material that has no reference counterpart (orphan rows)."""

    ADDITION = "addition"
    TRANSLATOR_NOTE = "translator_note"
    PARATEXT = "paratext"
    BELONGS_ELSEWHERE = "belongs_elsewhere"


class Grade(StrEnum):
    """Evidence grade of a cell. Grades are audit strata, not assumed accuracies.

    A  human verdict with a matching fingerprint (gold or final review verdict)
    B  all machine replicates agree on the outcome class and every check passed
    C  majority only, or a non-fatal verification flag
    X  fatal verification failure or no majority (the cell is UNALIGNED)
    """

    A = "A"
    B = "B"
    C = "C"
    X = "X"


class OutcomeClass(StrEnum):
    """Coarse outcome used for replicate voting, sampling strata and estimation."""

    NONDEV = "nondev"             # retained without deviation
    DEV_PRESENT = "dev_present"   # present but rewritten (substitution, reversal, ...)
    PARTIAL = "partial"
    ABSENT = "absent"


# Prefixes of UNALIGNED reasons. A refusal reason is "refused:<category>".
REASON_UNASSESSED = "unassessed"
REASON_REFUSED = "refused"
REASON_TRUNCATED = "truncated"
REASON_INVALID = "invalid"
REASON_SUBSTITUTED_MODEL = "substituted_model"
REASON_VERIFICATION_FAILED = "verification_failed"
REASON_NO_MAJORITY = "no_majority"
REASON_UNMAPPED_CHAPTER = "unmapped_chapter"


def failure_reason(status: str, refusal_category: str | None = None) -> str:
    """The UNALIGNED reason of an unusable LLM answer with ``status`` (refusal, truncated,
    anything else -> invalid): ``refused:<category>`` or ``refused`` without a category."""
    if status == "refusal":
        return f"{REASON_REFUSED}:{refusal_category}" if refusal_category else REASON_REFUSED
    return REASON_TRUNCATED if status == "truncated" else REASON_INVALID


@dataclass(frozen=True)
class Segment:
    """One alignable (or deliberately non-alignable) stretch of a witness.

    ``kind`` is one of: prose, verse_line, verse, mantra, head, meta, colophon,
    paratext, note. ``note`` segments are translator notes kept as evidence;
    ``paratext`` is material outside the work proper (e.g. translators' colophons).
    ``fingerprint`` is ``core.textnorm.fingerprint(text, lang)``; ingesters fill it,
    and human verdicts are only re-applied while it still matches.
    """

    id: str
    witness: str
    lang: str
    text: str
    start: str
    end: str
    kind: str
    local_chapter: str | None = None   # witness-local chapter key, e.g. "pin11", "D418:9"
    chapter: str | None = None         # reference chapter key once mapped, e.g. "II.9"
    fingerprint: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Quote:
    """A verbatim quotation supporting a link; ``side`` is "ref" or "wit"."""

    side: str
    text: str
    segment_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Link:
    """The counterpart of one reference unit in one witness, or witness-only material.

    ``ref_id is None``  witness-only material; ``relation`` is a ``WitnessOnlyKind``.
    ``wit_ids == ()``   no counterpart; ``relation`` is ``Relation.NO_COUNTERPART``.
    ``source`` names the producer, e.g. "claude-opus-5-5:collate.v1:r2",
    "dp:anchor", "dp:zero", "placebo:shuffled", "gold:test".
    """

    ref_id: str | None
    wit_ids: tuple[str, ...]
    relation: Relation | WitnessOnlyKind
    polarity_flip: bool = False
    confidence: str | None = None
    quotes: tuple[Quote, ...] = ()
    flags: frozenset[str] = frozenset()
    source: str = ""


@dataclass(frozen=True)
class Alignment:
    """All links produced by one aligner for one (reference, witness) pair."""

    source: str
    reference: str
    witness: str
    links: tuple[Link, ...] = ()

    def by_ref(self) -> dict[str, Link]:
        """Reference unit id -> its link (the first one if a producer repeated a unit)."""
        out: dict[str, Link] = {}
        for link in self.links:
            if link.ref_id is not None and link.ref_id not in out:
                out[link.ref_id] = link
        return out

    def witness_only(self) -> tuple[Link, ...]:
        return tuple(link for link in self.links if link.ref_id is None)

    def pairs(self) -> frozenset[tuple[str, str]]:
        """(reference unit id, witness segment id) pairs, the unit of link P/R/F1."""
        return frozenset(
            (link.ref_id, wid) for link in self.links if link.ref_id is not None for wid in link.wit_ids
        )


@dataclass(frozen=True)
class Diagnostic:
    """A non-fatal finding raised while verifying or building (e.g. "relocation")."""

    kind: str
    ref_ids: tuple[str, ...] = ()
    wit_ids: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class Cell:
    """One (reference unit, witness) cell of the matrix."""

    unit_id: str
    witness: str
    status: Status
    grade: Grade
    chapter: str = ""
    relation: str | None = None
    polarity_flip: bool = False
    reason: str | None = None          # set when status is UNALIGNED, e.g. "refused:bio"
    wit_ids: tuple[str, ...] = ()
    flags: frozenset[str] = frozenset()
    d_len: float | None = None
    d_ord: float | None = None
    d_lit: float | None = None
    source: str = ""                   # "gold:test", "verdict:<batch>", "claude:consensus", ...


@dataclass(frozen=True)
class Verdict:
    """A human decision about one reference unit, from a review or gold sheet.

    ``task`` is one of: gold, verify, audit, resolve. The blind columns are filled
    before the annotator sees any machine output; the final columns after reveal.
    Only blind gold verdicts may be used to estimate machine error rates.
    """

    batch_id: str
    item_id: str
    task: str
    unit_id: str
    fingerprint: str
    stratum: str = ""
    inclusion_prob: float | None = None
    instrument_digest: str = ""
    machine_relation: str = ""
    machine_status: str = ""
    machine_polarity_flip: bool = False
    blind_relation: str = ""
    blind_wit_ids: tuple[str, ...] = ()
    final_relation: str = ""
    final_wit_ids: tuple[str, ...] = ()
    polarity_flip: bool = False
    flags: frozenset[str] = frozenset()
    annotator: str = ""
    blind_date: str = ""
    final_date: str = ""
    note: str = ""
    quote_ref: str = ""               # short reference-side quote (licence-capped)
    quote_zh: str = ""                # short witness-side quote (licence-capped)
    minutes: float | None = None      # annotator time, to measure the real review cost


@dataclass(frozen=True)
class Estimate:
    """A reported number together with what it estimates and what its interval covers.

    When ``not_estimable`` is set the numeric fields are None and the report prints
    ``NOT_ESTIMABLE: <reason>`` instead of a number.
    """

    name: str
    point: float | None
    lo: float | None
    hi: float | None
    n: int
    scope: str
    sources: tuple[str, ...] = ()
    level: int = 0
    not_estimable: str | None = None

    @classmethod
    def missing(cls, name: str, reason: str, scope: str = "") -> "Estimate":
        return cls(name=name, point=None, lo=None, hi=None, n=0, scope=scope, not_estimable=reason)


@dataclass(frozen=True)
class InstrumentId:
    """Identity of a measurement instrument; any change means re-validation."""

    model: str
    task: str
    effort: str
    prompt_sha: str
    schema_sha: str
    code_version: str
    params_sha: str
    replicates: int

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
