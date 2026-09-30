"""The topic vocabulary (fixed in code) and the codebook file that defines it (data).

``data/codebook/topics.yaml`` is the single codebook for the pre-labeller prompt, the human
labels, the exposure of E4 and the experiment (fixes review defect #20). Its topic names,
groups and group precedence must equal the constants below, which are pre-registered;
``load_codebook`` refuses a file that disagrees, so the two can never drift apart.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..core.io import read_yaml

# Tokushige 2026's eight categories of self-censored verses: the "sensitive" group.
SENSITIVE_TOPICS: tuple[str, ...] = (
    "sexual", "female_agent", "flesh_food", "harm", "theft", "impure_substance", "bone_corpse", "ritual",
)
NEUTRAL, FRAME, MANTRA_CONTROL = "neutral", "frame", "mantra_control"
TOPICS: tuple[str, ...] = SENSITIVE_TOPICS + (NEUTRAL, FRAME, MANTRA_CONTROL)

TopicGroup = Literal["sensitive", "mantra_control", "frame", "neutral", "unlabelled"]
UNLABELLED: TopicGroup = "unlabelled"
GROUP_OF: Mapping[str, str] = MappingProxyType(
    {**{t: "sensitive" for t in SENSITIVE_TOPICS}, NEUTRAL: "neutral", FRAME: "frame", MANTRA_CONTROL: "mantra_control"}
)
# A unit's group is the first of these that holds one of its topics. Sensitive comes first
# because the exposure of E4 is "any of the eight categories" (synthesis 6.3).
GROUP_PRECEDENCE: tuple[str, ...] = ("sensitive", "mantra_control", "frame", "neutral")


class TopicError(ValueError):
    """An invalid topic set or an invalid topic-label file."""


class CodebookError(TopicError):
    """The codebook file is malformed or disagrees with the vocabulary fixed in code."""


def topic_set_error(topics: frozenset[str], allowed: Sequence[str] = TOPICS) -> str | None:
    """Why ``topics`` is not a valid label set for one unit, or None if it is.

    Valid: only known topics, and ``neutral`` never together with another topic. The empty
    set is valid (the unit is not labelled yet).
    """
    unknown = sorted(topics - set(allowed))
    if unknown:
        hint = " (separate topics with ';')" if any(re.search(r"[\s,]", u) for u in unknown) else ""
        return f"unknown topic(s) {unknown}{hint}; allowed: {', '.join(allowed)}"
    if NEUTRAL in topics and len(topics) > 1:
        return f"neutral must appear alone, not with {sorted(topics - {NEUTRAL})}"
    return None


def topic_group(topics: Iterable[str]) -> TopicGroup:
    """Group of one unit's topics under ``GROUP_PRECEDENCE``; no topics -> "unlabelled".

    Raises ``TopicError`` for an unknown topic or for neutral combined with another topic.
    """
    topics = frozenset(topics)
    problem = topic_set_error(topics)
    if problem:
        raise TopicError(problem)
    if not topics:
        return UNLABELLED
    groups = {GROUP_OF[t] for t in topics}
    return next(g for g in GROUP_PRECEDENCE if g in groups)  # type: ignore[return-value]


@dataclass(frozen=True)
class TopicDefinition:
    name: str
    group: str
    definition: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    tokushige_2026_verses: int | None = None


@dataclass(frozen=True)
class TopicCodebook:
    """Topic definitions (in the order shown to coders and to the model) and coding rules."""

    topics: tuple[TopicDefinition, ...]
    rules: tuple[str, ...]
    status: str                      # draft | verified (by the researcher named in verified_by)
    verified_by: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.topics)


_CODEBOOK_KEYS = frozenset({"schema_version", "status", "verified_by", "source", "group_precedence", "rules", "topics"})
_TOPIC_KEYS = frozenset({"name", "group", "tokushige_2026_verses", "definition", "include", "exclude"})


def load_codebook(path: Path) -> TopicCodebook:
    """Read and validate ``data/codebook/topics.yaml``; every problem is reported at once."""
    doc = read_yaml(path)
    if not isinstance(doc, Mapping):
        raise CodebookError(f"{path}: the top level must be a mapping")
    errors = [f"unknown key {k!r}" for k in sorted(set(doc) - _CODEBOOK_KEYS)]
    errors += [f"missing key {k!r}" for k in sorted(_CODEBOOK_KEYS - set(doc))]
    if doc.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if doc.get("status") not in ("draft", "verified"):
        errors.append("status must be draft or verified")
    elif doc["status"] == "verified" and not doc.get("verified_by"):
        errors.append("a verified codebook needs verified_by")
    if tuple(doc.get("group_precedence") or ()) != GROUP_PRECEDENCE:
        errors.append(f"group_precedence must be {list(GROUP_PRECEDENCE)} (fixed in hevajra_matrix.topics)")
    if not _texts(doc.get("rules")):
        errors.append("rules must be a non-empty list of non-empty strings")
    records = doc.get("topics")
    if not isinstance(records, list):
        errors.append("topics must be a list")
        records = []
    topics = [_topic(i, record, errors) for i, record in enumerate(records)]
    names = [t.name for t in topics]
    duplicated = sorted(n for n, c in Counter(names).items() if c > 1)
    if duplicated or set(names) != set(TOPICS):
        errors.append(
            f"topics must be exactly {list(TOPICS)} (fixed in hevajra_matrix.topics); "
            f"missing {sorted(set(TOPICS) - set(names))}, extra {sorted(set(names) - set(TOPICS))}, "
            f"duplicated {duplicated}"
        )
    if errors:
        raise CodebookError(f"{path}:\n  " + "\n  ".join(errors))
    return TopicCodebook(topics=tuple(topics), rules=tuple(r.strip() for r in doc["rules"]),
                         status=doc["status"], verified_by=doc.get("verified_by"))


def _topic(index: int, record: Any, errors: list[str]) -> TopicDefinition:
    if not isinstance(record, Mapping):
        errors.append(f"topics[{index}] must be a mapping")
        return TopicDefinition("", "", "", (), ())
    name = str(record.get("name", ""))
    where = f"topics[{index}] ({name})"
    errors += [f"{where}: unknown key {k!r}" for k in sorted(set(record) - _TOPIC_KEYS)]
    if name in GROUP_OF and record.get("group") != GROUP_OF[name]:
        errors.append(f"{where}: group must be {GROUP_OF[name]!r} (fixed in hevajra_matrix.topics)")
    if not (isinstance(record.get("definition"), str) and record["definition"].strip()):
        errors.append(f"{where}: definition must be a non-empty string")
    for key in ("include", "exclude"):
        if not _texts(record.get(key)):
            errors.append(f"{where}: {key} must be a non-empty list of non-empty strings")
    verses = record.get("tokushige_2026_verses")
    if verses is not None and not (isinstance(verses, int) and not isinstance(verses, bool) and verses >= 0):
        errors.append(f"{where}: tokushige_2026_verses must be a count or null")
    return TopicDefinition(
        name=name,
        group=str(record.get("group", "")),
        definition=str(record.get("definition", "")).strip(),
        include=tuple(str(x).strip() for x in record.get("include") or ()),
        exclude=tuple(str(x).strip() for x in record.get("exclude") or ()),
        tokushige_2026_verses=verses,
    )


def _texts(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) and v.strip() for v in value)
