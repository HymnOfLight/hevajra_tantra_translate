"""Human topic labels (``data/annotations/topics/<reference>.csv``) and their agreement.

File format (see ``data/annotations/topics/README.md``): exactly ``LABEL_COLUMNS``, one row
per reference unit, topic lists separated by ";". Labels are keyed by (unit id,
fingerprint): ``current_labels`` sets aside labels whose unit text has changed, so they
are listed for relabelling instead of being silently applied.

Agreement (critique A6)
    ``human_agreement`` compares the first coder with the blind second coder and is the
    only source of kappa_topic. ``prelabel_agreement`` compares the committed prelabel
    with one human column and is reported separately; it never enters kappa_topic.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Iterable, Literal, Mapping, Sequence

from ..core.io import write_csv
from ..core.types import Segment
from .codebook import TOPICS, TopicCodebook, TopicError, TopicGroup, topic_group, topic_set_error

LABEL_COLUMNS: tuple[str, ...] = (
    "unit_id", "fingerprint", "topics", "prelabel_topics", "coder", "date",
    "second_topics", "second_coder", "second_date",
)
TOPIC_SEPARATOR = ";"
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class TopicLabel:
    """One row of the committed topic-label file.

    ``topics``           the first coder's confirmed label (empty: not labelled yet)
    ``prelabel_topics``  the verified T3 prelabel that coder saw (empty: none usable)
    ``second_topics``    the second coder's label, made blind to prelabels (empty: not
                         double-coded)
    """

    unit_id: str
    fingerprint: str
    topics: frozenset[str] = frozenset()
    prelabel_topics: frozenset[str] = frozenset()
    coder: str = ""
    date: str = ""
    second_topics: frozenset[str] = frozenset()
    second_coder: str = ""
    second_date: str = ""

    @property
    def group(self) -> TopicGroup:
        return topic_group(self.topics)


def parse_topics(cell: str, codebook: TopicCodebook) -> frozenset[str]:
    """``"sexual;female_agent"`` -> frozenset; empty pieces are ignored, invalid sets rejected."""
    topics = frozenset(piece.strip() for piece in cell.split(TOPIC_SEPARATOR) if piece.strip())
    problem = topic_set_error(topics, codebook.names)
    if problem:
        raise TopicError(problem)
    return topics


def format_topics(topics: Iterable[str]) -> str:
    """The inverse of ``parse_topics``, in vocabulary order; invalid sets are rejected."""
    topics = frozenset(topics)
    problem = topic_set_error(topics)
    if problem:
        raise TopicError(problem)
    return TOPIC_SEPARATOR.join(t for t in TOPICS if t in topics)


def load_labels(path: Path, codebook: TopicCodebook) -> dict[str, TopicLabel]:
    """Read a committed topic-label file; every invalid row is reported at once.

    The columns must be exactly ``LABEL_COLUMNS`` (in any order). A text column is thus
    refused: the committed file holds ids and labels only (licence rule).
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or ())
        rows = list(reader)
    if sorted(header) != sorted(LABEL_COLUMNS):
        raise TopicError(f"{path}: columns must be exactly {', '.join(LABEL_COLUMNS)}; found {', '.join(header)}")
    labels: dict[str, TopicLabel] = {}
    errors: list[str] = []
    for line, row in enumerate(rows, start=2):
        try:
            label = _label(row, codebook)
        except TopicError as exc:
            errors.append(f"line {line}: {exc}")
            continue
        if label.unit_id in labels:
            errors.append(f"line {line}: duplicate unit_id {label.unit_id}")
            continue
        labels[label.unit_id] = label
    if errors:
        raise TopicError(f"{path}: {len(errors)} invalid row(s):\n  " + "\n  ".join(errors))
    return labels


def _label(row: Mapping[Any, Any], codebook: TopicCodebook) -> TopicLabel:
    if None in row or any(v is None for v in row.values()):
        raise TopicError("the row does not have exactly one value per column")
    v = {k: row[k].strip() for k in LABEL_COLUMNS}
    problems: list[str] = []

    def topics_in(column: str) -> frozenset[str]:
        try:
            return parse_topics(v[column], codebook)
        except TopicError as exc:
            problems.append(f"{column}: {exc}")
            return frozenset()

    topics, prelabel, second = topics_in("topics"), topics_in("prelabel_topics"), topics_in("second_topics")
    problems += [f"{k} is empty" for k in ("unit_id", "fingerprint") if not v[k]]
    for given, coder, date in ((topics, "coder", "date"), (second, "second_coder", "second_date")):
        if given and not v[coder]:
            problems.append(f"{coder} is empty")
        if v[date] and not _ISO_DATE.fullmatch(v[date]):
            problems.append(f"{date} {v[date]!r} is not YYYY-MM-DD")
    if v["second_coder"] and v["second_coder"].casefold() == v["coder"].casefold():
        problems.append("second_coder must be a different person from coder")
    if problems:
        raise TopicError("; ".join(problems))
    return TopicLabel(unit_id=v["unit_id"], fingerprint=v["fingerprint"], topics=topics,
                      prelabel_topics=prelabel, coder=v["coder"], date=v["date"], second_topics=second,
                      second_coder=v["second_coder"], second_date=v["second_date"])


def write_labels(path: Path, labels: Iterable[TopicLabel]) -> None:
    """Write labels in the committed format, rows in the given order (no BOM)."""
    rows = [{"unit_id": x.unit_id, "fingerprint": x.fingerprint, "topics": format_topics(x.topics),
             "prelabel_topics": format_topics(x.prelabel_topics), "coder": x.coder, "date": x.date,
             "second_topics": format_topics(x.second_topics), "second_coder": x.second_coder,
             "second_date": x.second_date} for x in labels]
    write_csv(path, rows, LABEL_COLUMNS)


def current_labels(
    labels: Mapping[str, TopicLabel], reference: Iterable[Segment]
) -> tuple[dict[str, TopicLabel], tuple[str, ...]]:
    """(labels whose unit exists with the same fingerprint, ids of the stale rest).

    A stale label belongs to a unit whose text changed or disappeared on re-ingest; it is
    listed for relabelling and never used as an exposure.
    """
    fingerprints = {s.id: s.fingerprint for s in reference}
    current = {u: x for u, x in labels.items() if fingerprints.get(u) == x.fingerprint}
    return current, tuple(u for u in labels if u not in current)


# --------------------------------------------------------------------------- agreement
@dataclass(frozen=True)
class TopicKappa:
    """Agreement of two label columns on one topic (present or absent in each unit).

    ``positive_agreement`` = 2a / (2a + b + c) is printed beside kappa because kappa is
    unstable for rare topics (critique A7). Either is None when undefined.
    """

    kappa: float | None
    positive_agreement: float | None
    both: int              # a: both columns have the topic
    first_only: int        # b
    second_only: int       # c


@dataclass(frozen=True)
class TopicAgreement:
    """Agreement over the units labelled in both columns.

    ``group_kappa`` is the overall figure: Cohen's kappa on ``topic_group``, i.e. on the
    exposure of E4, which gate G4 compares with ``min_topic_kappa``. ``group_agreement``
    is the observed share of units placed in the same group.
    """

    n_units: int
    group_kappa: float | None
    group_agreement: float | None
    per_topic: Mapping[str, TopicKappa]


def cohen_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> float | None:
    """Cohen's kappa of two coders' nominal codes, one pair per unit.

    None when there are no pairs or chance agreement is 1 (both coders used one and the
    same code throughout), where kappa is undefined.
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[k] * second[k] for k in first) / (n * n)
    if expected > 1 - 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def agreement(pairs: Sequence[tuple[frozenset[str], frozenset[str]]]) -> TopicAgreement:
    """Per-topic and group agreement of paired label sets (one pair per unit)."""
    groups = [(topic_group(a), topic_group(b)) for a, b in pairs]
    per_topic = {}
    for topic in TOPICS:
        flags = [(topic in a, topic in b) for a, b in pairs]
        both, first, second = (flags.count(case) for case in ((True, True), (True, False), (False, True)))
        denominator = 2 * both + first + second
        per_topic[topic] = TopicKappa(cohen_kappa(flags), 2 * both / denominator if denominator else None,
                                      both, first, second)
    observed = sum(a == b for a, b in groups) / len(groups) if groups else None
    return TopicAgreement(len(pairs), cohen_kappa(groups), observed, MappingProxyType(per_topic))


def human_agreement(labels: Mapping[str, TopicLabel]) -> TopicAgreement:
    """kappa_topic: the first coder against the blind second coder, on double-coded units.

    Human columns only (critique A6): prelabels never enter this statistic.
    """
    return agreement([(x.topics, x.second_topics) for x in labels.values() if x.topics and x.second_topics])


def prelabel_agreement(
    labels: Mapping[str, TopicLabel], against: Literal["second", "first"] = "second"
) -> TopicAgreement:
    """Agreement of the committed prelabel with one human column, reported apart from kappa_topic.

    ``second`` (default): the blind coder, who never saw a prelabel, so this estimates the
    pre-labeller's accuracy on the double-coded units. ``first``: the coder who confirmed
    the prelabel; anchored, so it measures how often prelabels were kept, not accuracy.
    Units without a usable prelabel are left out (see ``Prelabel.committed_topics``).
    """
    if against not in ("second", "first"):
        raise ValueError(f"against must be 'second' or 'first', not {against!r}")
    column = "second_topics" if against == "second" else "topics"
    return agreement([(x.prelabel_topics, getattr(x, column)) for x in labels.values()
                      if x.prelabel_topics and getattr(x, column)])
