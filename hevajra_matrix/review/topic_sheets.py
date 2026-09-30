"""Topic confirmation sheets (synthesis 7.3 C; critique A6).

    topics_<batch>.csv          first coder: the reference unit, two units of context on each
                                side, and the T3 prelabel with its verified cue as a hint
    topics_<batch>.second.csv   second coder: the same WITHOUT any prelabel column, so topic
                                kappa is human-human only

Both sheets show reference text only (the coders stay blind to the witness): a unit's
context is taken from segments of the unit's own witness. Import returns
``topics.TopicLabel`` rows for ``topics.write_labels``; ``merge_topic_labels`` folds a
first- or second-coder import into the committed labels. Written through
``review.sheets.export(..., task="topics" | "topics_second")``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..core.io import write_csv
from ..core.types import Segment
from ..matrix.build import segment_fingerprint
from ..topics import TopicCodebook, TopicLabel, format_topics, parse_topics
from ..topics.prelabel import Prelabel
from .sampling import ReviewItem
from .sheets import expect_columns, one_line, read_sheet
from .verdicts import VerdictError

TOPIC_COLUMNS = ("unit_id", "fingerprint", "locus", "text", "context_before", "context_after",
                 "prelabel_topics", "prelabel_cue", "topics", "note")
TOPIC_SECOND_COLUMNS = tuple(c for c in TOPIC_COLUMNS if not c.startswith("prelabel_"))
TOPIC_CONTEXT_UNITS = 2
SECOND_SUFFIX = ".second.csv"


def export_topics(items: Sequence[ReviewItem], task: str, out: Path, batch: str, segments: Sequence[Segment],
                  prelabels: Mapping[str, Prelabel]) -> Path:
    """Write one topics sheet (``task`` "topics" or "topics_second"); returns its path."""
    second = task == "topics_second"
    index = {s.id: s for s in segments}
    rows = []
    for item in items:
        seg = index[item.unit_id]
        same = [s for s in segments if s.witness == seg.witness]      # reference text only
        i = next(n for n, s in enumerate(same) if s.id == seg.id)
        row = {"unit_id": seg.id, "fingerprint": segment_fingerprint(seg), "locus": seg.start,
               "text": one_line(seg.text),
               "context_before": " / ".join(one_line(s.text) for s in same[max(i - TOPIC_CONTEXT_UNITS, 0):i]),
               "context_after": " / ".join(one_line(s.text) for s in same[i + 1:i + 1 + TOPIC_CONTEXT_UNITS])}
        if not second:
            p = prelabels.get(seg.id)
            row["prelabel_topics"] = format_topics(p.topics) if p else ""
            row["prelabel_cue"] = " | ".join(f"{t}: {cue}" for t, cue in p.cues) if p else ""
        rows.append(row)
    path = Path(out) / (f"topics_{batch}{SECOND_SUFFIX}" if second else f"topics_{batch}.csv")
    write_csv(path, rows, TOPIC_SECOND_COLUMNS if second else TOPIC_COLUMNS, bom=True)
    return path


def import_topics(path: Path, codebook: TopicCodebook, *, coder: str, date: str,
                  prelabels: Mapping[str, Prelabel] | None = None) -> list[TopicLabel]:
    """Read a filled topics sheet; rows without ``topics`` are skipped.

    A first-coder sheet fills ``topics``, ``coder``, ``date`` and ``prelabel_topics`` (the
    committed prelabel, ``Prelabel.committed_topics``, never the hint shown on the sheet). A
    second-coder sheet (``*.second.csv``) fills only the ``second_*`` fields. Every invalid
    row is reported at once.
    """
    path = Path(path)
    second = path.name.endswith(SECOND_SUFFIX)
    rows, header = read_sheet(path)
    expect_columns(header, TOPIC_SECOND_COLUMNS if second else TOPIC_COLUMNS, path)
    out, errors = [], []
    for n, row in enumerate(rows, start=2):
        if not row["topics"].strip():
            continue
        try:
            topics = parse_topics(row["topics"], codebook)
        except ValueError as exc:
            errors.append(f"line {n}: {exc}")
            continue
        uid, fp = row["unit_id"].strip(), row["fingerprint"].strip()
        if second:
            out.append(TopicLabel(uid, fp, second_topics=topics, second_coder=coder, second_date=date))
        else:
            pre = (prelabels or {}).get(uid)
            out.append(TopicLabel(uid, fp, topics=topics, prelabel_topics=pre.committed_topics if pre else frozenset(),
                                  coder=coder, date=date))
    if errors:
        raise VerdictError(f"{path}: {len(errors)} problem(s):\n  " + "\n  ".join(errors))
    return out


def merge_topic_labels(existing: Mapping[str, TopicLabel], imported: Iterable[TopicLabel]) -> dict[str, TopicLabel]:
    """Fold imported first- or second-coder labels into the committed labels.

    A first-coder label replaces the unit's first-coder fields (and the whole row when the
    unit's text changed); a second-coder label needs a first label on the same text.
    """
    out = dict(existing)
    for new in imported:
        old = out.get(new.unit_id)
        if new.second_coder:
            if old is None or old.fingerprint != new.fingerprint:
                raise VerdictError(f"{new.unit_id}: second-coder label without a first label on the same text")
            out[new.unit_id] = replace(old, second_topics=new.second_topics, second_coder=new.second_coder,
                                       second_date=new.second_date)
        elif old is not None and old.fingerprint == new.fingerprint:
            out[new.unit_id] = replace(old, topics=new.topics, prelabel_topics=new.prelabel_topics,
                                       coder=new.coder, date=new.date)
        else:
            out[new.unit_id] = new
    return out
