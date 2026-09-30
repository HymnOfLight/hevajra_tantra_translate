"""Classes of CBETA inline notes and the source of Taisho footnotes.

The rules are data (``data/lexicon/notes_zh.yaml``): an ordered list of
``(class, regular expression)`` pairs, each matched against the whole note text with
whitespace removed; the first match wins. A note that no rule matches is
``UNCLASSIFIED``, which the G0 ingest gate rejects, so a new kind of note is noticed
instead of being filed under the nearest class.

The seven classes separate evidence that the method treats differently: a
``vorlage_statement`` is evidence about the translator's Sanskrit manuscript, a
``substitution_instruction`` is evidence about the translator's own choices, and the
rest (pronunciation marks, mantra numbering, fanqie spellings, the title and fascicle
notes) describe the Chinese text itself.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..core.lexicon import LexiconError, read_lexicon

NOTE_CLASSES = (
    "phonetic",
    "mantra_numbering",
    "fanqie",
    "substitution_instruction",
    "fascicle",
    "title",
    "vorlage_statement",
)
UNCLASSIFIED = "unclassified"

FOOTNOTE_TOKYO335 = "tokyo335"   # reading of Tokyo Imperial University Sanskrit ms. no. 335
FOOTNOTE_TAISHO = "taisho"       # Sanskrit equivalent supplied by the Taisho editors


@dataclass(frozen=True)
class NoteRule:
    note_class: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class NoteRules:
    """Ordered note-class rules and the manuscript marker of Taisho footnotes."""

    rules: tuple[NoteRule, ...]
    tokyo335_marker: str


def load_rules(data_dir: Path) -> NoteRules:
    """Read ``data/lexicon/notes_zh.yaml``; unknown keys, classes or bad patterns raise."""
    doc = read_lexicon(data_dir, "notes_zh")
    unknown = sorted(set(doc) - {"rules", "tokyo335_marker"})
    if unknown:
        raise LexiconError(f"notes_zh: unknown key(s) {unknown}")
    rules = []
    for rec in doc.get("rules", ()):
        if set(rec) != {"class", "pattern"} or rec["class"] not in NOTE_CLASSES:
            raise LexiconError(f"notes_zh.rules: expected {{class, pattern}} with a class in "
                               f"{NOTE_CLASSES}, got {dict(rec)}")
        try:
            rules.append(NoteRule(rec["class"], re.compile(rec["pattern"])))
        except re.error as exc:
            raise LexiconError(f"notes_zh.rules: bad pattern {rec['pattern']!r}: {exc}") from exc
    marker = doc.get("tokyo335_marker")
    if not rules or not isinstance(marker, str) or not marker:
        raise LexiconError("notes_zh: needs a non-empty rules list and tokyo335_marker")
    return NoteRules(rules=tuple(rules), tokyo335_marker=marker)


def normalise_note(text: str) -> str:
    """NFC with all whitespace removed (CBETA breaks long notes across lines)."""
    return "".join(unicodedata.normalize("NFC", text).split())


def classify(text: str, rules: NoteRules) -> str:
    """Class of an inline note: the first rule whose pattern matches the whole text."""
    key = normalise_note(text)
    for rule in rules.rules:
        if rule.pattern.fullmatch(key):
            return rule.note_class
    return UNCLASSIFIED


def footnote_source(text: str, rules: NoteRules) -> str:
    """``FOOTNOTE_TOKYO335`` when a Taisho footnote carries the manuscript marker."""
    marked = rules.tokyo335_marker in unicodedata.normalize("NFC", text)
    return FOOTNOTE_TOKYO335 if marked else FOOTNOTE_TAISHO
