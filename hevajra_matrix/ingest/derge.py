"""Ingest the Esukhia Derge Kangyur volume text and slice the requested Toh texts.

File format (Esukhia/derge-kangyur, ``text/*.txt``)::

    [1b]                folio marker (no text)
    [1b.1]{D417}...     line marker "folio.line"; ``{Dnnn}`` starts the text Toh nnn
    ...{a,b}...         Esukhia edit mark. In all 26 marks of Toh 417-418, a is a
                        non-standard spelling of the block print and b its standard form

The file is NFC-normalised (as the marker patterns are) and read as one character stream
in which every character knows its ``folio.line``, so a phrase broken across lines is
matched as one string. The text of Toh nnn runs from its ``{Dnnn}`` to the next
text-start mark. Each ``{a,b}`` is replaced by b and logged as a ``Variant`` (kind
"orthographic_variant"); any other brace is an error, so no markup can leak into a
segment.

Units are cut after every shad-type character; a unit without any syllable (e.g. a bare
double shad or head mark) is dropped. Kinds, in order of precedence:

    paratext    after the text-end colophon of the Toh (the translators' colophons)
    meta        front matter: title block and homage before the text proper
    colophon    a chapter colophon, or the text-end colophon (the last unit containing
                the text-end marker); it closes the current chapter
    mantra      at least 3 syllables, and either more than half of them with
                Sanskrit-only letters, or a mantra cue (an opening om, a closing svaha
                or phat; ``mantra_cues`` in the marker file) and at least
                ``mantra_cues.min_ratio`` of them with Sanskrit-only letters
    verse_line  7, 9 or 11 syllables (the classical metres)
    prose       everything else

``local_chapter`` is "<Toh>:<n>", where n counts colophons (the colophon's own ordinal is
parsed and every disagreement is reported); meta and paratext have none. The
translators' colophons are parsed into ``metadata``: translators, reviser and the
verbatim revision statement. All Tibetan marker strings are in
``data/lexicon/derge_markers.yaml``.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.ids import assign_ids
from ..core.lexicon import LexiconError, read_lexicon
from ..core.textnorm import bo_sanskrit_syllable_ratio, bo_syllables, chunks, fingerprint, for_quote
from ..core.types import Segment
from . import CONTENT_KINDS, IngestResult, Variant, duplicate_ids, kind_counts

LANG = "bo"
REVISER_ROLE = "reviser"
VERSE_SYLLABLES = frozenset({7, 9, 11})
MANTRA_MIN_SYLLABLES = 3
MANTRA_MIN_RATIO = 0.5          # share of syllables with Sanskrit-only letters (exclusive)
# Why exclusive: at exactly 0.5 the real D417/418 text has name lists and transliterated
# Apabhramsa song lines (e.g. "Mamaki and", D418:21b.7.3), not mantras. The mantras at 0.5
# (D417:3b.2.10, D417:4a.7.2) open with om and are caught by the cue rule instead.

# Shad (U+0F0D), nyis shad (U+0F0E), tsheg shad (U+0F0F), nyis tsheg shad (U+0F10),
# rin chen spungs shad (U+0F11) and gter tsheg (U+0F14): a unit ends after any of them.
_SHADS = frozenset("\u0f0d\u0f0e\u0f0f\u0f10\u0f11\u0f14")
_LINE_RE = re.compile(r"\[([0-9]+[ab])(?:\.([0-9]+))?\]")
_MARK_RE = re.compile(r"\{(D[0-9]+[a-z]?)\}|\{([^{},]*),([^{},]*)\}|[{}]")


@dataclass(frozen=True)
class Markers:
    """The compiled contents of ``derge_markers.yaml`` (see the comments there)."""

    text_end: re.Pattern[str]
    chapter_colophon: re.Pattern[str]
    ordinals: Mapping[str, int]
    front_start: re.Pattern[str]
    front_end: re.Pattern[str]
    front_start_within: int
    front_max_units: int
    colophon_roles: tuple[tuple[str, re.Pattern[str]], ...]
    mantra_initial: tuple[tuple[str, ...], ...] = ()     # syllable sequences opening a mantra
    mantra_final: tuple[tuple[str, ...], ...] = ()       # syllable sequences closing a mantra
    mantra_cue_min_ratio: float = 1.0                    # Sanskrit ratio needed with a cue (inclusive)


@dataclass
class _Stream:
    chars: list[tuple[str, str]] = field(default_factory=list)   # (character, "folio.line")
    starts: dict[str, int] = field(default_factory=dict)         # Toh -> first position
    marks: list[tuple[int, str, str, str]] = field(default_factory=list)  # (pos, a, b, line)


def load_markers(data_dir: Path) -> Markers:
    doc = read_lexicon(data_dir, "derge_markers")
    expected = {"text_end", "chapter_colophon", "ordinals", "front_matter", "translator_colophon", "mantra_cues"}
    if set(doc) != expected:
        raise LexiconError(f"derge_markers: expected keys {sorted(expected)}, got {sorted(doc)}")
    front = doc["front_matter"]
    cues = doc["mantra_cues"]
    try:
        markers = Markers(
            text_end=re.compile(doc["text_end"]),
            chapter_colophon=re.compile(doc["chapter_colophon"]),
            ordinals={rec["form"]: int(rec["value"]) for rec in doc["ordinals"]},
            front_start=re.compile(front["start"]),
            front_end=re.compile(front["end"]),
            front_start_within=int(front["start_within_units"]),
            front_max_units=int(front["max_units"]),
            colophon_roles=tuple((r["role"], re.compile(r["pattern"])) for r in doc["translator_colophon"]),
            mantra_initial=tuple(_cue(c) for c in cues["initial"]),
            mantra_final=tuple(_cue(c) for c in cues["final"]),
            mantra_cue_min_ratio=float(cues["min_ratio"]),
        )
    except (KeyError, TypeError, ValueError, re.error) as exc:
        raise LexiconError(f"derge_markers: malformed entry ({exc})") from exc
    if not 0.0 < markers.mantra_cue_min_ratio <= 1.0:
        raise LexiconError("derge_markers.mantra_cues.min_ratio must be in (0, 1]")
    if "ordinal" not in markers.chapter_colophon.groupindex:
        raise LexiconError("derge_markers.chapter_colophon needs a group named 'ordinal'")
    if any("name" not in p.groupindex for _, p in markers.colophon_roles):
        raise LexiconError("derge_markers.translator_colophon: every pattern needs a group named 'name'")
    return markers


def parse(path: Path, toh: Sequence[str], witness: str, data_dir: Path) -> IngestResult:
    """Segments of the requested Toh texts (e.g. ``["D417", "D418"]``) of one volume file."""
    if not toh or len(set(toh)) != len(toh):
        raise ValueError(f"toh must list distinct text numbers, got {list(toh)!r}")
    markers = load_markers(data_dir)
    text = unicodedata.normalize("NFC", Path(path).read_text(encoding="utf-8-sig"))
    stream = _read(text)
    missing = [t for t in toh if t not in stream.starts]
    if missing:
        raise ValueError(f"{path}: no text start for {missing}; found {sorted(stream.starts)}")
    boundaries = sorted(stream.starts.values()) + [len(stream.chars)]
    segments: list[Segment] = []
    variants: list[Variant] = []
    chapters: list[dict[str, Any]] = []
    for number in toh:
        begin = stream.starts[number]
        end = boundaries[bisect_right(boundaries, begin)]
        units = _units(stream.chars, begin, end)
        segs, chaps = _segments(units, stream.chars, number, witness, markers)
        segments.extend(segs)
        chapters.extend(chaps)
        variants.extend(_variants(stream.marks, begin, end, units, segs))
    metadata = {"witness": witness, "toh": list(toh), "chapters": chapters, **_colophon(segments, markers)}
    report = {
        "segments": len(segments),
        "kinds": kind_counts(segments),
        "content_segments": sum(s.kind in CONTENT_KINDS for s in segments),
        "variants": len(variants),
        "variant_pairs": len({(v.alternative, v.reading) for v in variants}),
        "paratext": sum(s.kind == "paratext" for s in segments),
        "chapters": dict(Counter(c["local"].split(":")[0] for c in chapters)),
        "chapter_ordinal_mismatches": [c["local"] for c in chapters
                                       if c["ordinal"] is not None and c["local"] != f"{c['toh']}:{c['ordinal']}"],
        "colophon_roles": sorted(r["role"] for r in metadata["translators"])
                          + ([REVISER_ROLE] if metadata["reviser"] else []),
        "duplicate_ids": duplicate_ids(segments),
    }
    return IngestResult(tuple(segments), (), tuple(variants), metadata, report)


# --------------------------------------------------------------------------- reading
def _read(text: str) -> _Stream:
    stream = _Stream()
    coord = ""

    def add(chunk: str) -> None:
        stream.chars.extend((ch, coord) for ch in chunk if ch not in "\r\n")

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw
        m = _LINE_RE.match(line)
        if m:
            line = line[m.end():]
            if m.group(2) is None:
                if line.strip():
                    raise ValueError(f"line {number}: text after the bare folio marker {m.group(0)}")
                continue
            coord = f"{m.group(1)}.{m.group(2)}"
        pos = 0
        for mark in _MARK_RE.finditer(line):
            add(line[pos:mark.start()])
            if mark.group(1):
                if mark.group(1) in stream.starts:
                    raise ValueError(f"line {number} ({coord}): second start of {mark.group(1)}")
                stream.starts[mark.group(1)] = len(stream.chars)
            elif mark.group(2) is not None:
                stream.marks.append((len(stream.chars), mark.group(2), mark.group(3), coord))
                add(mark.group(3))
            else:
                raise ValueError(f"line {number} ({coord}): unrecognised brace markup in {raw.strip()[:60]!r}")
            pos = mark.end()
        add(line[pos:])
    return stream


def _units(chars: list[tuple[str, str]], begin: int, end: int) -> list[tuple[int, int]]:
    """(start, end) stream positions of the syllable-bearing units in ``[begin, end)``."""
    units = []
    a = begin
    for i in range(begin, end + 1):
        if i == end or chars[i][0] in _SHADS:
            b = min(i + 1, end)
            x, y = a, b
            while x < y and chars[x][0].isspace():
                x += 1
            while y > x and chars[y - 1][0].isspace():
                y -= 1
            if for_quote("".join(ch for ch, _ in chars[x:y]), LANG):
                units.append((x, y))
            a = b
    return units


# --------------------------------------------------------------------------- segments
def _segments(units: list[tuple[int, int]], chars: list[tuple[str, str]], toh: str, witness: str,
              markers: Markers) -> tuple[list[Segment], list[dict[str, Any]]]:
    texts = ["".join(ch for ch, _ in chars[a:b]) for a, b in units]
    ends = [i for i, t in enumerate(texts) if markers.text_end.search(t)]
    text_end = ends[-1] if ends else len(texts)
    front = _front_matter(texts, markers)
    ids = assign_ids(toh, [(chars[a][1], False) for a, _ in units])
    segments: list[Segment] = []
    chapters: list[dict[str, Any]] = []
    chapter = 1
    for i, ((a, b), text) in enumerate(zip(units, texts)):
        local: str | None = f"{toh}:{chapter}"
        colophon = markers.chapter_colophon.search(text)
        if i > text_end:
            kind, local = "paratext", None
        elif i < front:
            kind, local = "meta", None
        elif colophon or i == text_end:
            kind = "colophon"
            ordinal = markers.ordinals.get(colophon.group("ordinal")) if colophon else None
            chapters.append({"local": local, "toh": toh, "ordinal": ordinal, "end": chars[b - 1][1],
                             "segment_id": ids[i], "text_end": i == text_end})
            chapter += 1
        else:
            kind = _content_kind(text, markers)
        segments.append(Segment(id=ids[i], witness=witness, lang=LANG, text=text, start=chars[a][1],
                                end=chars[b - 1][1], kind=kind, local_chapter=local,
                                fingerprint=fingerprint(text, LANG)))
    return segments, chapters


def _front_matter(texts: list[str], m: Markers) -> int:
    """Number of leading units that are front matter (0 when there is no title block)."""
    if not any(m.front_start.search(t) for t in texts[:m.front_start_within]):
        return 0
    for i, t in enumerate(texts[:m.front_max_units]):
        if m.front_end.search(t) or m.chapter_colophon.search(t) or m.text_end.search(t):
            return i
    return min(len(texts), m.front_max_units)


def _cue(form: Any) -> tuple[str, ...]:
    """A cue as a syllable sequence; a cue must be one non-empty run of syllables."""
    if not isinstance(form, str):
        raise TypeError(f"mantra cue {form!r} is not a string")
    parts = chunks(form, LANG)
    if len(parts) != 1:
        raise ValueError(f"mantra cue {form!r} is empty or contains punctuation")
    return tuple(parts[0])


def _content_kind(text: str, m: Markers) -> str:
    syllables = bo_syllables(text)
    n = len(syllables)
    if n >= MANTRA_MIN_SYLLABLES:
        ratio = bo_sanskrit_syllable_ratio(text)
        if ratio > MANTRA_MIN_RATIO:
            return "mantra"
        cued = (any(tuple(syllables[:len(c)]) == c for c in m.mantra_initial)
                or any(tuple(syllables[-len(c):]) == c for c in m.mantra_final))
        if cued and ratio >= m.mantra_cue_min_ratio:
            return "mantra"
    return "verse_line" if n in VERSE_SYLLABLES else "prose"


def _variants(marks: list[tuple[int, str, str, str]], begin: int, end: int,
              units: list[tuple[int, int]], segments: list[Segment]) -> list[Variant]:
    starts = [a for a, _ in units]
    out = []
    for pos, a, b, coord in marks:
        if begin <= pos < end:
            i = max(bisect_right(starts, pos) - 1, 0)
            out.append(Variant(segment_id=segments[i].id, locus=coord, reading=b, alternative=a,
                               kind="orthographic_variant"))
    return out


def _colophon(segments: list[Segment], markers: Markers) -> dict[str, Any]:
    """Translators, reviser and the verbatim revision statement from the paratext."""
    translators: list[dict[str, str]] = []
    reviser = statement = None
    for role, pattern in markers.colophon_roles:
        for seg in (s for s in segments if s.kind == "paratext"):
            m = pattern.search(seg.text)
            if m is None:
                continue
            if role == REVISER_ROLE:
                reviser, statement = m.group("name"), seg.text
            else:
                translators.append({"role": role, "name": m.group("name"), "segment_id": seg.id})
            break
    return {"paratext": [s.id for s in segments if s.kind == "paratext"], "translators": translators,
            "reviser": reviser, "revision_statement": statement}
