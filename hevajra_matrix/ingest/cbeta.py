"""Ingest a CBETA TEI P5 file (T18n0892.xml) into coordinate-keyed segments and evidence.

Segments (in document order)
    head        chapter heading (``cb:div[@type="pin"]/head``)
    meta        fascicle headings, byline, document number
    verse_line  one clause of a verse line (``lg/l``)
    prose       one clause of a paragraph
    mantra      a dharani paragraph (``p[@cb:type="dharani"]``), never split
    note        an inline note (``note[@place="inline"]``), placed right after its host

Prose and verse lines are cut after clause-final punctuation: a Chinese clause is the
granularity that matches one Tibetan shad unit or one Sanskrit pada, whereas sentences
would force 3:1 to 6:1 beads. Punctuation-only pieces are dropped.

Notes never enter the reading text of a content segment. Each becomes its own segment
with ``extra = {"note_class": <class>, "host": <segment id>}``; the host is the content
segment holding the character just before the note (the note comments on what precedes
it). Classes come from ``data/lexicon/notes_zh.yaml`` (see ``ingest.notes``).

Evidence outside the reading text
    footnotes   all Taisho footnotes (``note[@type="orig"]``), located at their anchors
    variants    the CBETA apparatus (``app``): the lemma is in the text, each ``rdg`` is kept
    metadata    chapters and the Taisho glosses (``cb:tt``: Chinese term <-> Sanskrit)

Coordinates: a character's line is the ``lb/@n`` in force where it stands (line breaks
inside a note advance the line too). ``local_chapter`` is ``"pin<n>"`` inside a chapter
division and None outside (fascicle headings between chapters, byline).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..core.ids import assign_ids
from ..core.textnorm import fingerprint, for_quote
from ..core.types import Segment
from . import CONTENT_KINDS, Footnote, IngestResult, Variant, duplicate_ids, kind_counts
from . import notes as note_rules

TEI = "{http://www.tei-c.org/ns/1.0}"
CB = "{http://www.cbeta.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
LANG = "zh"

# Ideographic full stop; fullwidth question mark, exclamation mark, comma, semicolon,
# colon; ideographic comma. A clause ends after a run of these, plus one closing
# bracket (right corner bracket or right white corner bracket) if it follows.
_CLAUSE_PUNCT = "\u3002\uff1f\uff01\uff0c\uff1b\uff1a\u3001"
_CLOSE = "\u300d\u300f"
_CLAUSE_RE = re.compile(rf"[^{_CLAUSE_PUNCT}]*[{_CLAUSE_PUNCT}]+[{_CLOSE}]?|[^{_CLAUSE_PUNCT}]+$")
_SPLIT_KINDS = frozenset({"prose", "verse_line"})
_META_TAGS = frozenset({CB + "jhead", TEI + "byline", CB + "docNumber"})
_SILENT_TAGS = frozenset({TEI + "pb", CB + "mulu", TEI + "caesura", TEI + "milestone"})
_LAYOUT = frozenset("\n\r\t")                    # source layout, not text
_TEXT_ID_RE = re.compile(r"([A-Z]+)[0-9]+n([0-9]+[A-Za-z]?)")   # "T18n0892" -> ("T", "0892")
_LATIN_RE = re.compile("[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")      # start of a romanised text


@dataclass
class _Block:
    """Characters of one structural element (paragraph, verse line, heading)."""

    kind: str
    local: str | None
    base: int                                   # stream position of the first character
    chars: list[tuple[str, str]] = field(default_factory=list)   # (character, line)

    def text(self) -> str:
        return "".join(ch for ch, _ in self.chars)


@dataclass(frozen=True)
class _Note:
    pos: int            # stream position: number of text characters before the note
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class _Anchor:
    pos: int
    line: str


@dataclass
class _Walk:
    line: str = ""
    local: str | None = None
    pos: int = 0
    blocks: list[_Block] = field(default_factory=list)
    notes: list[_Note] = field(default_factory=list)
    anchors: dict[str, _Anchor] = field(default_factory=dict)


@dataclass(frozen=True)
class _Piece:
    """One future content segment: a slice of a block."""

    block: _Block
    a: int
    b: int

    @property
    def start(self) -> int:
        return self.block.base + self.a


def parse(path: Path, witness: str, data_dir: Path) -> IngestResult:
    """Parse one CBETA TEI file; ``data_dir`` holds ``lexicon/notes_zh.yaml``."""
    rules = note_rules.load_rules(data_dir)
    root = ET.parse(str(path)).getroot()
    prefix = _prefix(root)
    body = root.find(f"{TEI}text/{TEI}body")
    if body is None:
        raise ValueError(f"{path}: no <text>/<body>")
    walk = _Walk()
    _walk(body, walk)
    pieces = [_Piece(b, x, y) for b in walk.blocks for x, y in _spans(b)]
    if not pieces:
        raise ValueError(f"{path}: no text in <body>")
    piece_at = _locator(pieces)

    segments = _segments(pieces, walk.notes, piece_at, prefix, witness, rules)
    content_ids = [s.id for s in segments if s.kind != "note"]
    anchored = _anchored(walk.anchors, piece_at, content_ids)
    footnotes = _footnotes(root, anchored, rules)
    variants = _apparatus(root, anchored)
    notes = [s for s in segments if s.kind == "note"]
    metadata = {
        "witness": witness,
        "prefix": prefix,
        "chapters": [{"local": s.local_chapter, "title": s.text, "start": s.start}
                     for s in segments if s.kind == "head"],
        "glosses": _glosses(root, anchored),
    }
    report = {
        "segments": len(segments),
        "kinds": kind_counts(segments),
        "content_segments": sum(s.kind in CONTENT_KINDS for s in segments),
        "notes": len(notes),
        "note_classes": dict(sorted(Counter(s.extra["note_class"] for s in notes).items())),
        "unclassified_notes": [s.id for s in notes if s.extra["note_class"] == note_rules.UNCLASSIFIED],
        "footnotes": len(footnotes),
        "footnote_sources": dict(sorted(Counter(f.source for f in footnotes).items())),
        "variants": len(variants),
        "chapters": len(metadata["chapters"]),
        "duplicate_ids": duplicate_ids(segments),
    }
    return IngestResult(tuple(segments), tuple(footnotes), tuple(variants), metadata, report)


# --------------------------------------------------------------------------- walking
def _prefix(root: ET.Element) -> str:
    """Id prefix from the TEI ``xml:id``: "T18n0892" -> "T0892"."""
    m = _TEXT_ID_RE.fullmatch(root.get(XML_ID, ""))
    if m is None:
        raise ValueError(f"TEI root xml:id {root.get(XML_ID)!r} is not a CBETA text id such as 'T18n0892'")
    return m.group(1) + m.group(2)


def _walk(el: ET.Element, st: _Walk) -> None:
    """Structural level: chapters, headings, paragraphs, verse groups."""
    for child in el:
        tag = child.tag
        if tag == TEI + "lb":
            st.line = child.get("n", st.line)
        elif tag == CB + "div" and child.get("type") == "pin":
            outer, st.local = st.local, _chapter_key(child)
            _walk(child, st)
            st.local = outer
        elif tag == TEI + "head":
            _block(child, "head", st)
        elif tag in _META_TAGS:
            _block(child, "meta", st)
        elif tag == TEI + "p":
            _block(child, "mantra" if child.get(CB + "type") == "dharani" else "prose", st)
        elif tag == TEI + "l":
            _block(child, "verse_line", st)
        elif tag == TEI + "anchor":
            _anchor(child, st)
        elif tag == TEI + "note":
            _note(child, st)
        elif tag not in _SILENT_TAGS:
            _walk(child, st)          # lg, cb:juan, other containers


def _chapter_key(div: ET.Element) -> str:
    mulu = div.find(CB + "mulu")
    if mulu is None or not (mulu.get("n") or "").isdigit():
        raise ValueError(f"chapter division without a numbered cb:mulu near {div.get(XML_ID)!r}")
    return f"pin{int(mulu.get('n'))}"


def _block(el: ET.Element, kind: str, st: _Walk) -> None:
    block = _Block(kind=kind, local=st.local, base=st.pos)
    _inline(el, block, st)
    st.blocks.append(block)


def _inline(el: ET.Element, block: _Block, st: _Walk) -> None:
    """Text level: characters with their line; notes and anchors as positions."""
    _add(el.text, block, st)
    for child in el:
        tag = child.tag
        if tag == TEI + "lb":
            st.line = child.get("n", st.line)
        elif tag == TEI + "note":
            _note(child, st)
        elif tag == TEI + "anchor":
            _anchor(child, st)
        elif tag not in _SILENT_TAGS:
            _inline(child, block, st)   # g (rare characters), foreign, ...
        _add(child.tail, block, st)


def _add(text: str | None, block: _Block, st: _Walk) -> None:
    for ch in text or "":
        if ch not in _LAYOUT:
            block.chars.append((ch, st.line))
            st.pos += 1


def _note(el: ET.Element, st: _Walk) -> None:
    start, parts = st.line, []

    def collect(e: ET.Element) -> None:
        parts.append(e.text or "")
        for c in e:
            if c.tag == TEI + "lb":
                st.line = c.get("n", st.line)
            else:
                collect(c)
            parts.append(c.tail or "")

    collect(el)
    st.notes.append(_Note(pos=st.pos, start=start, end=st.line, text="".join(parts)))


def _anchor(el: ET.Element, st: _Walk) -> None:
    anchor_id = el.get(XML_ID)
    if anchor_id:
        st.anchors[anchor_id] = _Anchor(pos=st.pos, line=st.line)


# --------------------------------------------------------------------------- segments
def _spans(block: _Block) -> list[tuple[int, int]]:
    """(start, end) offsets in the block of the pieces that become segments."""
    text = block.text()
    if block.kind in _SPLIT_KINDS:
        cuts = [(m.start(), m.end()) for m in _CLAUSE_RE.finditer(text)]
    else:
        cuts = [(0, len(text))]
    spans = []
    for a, b in cuts:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if for_quote(text[a:b], LANG):
            spans.append((a, b))
    return spans


def _locator(pieces: list[_Piece]) -> Callable[[int], int]:
    """Index of the last piece starting at or before a stream position (0 if none)."""
    starts = [p.start for p in pieces]
    return lambda pos: max(bisect_right(starts, pos) - 1, 0)


def _segments(pieces: list[_Piece], notes: list[_Note], piece_at: Callable[[int], int],
              prefix: str, witness: str, rules: note_rules.NoteRules) -> list[Segment]:
    """Content segments in document order, each followed by the notes it hosts."""
    hosted: dict[int, list[_Note]] = {}
    for note in notes:
        hosted.setdefault(piece_at(note.pos - 1), []).append(note)
    entries: list[tuple[str, bool]] = []
    for i, piece in enumerate(pieces):
        entries.append((piece.block.chars[piece.a][1], False))
        entries.extend((note.start, True) for note in hosted.get(i, ()))
    ids = iter(assign_ids(prefix, entries))
    out: list[Segment] = []
    for i, piece in enumerate(pieces):
        chars = piece.block.chars[piece.a:piece.b]
        text = "".join(ch for ch, _ in chars)
        host = Segment(id=next(ids), witness=witness, lang=LANG, text=text, start=chars[0][1],
                       end=chars[-1][1], kind=piece.block.kind, local_chapter=piece.block.local,
                       fingerprint=fingerprint(text, LANG))
        out.append(host)
        for note in hosted.get(i, ()):
            text = note_rules.normalise_note(note.text)
            out.append(Segment(
                id=next(ids), witness=witness, lang=LANG, text=text, start=note.start, end=note.end,
                kind="note", local_chapter=host.local_chapter, fingerprint=fingerprint(text, LANG),
                extra={"note_class": note_rules.classify(text, rules), "host": host.id},
            ))
    return out


def _anchored(anchors: dict[str, _Anchor], piece_at: Callable[[int], int],
              content_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Anchor xml:id -> (line, id of the segment holding the character after the anchor)."""
    return {k: (a.line, content_ids[piece_at(a.pos)]) for k, a in anchors.items()}


# --------------------------------------------------------------------------- back matter
def _footnotes(root: ET.Element, anchored: dict[str, tuple[str, str]],
               rules: note_rules.NoteRules) -> list[Footnote]:
    out = []
    for note in root.iter(TEI + "note"):
        if note.get("type") != "orig":
            continue
        n = note.get("n", "")
        target = (note.get("target") or "").lstrip("#")
        if target not in anchored:
            raise ValueError(f"Taisho footnote {n!r}: target {target!r} is not an anchor in the body")
        line, segment_id = anchored[target]
        text = " ".join("".join(note.itertext()).split())
        zh_lemma, sa_text = _split_lemma(text)
        out.append(Footnote(n=n, locus=line, segment_id=segment_id, zh_lemma=zh_lemma, sa_text=sa_text,
                            source=note_rules.footnote_source(text, rules), text=text))
    return out


def _split_lemma(text: str) -> tuple[str, str]:
    """Chinese lemma (everything before the first Latin letter) and the verbatim rest."""
    m = _LATIN_RE.search(text)
    if m is None:
        return text, ""
    return text[:m.start()].strip().rstrip(","), text[m.start():].strip()


def _apparatus(root: ET.Element, anchored: dict[str, tuple[str, str]]) -> list[Variant]:
    out = []
    for app in root.iter(TEI + "app"):
        start = (app.get("from") or "").lstrip("#")
        if start not in anchored:
            raise ValueError(f"apparatus entry: 'from' {start!r} is not an anchor in the body")
        line, segment_id = anchored[start]
        lem = app.find(TEI + "lem")
        reading = _plain_text(lem) if lem is not None else ""
        for rdg in app.findall(TEI + "rdg"):
            out.append(Variant(segment_id=segment_id, locus=line, reading=reading,
                               alternative=_plain_text(rdg), kind="apparatus"))
    return out


def _glosses(root: ET.Element, anchored: dict[str, tuple[str, str]]) -> list[dict[str, str]]:
    """Taisho glosses: pairs of a Chinese term and its Sanskrit equivalent."""
    out = []
    for tt in root.iter(CB + "tt"):
        by_lang = {t.get(XML_LANG): _plain_text(t) for t in tt.findall(CB + "t")}
        zh, sa = by_lang.get("zh-Hant", ""), by_lang.get("sa", "").rstrip(".")
        if zh and sa:
            start = (tt.get("from") or "").lstrip("#")
            line, segment_id = anchored.get(start, ("", ""))
            out.append({"zh": zh, "sa": sa, "locus": line, "segment_id": segment_id})
    return out


def _plain_text(el: ET.Element) -> str:
    """Text of an element without nested notes, with layout whitespace removed."""
    parts = [el.text or ""]
    for child in el:
        if child.tag != TEI + "note":
            parts.append(_plain_text(child))
        parts.append(child.tail or "")
    return "".join(ch for ch in "".join(parts) if ch not in _LAYOUT).strip()
