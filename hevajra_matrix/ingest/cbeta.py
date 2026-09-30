"""Ingest a CBETA TEI P5 file (e.g. T18n0892.xml) into coordinate-bearing segments.

What is kept
    * every character with the Taishō line (``lb/@n``, e.g. ``0587c11``) it sits on
    * chapter (品) boundaries from ``cb:mulu[@type='品']``
    * verse lines (``lg/l``) as individual ``verse_line`` segments
    * prose split at sentence-final punctuation into ``prose`` segments
    * inline phonetic glosses (``note[@place='inline']`` such as 引 / 二合) are removed
      from the text but counted in ``extra['inline_notes']``
    * the Taishō Sanskrit foot-notes (``cb:tt`` in ``back``) are returned as a
      gloss table: they are ready-made anchors (漢文 ↔ Sanskrit)
    * the apparatus (``app``) is returned so that variant readings are never lost
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from ..segments import Segment

TEI = "{http://www.tei-c.org/ns/1.0}"
CB = "{http://www.cbeta.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

# Clause-level split: a Chinese clause (between ，。；：？！) is the granularity that
# matches one Tibetan shad-unit / one Sanskrit pāda; sentence-level would force
# 3:1–6:1 beads and inflate NULL alignments.
_CLAUSE_PUNCT = "。？！，；：、"
_SENT_RE = re.compile(rf"[^{_CLAUSE_PUNCT}]*[{_CLAUSE_PUNCT}]+[」』]?|[^{_CLAUSE_PUNCT}]+$")
_SKIP_TAGS = {TEI + "note", TEI + "anchor", TEI + "pb"}


@dataclass
class CbetaDocument:
    witness: str
    segments: list[Segment]
    glosses: list[dict]            # {"zh":..., "sa":..., "anchor":...}
    apparatus: list[dict]          # {"from":..., "lem":..., "rdg":[...], "wit":[...]}
    chapters: list[dict]           # {"n": 1, "title": "金剛部序品", "start": "0587c10"}
    title_note: str | None = None  # 卷一經題下夾注


@dataclass
class _State:
    witness: str
    line: str = "0000a00"
    chapter: int | None = None
    counter: int = 0
    segments: list[Segment] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)
    title_note: str | None = None


def _chars(el: ET.Element, st: _State) -> list[tuple[str, str]]:
    """Characters of an inline element with the Taishō line each one sits on.

    Advances ``st.line`` as ``lb`` elements are met, drops notes/anchors.
    """
    out: list[tuple[str, str]] = []

    def walk(e: ET.Element, top: bool) -> None:
        if e.tag == TEI + "lb":
            st.line = e.get("n", st.line)
        elif e.tag in _SKIP_TAGS:
            pass
        else:
            if e.text:
                out.extend((ch, st.line) for ch in e.text if ch not in "\n\r\t")
            for c in e:
                walk(c, False)
        if not top and e.tail:
            out.extend((ch, st.line) for ch in e.tail if ch not in "\n\r\t")

    walk(el, True)
    return out


def _emit(st: _State, chars: list[tuple[str, str]], kind: str, **extra) -> None:
    text = "".join(ch for ch, _ in chars).strip()
    if not text:
        return
    st.counter += 1
    st.segments.append(Segment(
        witness=st.witness, seg_id=f"zh:{chars[0][1]}:{st.counter:05d}", lang="zh", text=text,
        start=chars[0][1], end=chars[-1][1], kind=kind, local_chapter=st.chapter, extra=dict(extra),
    ))


def _emit_split(st: _State, chars: list[tuple[str, str]], kind: str, **extra) -> None:
    text = "".join(ch for ch, _ in chars)
    pos = 0
    for m in _SENT_RE.finditer(text):
        sent = m.group(0)
        n = len(sent)
        if sent.strip(_CLAUSE_PUNCT + "「」『』 "):
            _emit(st, chars[pos:pos + n], kind, **extra)
        pos += n


def _walk(el: ET.Element, st: _State) -> None:
    for child in el:
        tag = child.tag
        if tag == TEI + "lb":
            st.line = child.get("n", st.line)
        elif tag == CB + "mulu":
            if child.get("type") == "品":
                st.chapter = int(child.get("n"))
                st.chapters.append({"n": st.chapter, "title": (child.text or "").strip(), "start": st.line})
        elif tag == TEI + "head":
            _emit(st, _chars(child, st), "head")
        elif tag in (CB + "jhead", TEI + "byline", CB + "docNumber"):
            if tag == CB + "jhead" and st.title_note is None:
                note = child.find(f".//{TEI}note")
                if note is not None and note.text:
                    st.title_note = note.text.strip()
            _emit(st, _chars(child, st), "meta")
        elif tag == TEI + "lg":
            for sub in child:
                if sub.tag == TEI + "lb":
                    st.line = sub.get("n", st.line)
                elif sub.tag == TEI + "l":
                    _emit_split(st, _chars(sub, st), "verse_line")
        elif tag == TEI + "p":
            n_notes = sum(1 for _ in child.iter(TEI + "note"))
            _emit_split(st, _chars(child, st), "prose", **({"inline_notes": n_notes} if n_notes else {}))
        elif tag in _SKIP_TAGS:
            pass
        else:
            _walk(child, st)


def _parse_back(root: ET.Element) -> tuple[list[dict], list[dict]]:
    glosses, apparatus = [], []
    back = root.find(f".//{TEI}back")
    if back is None:
        return glosses, apparatus
    for tt in back.iter(CB + "tt"):
        zh = sa = None
        for t in tt.findall(CB + "t"):
            txt = "".join(t.itertext()).strip()
            if t.get(XML_LANG) == "zh-Hant":
                zh = txt
            elif t.get(XML_LANG) == "sa":
                sa = txt.rstrip(".")
        if zh and sa:
            glosses.append({"zh": zh, "sa": sa, "anchor": tt.get("from", "")})
    for app in back.iter(TEI + "app"):
        lem = app.find(TEI + "lem")
        rdgs = app.findall(TEI + "rdg")
        apparatus.append({
            "from": app.get("from", ""),
            "lem": "".join(lem.itertext()).strip() if lem is not None else "",
            "rdg": ["".join(r.itertext()).strip() for r in rdgs],
            "wit": [r.get("wit", "") for r in rdgs],
        })
    return glosses, apparatus


def parse_cbeta_tei(path: str | Path, witness: str = "zh_T0892_song") -> CbetaDocument:
    root = ET.parse(str(path)).getroot()
    body = root.find(f".//{TEI}body")
    if body is None:
        raise ValueError("no <body> in TEI file")
    st = _State(witness=witness)
    _walk(body, st)
    glosses, apparatus = _parse_back(root)
    return CbetaDocument(witness=witness, segments=st.segments, glosses=glosses,
                         apparatus=apparatus, chapters=st.chapters, title_note=st.title_note)
