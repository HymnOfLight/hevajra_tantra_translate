"""Ingest the Esukhia Derge Kangyur plain-text volume and slice one or more Toh numbers.

File format (Esukhia/derge-kangyur, text/*.txt)::

    [1b]
    [1b.1]{D417}༄༅༅། །རྒྱ་གར་སྐད་དུ། ...
    [1b.2]ཐོས་པ་དུས་གཅིག་ན། ...

* ``[Xa]``/``[Xb]`` are folio markers, ``[Xa.N]`` line markers, ``{Dnnn}`` text starts.
* Segments are cut at shad (།). Each keeps its ``folio.line`` start/end.
* Chapter colophons (…ལེའུ་སྟེ་…པའོ / …རྫོགས་སོ) are detected and used to number
  local chapters; the number is *not* trusted blindly — a running count is
  kept and the colophon's own ordinal is stored for cross-checking.
* ``kind`` is ``verse_line`` when the shad-unit has 7, 9 or 11 syllables
  (classical metres), ``colophon`` for chapter ends, ``mantra`` when the unit
  is dominated by Sanskrit transliteration, otherwise ``prose``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import normalize
from ..segments import Segment

_FOLIO_RE = re.compile(r"^\[(\d+[ab])(?:\.(\d+))?\]")
_TEXT_START_RE = re.compile(r"\{D(\d+)\}")
_COLOPHON_RE = re.compile(r"ལེའུ་(?:ཞེས་བྱ་བ་)?སྟེ་?([\u0f00-\u0fff་]*?)(པའོ|བའོ|པོའོ)")
_END_RE = re.compile(r"རྫོགས་སོ")
_NIDANA_RE = re.compile(r"འདི་སྐད་བདག་གིས་ཐོས་པ")
_TITLE_RE = re.compile(r"རྒྱ་གར་སྐད་དུ")
_ORDINALS = {
    "དང་པོ": 1, "གཉིས་པ": 2, "གསུམ་པ": 3, "བཞི་པ": 4, "ལྔ་པ": 5, "དྲུག་པ": 6, "བདུན་པ": 7,
    "བརྒྱད་པ": 8, "དགུ་པ": 9, "དགུ་བ": 9, "བཅུ་པ": 10, "བཅུ་གཅིག་པ": 11, "བཅུ་གཉིས་པ": 12,
}


@dataclass
class DergeText:
    witness: str
    toh: str
    segments: list[Segment]
    chapters: list[dict]   # {"n": running number, "ordinal": from colophon or None, "end": "3a.5", "text": ...}
    colophon: str | None = None


@dataclass
class _Stream:
    chars: list[tuple[str, str]] = field(default_factory=list)  # (char, folio.line)


def _read_stream(text: str) -> tuple[_Stream, dict[str, int]]:
    """Flatten the file into (char, coordinate) pairs; return {toh: char offset}."""
    stream = _Stream()
    starts: dict[str, int] = {}
    coord = "0a.0"
    for raw in text.splitlines():
        line = raw
        m = _FOLIO_RE.match(line)
        if m:
            folio, ln = m.group(1), m.group(2)
            if ln is None:
                continue  # bare folio marker
            coord = f"{folio}.{ln}"
            line = line[m.end():]
        i = 0
        while i < len(line):
            tm = _TEXT_START_RE.match(line, i)
            if tm:
                starts[f"D{tm.group(1)}"] = len(stream.chars)
                i = tm.end()
                continue
            if line[i] not in "\r\n":
                stream.chars.append((line[i], coord))
            i += 1
    return stream, starts


def _split_units(chars: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    units, cur = [], []
    for ch, c in chars:
        cur.append((ch, c))
        if ch in normalize.SHAD_CHARS:
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units


def _classify(text: str) -> str:
    if _COLOPHON_RE.search(text) or _END_RE.search(text):
        return "colophon"
    if normalize.bo_sanskrit_syllable_ratio(text) > 0.5 and normalize.bo_length(text) >= 3:
        return "mantra"
    n = normalize.bo_length(text)
    if n in (7, 9, 11):
        return "verse_line"
    return "prose"


def _ordinal(text: str) -> int | None:
    m = _COLOPHON_RE.search(text)
    if not m:
        return None
    body = m.group(1) + m.group(2)
    for k in sorted(_ORDINALS, key=len, reverse=True):
        if body.endswith(k + "འོ") or body.endswith(k):
            return _ORDINALS[k]
    return None


def parse_derge_volume(path: str | Path, toh_numbers: list[str] | None = None,
                       witness: str = "bo_derge_D417_418") -> list[DergeText]:
    """Parse one Esukhia volume file and return one ``DergeText`` per requested Toh.

    ``toh_numbers`` like ["D417", "D418"]; default: every text in the volume.
    The end of a text is the start of the next ``{Dnnn}`` marker.
    """
    text = Path(path).read_text(encoding="utf-8")
    stream, starts = _read_stream(text)
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    wanted = toh_numbers or [k for k, _ in ordered]
    out: list[DergeText] = []
    for idx, (toh, start) in enumerate(ordered):
        if toh not in wanted:
            continue
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(stream.chars)
        chars = stream.chars[start:end]
        segments: list[Segment] = []
        chapters: list[dict] = []
        running = 1
        units = _split_units(chars)
        head_text = "".join(ch for u in units[:4] for ch, _ in u)
        # title in Sanskrit/Tibetan and translator's homage precede the nidāna; they are
        # paratext with no counterpart in the Chinese and must not enter alignment
        in_front_matter = _TITLE_RE.search(head_text) is not None
        for i, unit in enumerate(units):
            utext = "".join(ch for ch, _ in unit).strip()
            if not utext or utext in ("།", "། །", "།། །།"):
                continue
            kind = _classify(utext)
            if in_front_matter:
                if _NIDANA_RE.search(utext) or i > 12 or kind == "colophon":
                    in_front_matter = False
                else:
                    kind = "meta"
            seg = Segment(
                witness=witness, seg_id=f"bo:{toh}:{i:05d}", lang="bo", text=utext,
                start=unit[0][1], end=unit[-1][1], kind=kind, local_chapter=running,
                extra={"toh": toh},
            )
            segments.append(seg)
            if kind == "colophon":
                is_text_end = bool(_END_RE.search(utext)) and not _COLOPHON_RE.search(utext)
                chapters.append({"n": running, "ordinal": _ordinal(utext), "end": unit[-1][1],
                                 "text": utext, "text_end": is_text_end})
                if not is_text_end:
                    running += 1
        colophon = segments[-1].text if segments else None
        out.append(DergeText(witness=witness, toh=toh, segments=segments, chapters=chapters, colophon=colophon))
    return out
