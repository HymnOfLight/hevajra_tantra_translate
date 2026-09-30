"""Load reference (Sanskrit) units from a researcher-supplied TSV.

The Sanskrit critical editions are under copyright and are **not** shipped in
this repository. Put a file such as ``data/reference/sa_snellgrove1959.tsv``
in place with the format::

    # comment lines start with '#'
    I.1.1<TAB>evaṃ mayā śrutam ekasmin samaye bhagavān ...
    I.1.2<TAB>...
    I.1.p01<TAB>...          (prose propositions)

One file per Sanskrit witness; the witness id is the file stem. Optional third
column: ``LACUNA`` marks a unit the edition reports as physically missing in
its manuscripts, ``ABSENT`` marks a unit the edition does not have at all
(useful when one edition numbers a verse another edition lacks).
"""

from __future__ import annotations

from pathlib import Path

from .. import normalize
from ..ids import parse_unit_id
from ..segments import Segment


def load_reference_tsv(path: str | Path, witness: str | None = None) -> list[Segment]:
    path = Path(path)
    witness = witness or path.stem
    segs: list[Segment] = []
    for ln, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"{path}:{ln}: expected 'unit_id<TAB>text'")
        uid = parts[0].strip()
        u = parse_unit_id(uid)
        text = parts[1].strip()
        flag = parts[2].strip().upper() if len(parts) > 2 and parts[2].strip() else None
        kind = "verse" if u.kind == "verse" else "prose"
        segs.append(Segment(
            witness=witness, seg_id=uid, lang="sa", text=normalize.sa_normalize(text) if text else "",
            start=uid, end=uid, kind=kind, chapter=u.chapter_key, local_chapter=u.chapter,
            extra={"flag": flag} if flag else {},
        ))
    return segs
