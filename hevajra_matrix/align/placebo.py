"""Control P2: an alignment with its content shuffled within chapters.

``shuffle_alignment`` keeps every reference unit and every count of the input but hands
each unit the (wit_ids, relation, polarity_flip) of a random unit of the same chapter.
It therefore has exactly the input's relation and status counts per chapter and none of
its content, which makes it the chance floor for link and status agreement (synthesis
5.3). For the same reason P2 never appears in aggregate count tables (critique A7).

Fields that describe the original unit's own text (confidence, quotes, flags) are
cleared. Witness-only links are kept as they are (only their ``source`` changes), so
witness-only metrics of P2 equal those of its input by construction and carry no
information.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Mapping

from ..core.types import Alignment, Link

SOURCE_SHUFFLED = "placebo:shuffled"


def shuffle_alignment(alignment: Alignment, chapter_of: Mapping[str, str], seed: int) -> Alignment:
    """Permute (wit_ids, relation, polarity_flip) across the reference units of each chapter.

    ``chapter_of`` maps every reference unit id of ``alignment`` to its chapter; a missing
    unit raises ``ValueError``. Each chapter is shuffled by its own generator, seeded
    from ``seed`` and the chapter key, so adding a chapter does not change the others.
    Link order is preserved.
    """
    missing = sorted({link.ref_id for link in alignment.links
                      if link.ref_id is not None and link.ref_id not in chapter_of})
    if missing:
        raise ValueError(f"chapter_of has no chapter for {len(missing)} unit(s), e.g. {missing[:3]}")
    positions: dict[str, list[int]] = {}
    for pos, link in enumerate(alignment.links):
        if link.ref_id is not None:
            positions.setdefault(chapter_of[link.ref_id], []).append(pos)
    out = list(alignment.links)
    for chapter, where in positions.items():
        contents = [(out[p].wit_ids, out[p].relation, out[p].polarity_flip) for p in where]
        random.Random(f"{seed}|{chapter}").shuffle(contents)
        for p, (wit_ids, relation, flip) in zip(where, contents):
            out[p] = Link(ref_id=out[p].ref_id, wit_ids=wit_ids, relation=relation,
                          polarity_flip=flip, source=SOURCE_SHUFFLED)
    for pos, link in enumerate(out):
        if link.ref_id is None:
            out[pos] = replace(link, source=SOURCE_SHUFFLED)
    return Alignment(source=SOURCE_SHUFFLED, reference=alignment.reference,
                     witness=alignment.witness, links=tuple(out))
