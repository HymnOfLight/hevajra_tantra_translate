"""Alignment baselines and controls; every one produces the shared ``core.types.Alignment``.

    anchors     cross-lingual anchor keys of a segment (lexicon: data/lexicon/anchors.yaml)
    similarity  the ``Similarity`` protocol: ``ZeroSimilarity`` and ``AnchorSimilarity``
    dp          monotone bead DP: P1 (dp:zero, length only) and B0 (dp:anchor)
    placebo     P2: an alignment's content shuffled within chapters (placebo:shuffled)
    external    TSV import/export, e.g. of MITRA-E or DharmaNexus alignments

None of these ever populates the matrix; they are the controls Claude must beat.
"""

from .anchors import AnchorLexicon, extract, load_anchor_lexicon
from .dp import SOURCE_ANCHOR, SOURCE_ZERO, DPParams, align, align_groups
from .external import load_tsv, write_tsv
from .placebo import SOURCE_SHUFFLED, shuffle_alignment
from .similarity import AnchorSimilarity, Similarity, ZeroSimilarity

__all__ = [
    "AnchorLexicon", "AnchorSimilarity", "DPParams", "SOURCE_ANCHOR", "SOURCE_SHUFFLED", "SOURCE_ZERO",
    "Similarity", "ZeroSimilarity", "align", "align_groups", "extract", "load_anchor_lexicon",
    "load_tsv", "shuffle_alignment", "write_tsv",
]
