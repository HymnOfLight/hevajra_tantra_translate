"""align.similarity: IDF weights, weighted Jaccard, ZeroSimilarity and AnchorSimilarity.

Segments here are Sanskrit (IAST is Latin script), so no fixture file is needed; the
anchors are the real lexicon's.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from hevajra_matrix.align.anchors import load_anchor_lexicon
from hevajra_matrix.align.similarity import (
    AnchorSimilarity,
    Similarity,
    ZeroSimilarity,
    idf_weights,
    weighted_jaccard,
)
from hevajra_matrix.core.types import Segment

DATA = Path(__file__).resolve().parents[2] / "data"
LEXICON = load_anchor_lexicon(DATA)


def seg(sid: str, text: str, lang: str = "sa") -> Segment:
    return Segment(id=sid, witness="w", lang=lang, text=text, start="1", end="1", kind="verse")


# --------------------------------------------------------------------------- idf and jaccard
def test_idf_is_smoothed_and_decreases_with_document_frequency():
    bags = [{"a"}, {"a"}, {"a", "b"}, set()]
    w = idf_weights(bags)
    n = 4
    assert w["a"] == pytest.approx(math.log((1 + n) / (1 + 3)) + 1)
    assert w["b"] == pytest.approx(math.log((1 + n) / (1 + 1)) + 1)
    assert w["b"] > w["a"] >= 1.0
    assert set(w) == {"a", "b"}


def test_idf_counts_a_key_once_per_bag():
    assert idf_weights([["a", "a"], ["b"]])["a"] == idf_weights([["a"], ["b"]])["a"]


def test_idf_of_no_bags_is_empty():
    assert idf_weights([]) == {}


def test_weighted_jaccard_by_hand():
    w = {"a": 1.0, "b": 3.0, "c": 2.0}
    assert weighted_jaccard({"a", "b"}, {"b", "c"}, w) == pytest.approx(3.0 / 6.0)
    assert weighted_jaccard({"a"}, {"a"}, w) == 1.0
    assert weighted_jaccard({"a"}, {"c"}, w) == 0.0
    assert weighted_jaccard(set(), set(), w) == 0.0
    assert weighted_jaccard({"a", "x"}, {"a"}, w, default=3.0) == pytest.approx(1.0 / 4.0)


def test_a_rare_shared_key_outweighs_a_common_one():
    """IDF weighting: sharing a rare anchor is stronger evidence than sharing a common one."""
    w = idf_weights([{"common"}] * 50 + [{"rare"}])
    rare = weighted_jaccard({"common", "rare"}, {"rare"}, w)
    common = weighted_jaccard({"common", "rare"}, {"common"}, w)
    assert rare > 0.5 > common


# --------------------------------------------------------------------------- similarities
def test_both_implementations_satisfy_the_protocol():
    sim: Similarity = ZeroSimilarity()
    assert sim.score([seg("a", "vajra")], [seg("b", "vajra")]) == 0.0
    assert sim.conflict([seg("a", "vajra")], [seg("b", "padma")]) is False
    sim = AnchorSimilarity(LEXICON, {})
    assert sim.score([seg("a", "vajra")], [seg("b", "vajra")]) == 1.0


def test_anchor_similarity_scores_shared_anchors():
    ref = [seg("r1", "vajragarbha uvāca")]
    sim = AnchorSimilarity.fit(LEXICON, ref + [seg("w1", "vajragarbha padma"), seg("w2", "padma")])
    assert sim.score(ref, [seg("w1", "vajragarbha padma")]) == pytest.approx(
        sim.idf["name:vajragarbha"] / (sim.idf["name:vajragarbha"] + sim.idf["term:padma"]))
    assert sim.score(ref, [seg("w2", "padma")]) == 0.0


def test_a_side_is_the_union_of_its_segments():
    sim = AnchorSimilarity(LEXICON, {})
    both = [seg("r1", "vajragarbha"), seg("r2", "padma")]
    assert sim.side(both) == {"name:vajragarbha", "term:padma"}
    assert sim.score(both, [seg("w", "padma vajragarbha")]) == 1.0
    assert sim.score(both, [seg("w", "padma")]) == pytest.approx(0.5)


def test_conflict_needs_evidence_on_both_sides_and_no_overlap():
    sim = AnchorSimilarity(LEXICON, {})
    assert sim.conflict([seg("r", "vajragarbha")], [seg("w", "padma")])
    assert not sim.conflict([seg("r", "vajragarbha")], [seg("w", "vajragarbha padma")])
    assert not sim.conflict([seg("r", "vajragarbha")], [seg("w", "uvāca")])
    assert not sim.conflict([seg("r", "uvāca")], [seg("w", "uvāca")])


def test_unseen_keys_weigh_like_the_rarest_known_key():
    sim = AnchorSimilarity(LEXICON, {"term:padma": 1.0, "term:vajra": 5.0})
    assert sim.unseen_weight == 5.0
    assert sim.score([seg("r", "padma")], [seg("w", "padma śmaśānaṃ")]) == pytest.approx(1.0 / 6.0)


def test_fit_weights_come_from_the_given_segments():
    segments = [seg("a", "vajra"), seg("b", "vajra"), seg("c", "vajra padma")]
    sim = AnchorSimilarity.fit(LEXICON, segments)
    assert sim.idf["term:padma"] > sim.idf["term:vajra"]
    assert sim.idf == idf_weights([{"term:vajra"}, {"term:vajra"}, {"term:vajra", "term:padma"}])


def test_embedding_similarity_is_gone():
    """Defect #15: the hybrid anchor/embedding similarity (and its noise floor) was deleted."""
    assert importlib.util.find_spec("hevajra_matrix.llm.embeddings") is None


def test_memo_does_not_change_results():
    sim = AnchorSimilarity.fit(LEXICON, [seg("a", "vajra padma")])
    fresh = AnchorSimilarity(LEXICON, sim.idf)
    pair = ([seg("a", "vajra padma")], [seg("b", "padma")])
    assert sim.score(*pair) == fresh.score(*pair) == sim.score(*pair)
