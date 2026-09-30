"""Embedding-based similarity for the aligner (B1 baseline) and a hybrid with anchors.

``EmbeddingSimilarityBackend`` implements ``align.SimilarityBackend`` with any
sentence-transformers model that covers the languages involved: MITRA-E for
Sanskrit/Tibetan/Chinese Buddhist text, ``BAAI/bge-m3`` or
``Qwen/Qwen3-Embedding-*`` as general multilingual fallbacks. On an RTX 5090
the 8B embedding model fits in bf16; bge-m3 needs < 3 GB.

If ``sentence_transformers`` is not installed the backend can still be
constructed with an explicit ``encode`` callable (tests use a toy encoder).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from ..align import AnchorBackend
from ..segments import Segment

Encoder = Callable[[list[str]], list[list[float]]]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _mean(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


class EmbeddingSimilarityBackend:
    def __init__(self, model_id: str = "BAAI/bge-m3", encode: Encoder | None = None, device: str | None = None,
                 batch_size: int = 64) -> None:
        self.model_id = model_id
        self._encode = encode
        self.device = device
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}

    def _ensure(self) -> Encoder:
        if self._encode is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(self.model_id, device=self.device)

            def enc(texts: list[str]) -> list[list[float]]:
                return model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True).tolist()

            self._encode = enc
        return self._encode

    def warm(self, segments: Sequence[Segment]) -> None:
        """Embed all segments in batches once; the DP then only does cosine look-ups."""
        enc = self._ensure()
        todo = [s for s in segments if self._key(s) not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            for s, v in zip(batch, enc([s.text for s in batch])):
                self._cache[self._key(s)] = v

    @staticmethod
    def _key(s: Segment) -> str:
        return f"{s.witness}|{s.seg_id}"

    def embed(self, s: Segment) -> list[float]:
        k = self._key(s)
        if k not in self._cache:
            self._cache[k] = self._ensure()([s.text])[0]
        return self._cache[k]

    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        if not ref or not wit:
            return 0.0
        c = _cos(_mean([self.embed(s) for s in ref]), _mean([self.embed(s) for s in wit]))
        return max(0.0, c)


class HybridSimilarityBackend:
    """max(anchor, w·embedding) — anchors are sharp when present, embeddings fill the gaps.

    Exposes ``has_anchors`` so the aligner's anchor-conflict penalty still works.
    """

    def __init__(self, anchors: AnchorBackend, embeddings: EmbeddingSimilarityBackend, weight: float = 0.8) -> None:
        self.anchors = anchors
        self.embeddings = embeddings
        self.weight = weight

    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        return max(self.anchors.score(ref, wit), self.weight * self.embeddings.score(ref, wit))

    def has_anchors(self, segs: Sequence[Segment]) -> bool:
        return self.anchors.has_anchors(segs)
