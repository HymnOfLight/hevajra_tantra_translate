"""LLM layer: local open-weight models (Qwen, Gemma, MITRA-Qwen) as *proposal* generators.

Every output of this package is written to the derived layer and enters the
statistics only after human adjudication, or is reported separately as a
model column. See docs/04 for the role catalogue and the hardware plan.
"""

from .backends import LLMBackend, MockBackend, OpenAICompatibleBackend, TransformersBackend, load_backend  # noqa: F401
from .embeddings import EmbeddingSimilarityBackend, HybridSimilarityBackend  # noqa: F401
from . import tasks  # noqa: F401
