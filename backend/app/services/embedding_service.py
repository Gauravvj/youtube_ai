"""Embedding service – produces dense vectors using sentence-transformers.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` by default, loaded lazily so
import time is zero until the first call.  All computation runs locally on CPU.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from backend.app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Module-level cache – loaded once on first use.
_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer as _ST

        model_name = settings.EMBEDDING_MODEL_NAME
        logger.info("Loading embedding model '%s' …", model_name)
        _model = _ST(model_name, device="cpu")
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings for a list of strings.

    Args:
        texts: Plain-text strings to embed.

    Returns:
        A 2-D float32 numpy array of shape ``(len(texts), dim)``.
    """
    if not texts:
        dim = embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def embedding_dimension() -> int:
    """Return the dimensionality of the loaded model's embeddings."""
    model = _get_model()
    return int(model.get_sentence_embedding_dimension())


def set_model(model: "SentenceTransformer | None") -> None:
    """Explicitly set or reset the cached model (useful for unit tests)."""
    global _model
    _model = model
