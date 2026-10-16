"""Vector search service – FAISS CPU index over transcript chunks.

Wraps a flat (brute-force) FAISS index with inner-product similarity
(equivalent to cosine when vectors are L2-normalised).
"""

from __future__ import annotations

import logging

import faiss
import numpy as np

from backend.app.core.config import settings
from backend.app.services.chunking_service import TranscriptChunk
from backend.app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)


class VectorIndex:
    """In-memory FAISS index that maps query strings to transcript chunks."""

    def __init__(
        self,
        chunks: list[TranscriptChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Build the index from pre-computed embeddings.

        Args:
            chunks: The transcript chunks (same order as *embeddings*).
            embeddings: 2-D float32 array of shape ``(len(chunks), dim)``.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
                "must have the same length."
            )

        self._chunks = list(chunks)
        dim = int(embeddings.shape[1])

        # Inner-product index (cosine when embeddings are L2-normalised).
        self._index = faiss.IndexFlatIP(dim)
        if len(chunks) > 0:
            contiguous_emb = np.ascontiguousarray(embeddings, dtype=np.float32)
            self._index.add(contiguous_emb)  # type: ignore[arg-type]

        logger.info(
            "FAISS index built: %d vectors of dimension %d", len(chunks), dim
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return self._index.ntotal

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[TranscriptChunk, float]]:
        """Return the *top_k* most relevant chunks for *query*.

        Args:
            query: Free-text question from the user.
            top_k: Number of results (default from settings).

        Returns:
            A list of ``(chunk, score)`` tuples sorted by descending score.
        """
        if self.size == 0:
            return []

        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        top_k = min(top_k, self.size)
        if top_k <= 0:
            return []

        query_vec = np.ascontiguousarray(embed_texts([query]), dtype=np.float32)
        distances, indices = self._index.search(query_vec, top_k)

        results: list[tuple[TranscriptChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue  # FAISS uses -1 for empty slots
            results.append((self._chunks[idx], float(dist)))

        return results


def build_index(chunks: list[TranscriptChunk]) -> VectorIndex:
    """Embed *chunks* and build a FAISS index.

    This is the primary entry-point used by the RAG pipeline.
    """
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    return VectorIndex(chunks, embeddings)
