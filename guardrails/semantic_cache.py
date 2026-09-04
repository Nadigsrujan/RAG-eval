"""
Semantic Cache — in-memory cache mapping question embeddings to retrieval and generation results.

Uses cosine similarity over query embeddings to bypass the BM25/FAISS/reranker pipeline
for identical or semantically equivalent questions (similarity >= threshold).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fallback cosine similarity using numpy."""
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        denom = norm_a * norm_b
        denom[denom == 0] = 1e-12
        return dot / denom


class SemanticCache:
    """
    In-memory cache mapping question embeddings to retrieval results.

    Stores (question, embedding, result) entries and provides fast nearest-neighbor
    lookup via cosine similarity.

    Args:
        similarity_threshold: Minimum cosine similarity score (0.0 - 1.0)
            required to consider a query a cache hit. Default 0.95.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
        self._cache: dict[str, dict[str, Any]] = {}  # question -> {embedding, result}

    def check(
        self,
        query: str,
        embedder: Embedder | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> dict | None:
        """
        Check if query matches a cached question above the similarity threshold.

        Args:
            query: The user's question string.
            embedder: Embedder instance to compute embedding if query_embedding is None.
            query_embedding: Precomputed query embedding vector (optional).

        Returns:
            The cached result dictionary if similarity >= threshold, otherwise None.
        """
        if not self._cache:
            return None

        if query_embedding is None:
            if embedder is None:
                raise ValueError("Either embedder or query_embedding must be provided to check cache.")
            query_embedding = embedder.embed_query(query)

        q_vec = query_embedding.reshape(1, -1)

        best_sim = -1.0
        best_result: dict | None = None
        best_question: str | None = None

        for cached_q, cached_item in self._cache.items():
            cached_vec = cached_item["embedding"].reshape(1, -1)
            sim = float(cosine_similarity(q_vec, cached_vec)[0][0])
            if (sim >= self.threshold or np.isclose(sim, self.threshold, atol=1e-5)) and sim > best_sim:
                best_sim = sim
                best_result = cached_item["result"]
                best_question = cached_q

        if best_result is not None:
            logger.info(
                "Semantic cache HIT: '%s' matched cached query '%s' (similarity=%.4f >= %.2f)",
                query,
                best_question,
                best_sim,
                self.threshold,
            )
            # Make a shallow copy and tag cache provenance
            result_copy = dict(best_result)
            if "metadata" in result_copy and isinstance(result_copy["metadata"], dict):
                result_copy["metadata"] = dict(result_copy["metadata"])
                result_copy["metadata"]["cached"] = True
                result_copy["metadata"]["cached_similarity"] = round(best_sim, 4)
                result_copy["metadata"]["matched_question"] = best_question
            return result_copy

        logger.debug("Semantic cache MISS for query: '%s'", query)
        return None

    def store(self, query: str, embedding: np.ndarray, result: dict) -> None:
        """
        Store a query embedding and its retrieval result in cache.

        Args:
            query: The user's question string.
            embedding: 1D or 2D embedding vector for the question.
            result: Result dictionary to cache.
        """
        self._cache[query] = {
            "embedding": np.asarray(embedding),
            "result": result,
        }
        logger.debug("Stored query in semantic cache (total entries: %d)", len(self._cache))

    def clear(self) -> None:
        """Clear all entries from the semantic cache."""
        count = len(self._cache)
        self._cache.clear()
        logger.info("Cleared semantic cache (%d entries removed)", count)

    def __len__(self) -> int:
        return len(self._cache)
