"""
Reranker — cross-encoder reranking for fine-grained relevance scoring.

Unlike bi-encoder retrieval (which embeds query and document independently),
the cross-encoder sees [query + document] together, giving it much more
information for relevance judgment — at the cost of speed.

That's why we use it as a second stage on the top-K candidates.
"""

from __future__ import annotations

import logging
from typing import Optional

from retrieval.bm25_retriever import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker for refining retrieval results.

    Args:
        model_name: HuggingFace cross-encoder model identifier.
        device: Device to run on. None = auto-detect.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy-load the CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self._device)
            logger.info("Reranker loaded")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """
        Rerank candidate chunks using the cross-encoder.

        The cross-encoder scores each (query, chunk) pair jointly:

            [CLS] query [SEP] chunk_text [SEP]
                           ↓
                     Transformer
                           ↓
                    relevance score

        Args:
            query: The search query.
            candidates: Retrieved candidate chunks to rerank.
            top_k: Number of top results to return after reranking.

        Returns:
            List of ScoredChunk sorted by cross-encoder score.
        """
        if not candidates:
            return []

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, candidate.chunk.text) for candidate in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        reranked = []
        for candidate, score in zip(candidates, scores):
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    source="reranked",
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        results = reranked[:top_k]

        logger.debug(
            "Reranked %d → %d candidates (top score=%.4f, bottom=%.4f)",
            len(candidates),
            len(results),
            results[0].score if results else 0,
            results[-1].score if results else 0,
        )
        return results
