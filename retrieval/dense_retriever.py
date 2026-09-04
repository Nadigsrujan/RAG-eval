"""
Dense Retriever — FAISS-based dense vector retrieval.

Uses inner product (equivalent to cosine similarity for L2-normalized vectors).
"""

from __future__ import annotations

import logging

import numpy as np

from ingestion.chunker import Chunk
from retrieval.bm25_retriever import ScoredChunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense retrieval using FAISS index and SentenceTransformer embeddings.

    Since embeddings are L2-normalized at indexing time, FAISS inner product
    scores are equivalent to cosine similarity and already in [-1, 1].
    We shift them to [0, 1] for fusion.
    """

    def __init__(self, faiss_index, chunks: list[Chunk], embedder):
        self.index = faiss_index
        self.chunks = chunks
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """
        Retrieve the top-k most similar chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of ScoredChunk sorted by cosine similarity (highest first).
        """
        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            # Inner product of normalized vectors is in [-1, 1]
            # Shift to [0, 1] for consistency with BM25 scores
            normalized_score = (float(score) + 1.0) / 2.0
            results.append(
                ScoredChunk(
                    chunk=self.chunks[idx],
                    score=normalized_score,
                    source="dense",
                )
            )

        logger.debug("Dense retrieved %d chunks for query: %s...", len(results), query[:50])
        return results

    def retrieve_by_embedding(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[ScoredChunk]:
        """Retrieve using a pre-computed query embedding."""
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            normalized_score = (float(score) + 1.0) / 2.0
            results.append(
                ScoredChunk(
                    chunk=self.chunks[idx],
                    score=normalized_score,
                    source="dense",
                )
            )
        return results
