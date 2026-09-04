"""
BM25 Retriever — sparse keyword-based retrieval using BM25Okapi.

Scores are normalized to [0, 1] for fusion compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    """A chunk with its retrieval score."""

    chunk: Chunk
    score: float
    source: str = "bm25"  # which retriever produced this

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class BM25Retriever:
    """
    BM25-based sparse retrieval.

    Tokenizes the query by whitespace + lowercasing (matching the index).
    Scores are min-max normalized to [0, 1].
    """

    def __init__(self, bm25_index, chunks: list[Chunk]):
        self.bm25 = bm25_index
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of ScoredChunk sorted by relevance (highest first).
        """
        tokenized_query = query.lower().split()
        raw_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        scores = self._normalize_scores(raw_scores)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    ScoredChunk(
                        chunk=self.chunks[idx],
                        score=float(scores[idx]),
                        source="bm25",
                    )
                )

        logger.debug("BM25 retrieved %d chunks for query: %s...", len(results), query[:50])
        return results

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)
