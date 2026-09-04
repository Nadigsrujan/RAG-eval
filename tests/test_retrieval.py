"""Tests for the retrieval components — BM25, dense, hybrid, reranker."""

import numpy as np
import pytest

from ingestion.chunker import Chunk
from retrieval.bm25_retriever import BM25Retriever, ScoredChunk

# ── Test Fixtures ──────────────────────────────────────────────

def make_chunks(n: int = 5) -> list[Chunk]:
    """Create test chunks."""
    return [
        Chunk(
            chunk_id=f"chunk_{i}",
            text=f"This is test chunk number {i} about topic {chr(65 + i)}.",
            document="test.pdf",
            page=i + 1,
            start_char=i * 100,
            end_char=(i + 1) * 100,
        )
        for i in range(n)
    ]


# ── BM25 Retriever Tests ──────────────────────────────────────


class TestBM25Retriever:
    def test_basic_retrieval(self):
        from rank_bm25 import BM25Okapi

        chunks = make_chunks(5)
        tokenized = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)

        retriever = BM25Retriever(bm25, chunks)
        results = retriever.retrieve("chunk number 0", top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, ScoredChunk) for r in results)
        assert all(r.source == "bm25" for r in results)

    def test_score_normalization(self):
        scores = np.array([0.0, 5.0, 10.0])
        normalized = BM25Retriever._normalize_scores(scores)
        assert normalized[0] == pytest.approx(0.0)
        assert normalized[2] == pytest.approx(1.0)

    def test_zero_scores_normalization(self):
        scores = np.array([0.0, 0.0, 0.0])
        normalized = BM25Retriever._normalize_scores(scores)
        assert all(s == 0.0 for s in normalized)


# ── Hybrid Retriever Tests ─────────────────────────────────────


class TestHybridLogic:
    def test_rrf_fusion(self):
        """Test that RRF correctly fuses rankings."""
        from retrieval.hybrid import HybridRetriever

        chunks = make_chunks(3)

        # Simulate: BM25 ranks [0, 1, 2], Dense ranks [2, 0, 1]
        bm25_results = [
            ScoredChunk(chunk=chunks[0], score=0.9, source="bm25"),
            ScoredChunk(chunk=chunks[1], score=0.7, source="bm25"),
            ScoredChunk(chunk=chunks[2], score=0.5, source="bm25"),
        ]
        dense_results = [
            ScoredChunk(chunk=chunks[2], score=0.95, source="dense"),
            ScoredChunk(chunk=chunks[0], score=0.8, source="dense"),
            ScoredChunk(chunk=chunks[1], score=0.6, source="dense"),
        ]

        # Create a mock HybridRetriever and call RRF directly
        hybrid = HybridRetriever.__new__(HybridRetriever)
        fused = hybrid._reciprocal_rank_fusion(bm25_results, dense_results, k=60)

        assert len(fused) == 3

        # chunk_0 is rank 1 in BM25, rank 2 in dense → highest RRF
        fused.sort(key=lambda x: x.score, reverse=True)
        # All chunks should be present
        chunk_ids = {f.chunk.chunk_id for f in fused}
        assert chunk_ids == {"chunk_0", "chunk_1", "chunk_2"}
