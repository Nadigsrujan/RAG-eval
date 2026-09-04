"""
Retrieval Pipeline — orchestrates hybrid retrieval + reranking.

Wires together BM25 → Dense → Fusion → Reranker into a single
configurable pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ingestion.embedder import Embedder
from ingestion.indexer import IndexBundle
from retrieval.bm25_retriever import BM25Retriever, ScoredChunk
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

# Re-export ScoredChunk from the canonical location
__all__ = ["RetrievalPipeline", "ScoredChunk"]


@dataclass
class RetrievalResult:
    """Full retrieval result with diagnostics."""

    chunks: list[ScoredChunk]
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline.

    Supports multiple configurations:
      - BM25 only
      - Dense only
      - Hybrid (BM25 + Dense)
      - Hybrid + Reranker

    Args:
        index_bundle: Pre-built IndexBundle from the ingestion stage.
        config: Pipeline configuration dictionary.
        embedder: Embedder instance (reused from ingestion).
    """

    def __init__(self, index_bundle: IndexBundle, config: dict, embedder: Embedder):
        retrieval_cfg = config.get("retrieval", {})
        reranker_cfg = config.get("reranker", {})

        self.config = config
        self.embedder = embedder

        # Build component retrievers
        self.bm25_retriever = BM25Retriever(index_bundle.bm25_index, index_bundle.chunks)
        self.dense_retriever = DenseRetriever(
            index_bundle.faiss_index, index_bundle.chunks, embedder
        )

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            fusion_method=retrieval_cfg.get("fusion_method", "rrf"),
            weights=retrieval_cfg.get("fusion_weights", {"bm25": 0.4, "dense": 0.6}),
        )

        # Reranker (optional)
        self.reranker = None
        if reranker_cfg.get("enabled", True):
            self.reranker = Reranker(
                model_name=reranker_cfg.get(
                    "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
            )

        self.top_k_initial = retrieval_cfg.get("top_k_initial", 20)
        self.top_k_final = reranker_cfg.get("top_k_final", 5)
        self.bm25_enabled = retrieval_cfg.get("bm25_enabled", True)
        self.dense_enabled = retrieval_cfg.get("dense_enabled", True)

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Run the full retrieval pipeline for a query.

        Pipeline:
            Query → [BM25] → [Dense] → Fusion → [Reranker] → Top-K

        Returns:
            RetrievalResult with scored chunks and timing diagnostics.
        """
        timings = {}
        total_start = time.time()

        # Step 1: Initial retrieval
        t0 = time.time()
        if self.bm25_enabled and self.dense_enabled:
            candidates = self.hybrid_retriever.retrieve(query, top_k=self.top_k_initial)
            retrieval_mode = "hybrid"
        elif self.bm25_enabled:
            candidates = self.bm25_retriever.retrieve(query, top_k=self.top_k_initial)
            retrieval_mode = "bm25_only"
        elif self.dense_enabled:
            candidates = self.dense_retriever.retrieve(query, top_k=self.top_k_initial)
            retrieval_mode = "dense_only"
        else:
            raise ValueError("At least one retriever (BM25 or Dense) must be enabled")
        timings["retrieval_ms"] = (time.time() - t0) * 1000

        # Step 2: Reranking (if enabled)
        if self.reranker and candidates:
            t0 = time.time()
            results = self.reranker.rerank(query, candidates, top_k=self.top_k_final)
            timings["reranking_ms"] = (time.time() - t0) * 1000
        else:
            results = candidates[: self.top_k_final]
            timings["reranking_ms"] = 0.0

        timings["total_ms"] = (time.time() - total_start) * 1000

        logger.info(
            "Retrieved %d chunks (%s, reranker=%s) in %.0fms",
            len(results),
            retrieval_mode,
            "on" if self.reranker else "off",
            timings["total_ms"],
        )

        return RetrievalResult(
            chunks=results,
            timings=timings,
            metadata={
                "mode": retrieval_mode,
                "reranker_enabled": self.reranker is not None,
                "initial_candidates": len(candidates),
                "final_results": len(results),
            },
        )
