"""
Hybrid Retriever — fuses BM25 and Dense retrieval results.

Supports two fusion strategies:
  - Reciprocal Rank Fusion (RRF): rank-based, parameter-free
  - Weighted: linear combination of normalized scores
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Literal

from retrieval.bm25_retriever import BM25Retriever, ScoredChunk
from retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 (sparse) and Dense (vector) retrieval with score fusion.

    Args:
        bm25_retriever: BM25 retriever instance.
        dense_retriever: Dense retriever instance.
        fusion_method: 'rrf' for Reciprocal Rank Fusion, 'weighted' for
                       linear score combination.
        weights: Dict with 'bm25' and 'dense' weights (used only for
                 'weighted' fusion). Must sum to 1.0.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        weights: dict[str, float] | None = None,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.fusion_method = fusion_method
        self.weights = weights or {"bm25": 0.4, "dense": 0.6}

    def retrieve(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """
        Retrieve chunks using hybrid BM25 + Dense fusion.

        Args:
            query: The search query.
            top_k: Number of final results to return.

        Returns:
            List of ScoredChunk sorted by fused score.
        """
        # Get candidates from both retrievers (fetch more than top_k for better fusion)
        fetch_k = top_k * 2
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)
        dense_results = self.dense.retrieve(query, top_k=fetch_k)

        if self.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(bm25_results, dense_results)
        else:
            fused = self._weighted_fusion(bm25_results, dense_results)

        # Sort by fused score and take top_k
        fused.sort(key=lambda x: x.score, reverse=True)
        results = fused[:top_k]

        logger.debug(
            "Hybrid retrieved %d chunks (BM25=%d, Dense=%d, fused=%d)",
            len(results),
            len(bm25_results),
            len(dense_results),
            len(fused),
        )
        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[ScoredChunk],
        dense_results: list[ScoredChunk],
        k: int = 60,
    ) -> list[ScoredChunk]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF(d) = Σ  1 / (k + rank_i(d))

        where k is a smoothing constant (typically 60) and rank_i(d) is the
        rank of document d in retriever i's result list.

        RRF is effective because it's parameter-free (aside from k) and
        robust to score distribution differences between retrievers.
        """
        chunk_map: dict[str, ScoredChunk] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        # BM25 rankings
        for rank, result in enumerate(bm25_results):
            cid = result.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank + 1)
            chunk_map[cid] = result

        # Dense rankings
        for rank, result in enumerate(dense_results):
            cid = result.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = result

        # Build fused results
        fused = []
        for cid, score in rrf_scores.items():
            original = chunk_map[cid]
            fused.append(
                ScoredChunk(
                    chunk=original.chunk,
                    score=score,
                    source="hybrid_rrf",
                )
            )
        return fused

    def _weighted_fusion(
        self,
        bm25_results: list[ScoredChunk],
        dense_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """
        Weighted linear combination of BM25 and Dense scores.

        fused_score(d) = w_bm25 * score_bm25(d) + w_dense * score_dense(d)
        """
        w_bm25 = self.weights["bm25"]
        w_dense = self.weights["dense"]

        chunk_map: dict[str, ScoredChunk] = {}
        scores: dict[str, float] = defaultdict(float)

        for result in bm25_results:
            cid = result.chunk_id
            scores[cid] += w_bm25 * result.score
            chunk_map[cid] = result

        for result in dense_results:
            cid = result.chunk_id
            scores[cid] += w_dense * result.score
            if cid not in chunk_map:
                chunk_map[cid] = result

        fused = []
        for cid, score in scores.items():
            original = chunk_map[cid]
            fused.append(
                ScoredChunk(
                    chunk=original.chunk,
                    score=score,
                    source="hybrid_weighted",
                )
            )
        return fused
