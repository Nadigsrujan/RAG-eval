"""
Retrieval Evaluator — measures how well the retrieval pipeline finds relevant evidence.

Metrics:
  - Recall@K: fraction of relevant chunks found in top-K results
  - MRR (Mean Reciprocal Rank): average 1/rank of first relevant result
  - NDCG@K: normalized discounted cumulative gain
  - Precision@K: fraction of top-K results that are relevant
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval evaluation metrics."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    num_queries: int = 0

    def to_dict(self) -> dict:
        return {
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "precision@5": round(self.precision_at_5, 4),
            "precision@10": round(self.precision_at_10, 4),
            "num_queries": self.num_queries,
        }


class RetrievalEvaluator:
    """
    Evaluates retrieval quality against ground truth relevant chunks.

    Usage:
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate(eval_data, retrieval_pipeline)
    """

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Fraction of relevant items found in top-K retrieved."""
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Fraction of top-K retrieved items that are relevant."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for rid in top_k if rid in relevant_ids)
        return hits / len(top_k)

    @staticmethod
    def reciprocal_rank(
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:
        """1/rank of the first relevant result (0 if none found)."""
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.

        Measures ranking quality — relevant items at higher ranks get more credit.
        """
        top_k = retrieved_ids[:k]

        # DCG: sum of 1/log2(rank+1) for relevant items
        dcg = 0.0
        for i, rid in enumerate(top_k):
            if rid in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)  # +2 because rank is 1-indexed

        # Ideal DCG: all relevant items at the top
        ideal_length = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_length))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def evaluate(
        self,
        eval_data: list[dict],
        retrieval_fn,
    ) -> RetrievalMetrics:
        """
        Run evaluation across all queries in the eval dataset.

        Args:
            eval_data: List of {"question": str, "relevant_chunks": [str, ...]}
            retrieval_fn: Callable(question) -> list of chunk_ids

        Returns:
            Aggregated RetrievalMetrics.
        """
        recalls_1, recalls_5, recalls_10 = [], [], []
        mrrs = []
        ndcgs_5, ndcgs_10 = [], []
        precs_5, precs_10 = [], []

        for item in eval_data:
            question = item["question"]
            relevant = set(item.get("relevant_chunks", []))

            if not relevant:
                continue

            retrieved = retrieval_fn(question)

            recalls_1.append(self.recall_at_k(retrieved, relevant, 1))
            recalls_5.append(self.recall_at_k(retrieved, relevant, 5))
            recalls_10.append(self.recall_at_k(retrieved, relevant, 10))
            mrrs.append(self.reciprocal_rank(retrieved, relevant))
            ndcgs_5.append(self.ndcg_at_k(retrieved, relevant, 5))
            ndcgs_10.append(self.ndcg_at_k(retrieved, relevant, 10))
            precs_5.append(self.precision_at_k(retrieved, relevant, 5))
            precs_10.append(self.precision_at_k(retrieved, relevant, 10))

        n = len(recalls_1)
        if n == 0:
            return RetrievalMetrics()

        metrics = RetrievalMetrics(
            recall_at_1=sum(recalls_1) / n,
            recall_at_5=sum(recalls_5) / n,
            recall_at_10=sum(recalls_10) / n,
            mrr=sum(mrrs) / n,
            ndcg_at_5=sum(ndcgs_5) / n,
            ndcg_at_10=sum(ndcgs_10) / n,
            precision_at_5=sum(precs_5) / n,
            precision_at_10=sum(precs_10) / n,
            num_queries=n,
        )

        logger.info(
            "Retrieval eval: %d queries — Recall@5=%.3f, MRR=%.3f, NDCG@5=%.3f",
            n,
            metrics.recall_at_5,
            metrics.mrr,
            metrics.ndcg_at_5,
        )
        return metrics
