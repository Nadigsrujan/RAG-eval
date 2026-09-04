"""
Citation Evaluator — measures how accurately the system cites its sources.

Metrics:
  - Citation Precision: fraction of generated citations that are correct
  - Citation Recall: fraction of expected citations that were generated
  - Citation F1: harmonic mean of precision and recall
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CitationMetrics:
    """Aggregated citation evaluation metrics."""

    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    avg_citations_per_answer: float = 0.0
    num_queries: int = 0

    def to_dict(self) -> dict:
        return {
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "citation_f1": round(self.citation_f1, 4),
            "avg_citations_per_answer": round(self.avg_citations_per_answer, 2),
            "num_queries": self.num_queries,
        }


class CitationEvaluator:
    """
    Evaluates citation quality by comparing generated citations against
    expected citations from the ground truth.

    A citation is represented as (document, page) tuple.
    """

    @staticmethod
    def citation_precision(
        generated: set[tuple[str, int]],
        expected: set[tuple[str, int]],
    ) -> float:
        """Fraction of generated citations that are correct."""
        if not generated:
            return 0.0
        return len(generated & expected) / len(generated)

    @staticmethod
    def citation_recall(
        generated: set[tuple[str, int]],
        expected: set[tuple[str, int]],
    ) -> float:
        """Fraction of expected citations that were generated."""
        if not expected:
            return 0.0
        return len(generated & expected) / len(expected)

    @staticmethod
    def citation_f1(precision: float, recall: float) -> float:
        """Harmonic mean of citation precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def evaluate(
        self,
        eval_data: list[dict],
        citation_fn,
    ) -> CitationMetrics:
        """
        Run citation evaluation across all queries.

        Args:
            eval_data: List of dicts with "expected_citations": [{"document": str, "page": int}]
            citation_fn: Callable(question) -> [{"document": str, "page": int}]

        Returns:
            Aggregated CitationMetrics.
        """
        precisions = []
        recalls = []
        f1s = []
        citation_counts = []

        for item in eval_data:
            question = item["question"]
            expected = set()
            for c in item.get("expected_citations", []):
                expected.add((c["document"], c["page"]))

            if not expected:
                continue

            generated_raw = citation_fn(question)
            generated = set()
            for c in generated_raw:
                generated.add((c["document"], c["page"]))

            p = self.citation_precision(generated, expected)
            r = self.citation_recall(generated, expected)
            f1 = self.citation_f1(p, r)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            citation_counts.append(len(generated))

        n = len(precisions)
        if n == 0:
            return CitationMetrics()

        metrics = CitationMetrics(
            citation_precision=sum(precisions) / n,
            citation_recall=sum(recalls) / n,
            citation_f1=sum(f1s) / n,
            avg_citations_per_answer=sum(citation_counts) / n,
            num_queries=n,
        )

        logger.info(
            "Citation eval: %d queries — P=%.3f, R=%.3f, F1=%.3f",
            n,
            metrics.citation_precision,
            metrics.citation_recall,
            metrics.citation_f1,
        )
        return metrics
