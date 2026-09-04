"""
Generation Evaluator — measures answer quality from the LLM.

Metrics:
  - Exact Match: strict string equality
  - Fuzzy Match: token overlap (F1)
  - Contains Match: ground truth substring in answer
  - Answer Relevance: does it address the question? (heuristic)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Aggregated generation evaluation metrics."""

    exact_match: float = 0.0
    fuzzy_f1: float = 0.0
    contains_match: float = 0.0
    avg_answer_length: float = 0.0
    num_queries: int = 0

    def to_dict(self) -> dict:
        return {
            "exact_match": round(self.exact_match, 4),
            "fuzzy_f1": round(self.fuzzy_f1, 4),
            "contains_match": round(self.contains_match, 4),
            "avg_answer_length": round(self.avg_answer_length, 1),
            "num_queries": self.num_queries,
        }


class GenerationEvaluator:
    """
    Evaluates LLM-generated answers against ground truth.

    These are automated metrics (no LLM judge needed).
    For LLM-based evaluation, see llm_judge.py.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison: lowercase, strip punctuation."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        """Check if normalized prediction equals normalized ground truth."""
        return GenerationEvaluator.normalize(prediction) == GenerationEvaluator.normalize(
            ground_truth
        )

    @staticmethod
    def contains_match(prediction: str, ground_truth: str) -> bool:
        """Check if the ground truth is contained in the prediction."""
        return GenerationEvaluator.normalize(ground_truth) in GenerationEvaluator.normalize(
            prediction
        )

    @staticmethod
    def token_f1(prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score between prediction and ground truth.

        This is the standard SQuAD-style F1 metric.
        """
        pred_tokens = GenerationEvaluator.normalize(prediction).split()
        truth_tokens = GenerationEvaluator.normalize(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return float(pred_tokens == truth_tokens)

        pred_counter = Counter(pred_tokens)
        truth_counter = Counter(truth_tokens)

        common = sum((pred_counter & truth_counter).values())

        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def evaluate(
        self,
        eval_data: list[dict],
        generation_fn,
    ) -> GenerationMetrics:
        """
        Run evaluation across all queries.

        Args:
            eval_data: List of {"question": str, "answer": str}
            generation_fn: Callable(question) -> predicted_answer

        Returns:
            Aggregated GenerationMetrics.
        """
        exact_matches = []
        f1_scores = []
        contains_matches = []
        answer_lengths = []

        for item in eval_data:
            question = item["question"]
            ground_truth = item["answer"]

            prediction = generation_fn(question)

            exact_matches.append(float(self.exact_match(prediction, ground_truth)))
            f1_scores.append(self.token_f1(prediction, ground_truth))
            contains_matches.append(float(self.contains_match(prediction, ground_truth)))
            answer_lengths.append(len(prediction.split()))

        n = len(exact_matches)
        if n == 0:
            return GenerationMetrics()

        metrics = GenerationMetrics(
            exact_match=sum(exact_matches) / n,
            fuzzy_f1=sum(f1_scores) / n,
            contains_match=sum(contains_matches) / n,
            avg_answer_length=sum(answer_lengths) / n,
            num_queries=n,
        )

        logger.info(
            "Generation eval: %d queries — EM=%.3f, F1=%.3f, Contains=%.3f",
            n,
            metrics.exact_match,
            metrics.fuzzy_f1,
            metrics.contains_match,
        )
        return metrics
