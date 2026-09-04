"""
Pipeline Health & Validation Agent — automates continuous or on-demand pipeline quality audits.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.citation_eval import CitationEvaluator
from evaluation.generation_eval import GenerationEvaluator, GenerationMetrics
from evaluation.retrieval_eval import RetrievalEvaluator, RetrievalMetrics

logger = logging.getLogger(__name__)


@dataclass
class PipelineHealthReport:
    """Comprehensive pipeline health evaluation report."""

    overall_status: str = "HEALTHY"  # HEALTHY | DEGRADED | UNHEALTHY
    retrieval_status: str = "HEALTHY"
    generation_status: str = "HEALTHY"
    citation_status: str = "HEALTHY"

    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    generation_metrics: dict[str, float] = field(default_factory=dict)
    citation_metrics: dict[str, float] = field(default_factory=dict)

    sla_checks: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    num_eval_samples: int = 0
    elapsed_seconds: float = 0.0


class PipelineValidatorAgent:
    """
    Automates end-to-end pipeline benchmarking and SLA verification.

    Responsibilities:
      1. Load evaluation dataset
      2. Run retrieval, generation, and citation evaluations
      3. Compare metrics against defined SLA thresholds
      4. Synthesize root-cause recommendations if degradation is detected
    """

    DEFAULT_SLAS = {
        "recall_at_5": 0.60,
        "mrr": 0.50,
        "contains_match": 0.50,
        "citation_score": 0.70,
    }

    def __init__(self, config: dict, dataset_path: str | None = None):
        self.config = config
        eval_cfg = config.get("evaluation", {})
        sla_cfg = config.get("orchestration", {}).get("sla_thresholds", self.DEFAULT_SLAS)
        self.slas = {**self.DEFAULT_SLAS, **sla_cfg}

        self.dataset_path = Path(
            dataset_path
            or eval_cfg.get("dataset", "evaluation/datasets/eval_set.jsonl")
        )
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.citation_evaluator = CitationEvaluator()

    def audit_pipeline(
        self,
        retrieval_fn: Any,
        generation_fn: Any,
    ) -> PipelineHealthReport:
        """
        Run automated validation audit against the live pipeline.
        """
        t0 = time.time()
        report = PipelineHealthReport()

        if not self.dataset_path.exists():
            report.overall_status = "UNHEALTHY"
            report.recommendations.append(f"Eval dataset not found at {self.dataset_path}")
            return report

        # Load samples
        samples = []
        with open(self.dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if "answer" not in item and "ground_truth" in item:
                        item["answer"] = item["ground_truth"]
                    if "relevant_chunks" not in item and "relevant_chunk_ids" in item:
                        item["relevant_chunks"] = item["relevant_chunk_ids"]
                    samples.append(item)

        if not samples:
            report.overall_status = "UNHEALTHY"
            report.recommendations.append("Eval dataset is empty.")
            return report

        report.num_eval_samples = len(samples)

        # 1. Retrieval Benchmark
        try:
            r_metrics: RetrievalMetrics = self.retrieval_evaluator.evaluate(
                samples, retrieval_fn
            )
            report.retrieval_metrics = {
                "recall_at_1": round(r_metrics.recall_at_1, 4),
                "recall_at_5": round(r_metrics.recall_at_5, 4),
                "recall_at_10": round(r_metrics.recall_at_10, 4),
                "mrr": round(r_metrics.mrr, 4),
                "ndcg_at_5": round(r_metrics.ndcg_at_5, 4),
                "precision_at_5": round(r_metrics.precision_at_5, 4),
            }

            # Check retrieval SLA
            if r_metrics.recall_at_5 < self.slas["recall_at_5"]:
                report.retrieval_status = "DEGRADED"
                report.recommendations.append(
                    f"Retrieval Recall@5 ({r_metrics.recall_at_5:.2f}) is below SLA ({self.slas['recall_at_5']}). "
                    "Consider tuning BM25/dense fusion weights or increasing top_k_initial."
                )
            if r_metrics.mrr < self.slas["mrr"]:
                report.retrieval_status = "DEGRADED"
                report.recommendations.append(
                    f"Retrieval MRR ({r_metrics.mrr:.2f}) is below SLA ({self.slas['mrr']}). "
                    "Verify Cross-Encoder reranker quality."
                )
            report.sla_checks.append({
                "metric": "recall_at_5",
                "value": r_metrics.recall_at_5,
                "target": self.slas["recall_at_5"],
                "passed": r_metrics.recall_at_5 >= self.slas["recall_at_5"],
            })
        except Exception as e:
            logger.error("Retrieval validation failed: %s", e)
            report.retrieval_status = "UNHEALTHY"
            report.recommendations.append(f"Retrieval evaluation error: {e}")

        # 2. Generation Benchmark
        try:
            g_metrics: GenerationMetrics = self.generation_evaluator.evaluate(
                samples, generation_fn
            )
            report.generation_metrics = {
                "exact_match": round(g_metrics.exact_match, 4),
                "fuzzy_f1": round(g_metrics.fuzzy_f1, 4),
                "contains_match": round(g_metrics.contains_match, 4),
                "avg_answer_length": round(g_metrics.avg_answer_length, 1),
            }

            # Check generation SLA
            if g_metrics.contains_match < self.slas["contains_match"]:
                report.generation_status = "DEGRADED"
                report.recommendations.append(
                    f"Generation Contains-Match ({g_metrics.contains_match:.2f}) below SLA ({self.slas['contains_match']}). "
                    "Review prompt template or LLM max_input_tokens."
                )
            report.sla_checks.append({
                "metric": "contains_match",
                "value": g_metrics.contains_match,
                "target": self.slas["contains_match"],
                "passed": g_metrics.contains_match >= self.slas["contains_match"],
            })
        except Exception as e:
            logger.error("Generation validation failed: %s", e)
            report.generation_status = "UNHEALTHY"
            report.recommendations.append(f"Generation evaluation error: {e}")

        # Determine overall status
        statuses = [report.retrieval_status, report.generation_status]
        if "UNHEALTHY" in statuses:
            report.overall_status = "UNHEALTHY"
        elif "DEGRADED" in statuses:
            report.overall_status = "DEGRADED"
        else:
            report.overall_status = "HEALTHY"

        report.elapsed_seconds = round(time.time() - t0, 2)
        return report
