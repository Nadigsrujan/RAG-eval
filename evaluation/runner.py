"""
Evaluation Runner — orchestrates all evaluation components and produces
a unified report.

Usage:
    python -m evaluation.runner --config configs/default.yaml
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.retrieval_eval import RetrievalEvaluator, RetrievalMetrics
from evaluation.generation_eval import GenerationEvaluator, GenerationMetrics
from evaluation.citation_eval import CitationEvaluator, CitationMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Complete evaluation report across all dimensions."""

    retrieval: RetrievalMetrics | None = None
    generation: GenerationMetrics | None = None
    citation: CitationMetrics | None = None
    config_snapshot: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        report = {
            "timestamp": self.timestamp,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "config": self.config_snapshot,
        }
        if self.retrieval:
            report["retrieval"] = self.retrieval.to_dict()
        if self.generation:
            report["generation"] = self.generation.to_dict()
        if self.citation:
            report["citation"] = self.citation.to_dict()
        return report

    def to_markdown(self) -> str:
        """Format the report as a readable markdown string."""
        lines = ["# RAG Evaluation Report", ""]

        if self.retrieval:
            r = self.retrieval
            lines.extend([
                "## Retrieval Metrics",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Recall@1 | {r.recall_at_1:.4f} |",
                f"| Recall@5 | {r.recall_at_5:.4f} |",
                f"| Recall@10 | {r.recall_at_10:.4f} |",
                f"| MRR | {r.mrr:.4f} |",
                f"| NDCG@5 | {r.ndcg_at_5:.4f} |",
                f"| Precision@5 | {r.precision_at_5:.4f} |",
                f"| Queries | {r.num_queries} |",
                "",
            ])

        if self.generation:
            g = self.generation
            lines.extend([
                "## Generation Metrics",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Exact Match | {g.exact_match:.4f} |",
                f"| Token F1 | {g.fuzzy_f1:.4f} |",
                f"| Contains Match | {g.contains_match:.4f} |",
                f"| Avg Answer Length | {g.avg_answer_length:.1f} |",
                "",
            ])

        if self.citation:
            c = self.citation
            lines.extend([
                "## Citation Metrics",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Precision | {c.citation_precision:.4f} |",
                f"| Recall | {c.citation_recall:.4f} |",
                f"| F1 | {c.citation_f1:.4f} |",
                "",
            ])

        lines.append(f"\n*Evaluated in {self.elapsed_seconds:.1f}s*")
        return "\n".join(lines)


class EvalRunner:
    """
    Orchestrates the full evaluation pipeline.

    Loads the eval dataset, runs retrieval + generation + citation
    evaluation, and produces a unified report.
    """

    def __init__(self, config: dict):
        self.config = config
        self.retrieval_eval = RetrievalEvaluator()
        self.generation_eval = GenerationEvaluator()
        self.citation_eval = CitationEvaluator()

    def load_dataset(self, path: str | Path) -> list[dict]:
        """Load a JSONL evaluation dataset."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Eval dataset not found: {path}")

        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        logger.info("Loaded %d evaluation items from %s", len(data), path)
        return data

    def run(
        self,
        eval_data: list[dict],
        retrieval_fn=None,
        generation_fn=None,
        citation_fn=None,
    ) -> EvalReport:
        """
        Run all applicable evaluations.

        Args:
            eval_data: List of evaluation items (JSONL records).
            retrieval_fn: Optional callable for retrieval evaluation.
            generation_fn: Optional callable for generation evaluation.
            citation_fn: Optional callable for citation evaluation.

        Returns:
            EvalReport with all computed metrics.
        """
        from datetime import datetime, timezone

        start = time.time()
        report = EvalReport(
            config_snapshot=self.config,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if retrieval_fn:
            logger.info("Running retrieval evaluation...")
            report.retrieval = self.retrieval_eval.evaluate(eval_data, retrieval_fn)

        if generation_fn:
            logger.info("Running generation evaluation...")
            report.generation = self.generation_eval.evaluate(eval_data, generation_fn)

        if citation_fn:
            logger.info("Running citation evaluation...")
            report.citation = self.citation_eval.evaluate(eval_data, citation_fn)

        report.elapsed_seconds = time.time() - start

        logger.info("Evaluation complete in %.1fs", report.elapsed_seconds)
        return report

    def save_report(self, report: EvalReport, output_dir: str | Path) -> Path:
        """Save evaluation report as JSON and Markdown."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = output_dir / "eval_report.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Markdown report
        md_path = output_dir / "eval_report.md"
        with open(md_path, "w") as f:
            f.write(report.to_markdown())

        logger.info("Reports saved to %s", output_dir)
        return json_path


# ── CLI Entry Point ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--dataset", default=None, help="Eval dataset path (overrides config)")
    parser.add_argument("--output", default="experiments/results/", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_path = args.dataset or config.get("evaluation", {}).get("dataset", "")
    runner = EvalRunner(config)
    eval_data = runner.load_dataset(dataset_path)

    # Note: In a real run, you'd wire in the actual pipeline functions here
    report = runner.run(eval_data)
    runner.save_report(report, args.output)

    print(report.to_markdown())
