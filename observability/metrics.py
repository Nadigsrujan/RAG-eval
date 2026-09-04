"""
Metrics Collector — tracks pipeline performance metrics in-memory
with periodic flush to JSON.

Tracks p50/p95/p99 latencies, token counts, error rates,
and per-stage timing breakdowns.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricsSummary:
    """Aggregated metrics summary."""

    total_requests: int = 0
    total_errors: int = 0
    error_rate: float = 0.0

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0

    stage_latencies: dict[str, dict[str, float]] = field(default_factory=dict)

    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    avg_citation_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "latency": {
                "p50_ms": round(self.latency_p50_ms, 1),
                "p95_ms": round(self.latency_p95_ms, 1),
                "p99_ms": round(self.latency_p99_ms, 1),
                "avg_ms": round(self.latency_avg_ms, 1),
            },
            "stage_latencies": self.stage_latencies,
            "tokens": {
                "avg_input": round(self.avg_input_tokens, 1),
                "avg_output": round(self.avg_output_tokens, 1),
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
            },
            "avg_citation_score": round(self.avg_citation_score, 4),
        }


class MetricsCollector:
    """
    In-memory metrics collector with percentile computation and JSON export.

    Usage:
        collector = MetricsCollector()

        collector.record_request(
            total_ms=1500,
            stage_timings={"retrieval": 80, "reranking": 200, "generation": 1100},
            input_tokens=1200,
            output_tokens=150,
            citation_score=0.85,
        )

        summary = collector.summarize()
    """

    def __init__(self, log_dir: Optional[str | Path] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self._total_latencies: list[float] = []
        self._stage_latencies: dict[str, list[float]] = defaultdict(list)
        self._input_tokens: list[int] = []
        self._output_tokens: list[int] = []
        self._citation_scores: list[float] = []
        self._error_count: int = 0

    def record_request(
        self,
        total_ms: float,
        stage_timings: dict[str, float] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        citation_score: float = 0.0,
        is_error: bool = False,
    ) -> None:
        """Record metrics for a single request."""
        self._total_latencies.append(total_ms)

        if stage_timings:
            for stage, ms in stage_timings.items():
                self._stage_latencies[stage].append(ms)

        self._input_tokens.append(input_tokens)
        self._output_tokens.append(output_tokens)
        self._citation_scores.append(citation_score)

        if is_error:
            self._error_count += 1

    def summarize(self) -> MetricsSummary:
        """Compute aggregated metrics summary."""
        n = len(self._total_latencies)
        if n == 0:
            return MetricsSummary()

        sorted_latencies = sorted(self._total_latencies)

        # Stage-level percentiles
        stage_summaries = {}
        for stage, values in self._stage_latencies.items():
            if values:
                sv = sorted(values)
                stage_summaries[stage] = {
                    "p50_ms": round(self._percentile(sv, 50), 1),
                    "p95_ms": round(self._percentile(sv, 95), 1),
                    "avg_ms": round(statistics.mean(sv), 1),
                }

        return MetricsSummary(
            total_requests=n,
            total_errors=self._error_count,
            error_rate=self._error_count / n if n else 0,
            latency_p50_ms=self._percentile(sorted_latencies, 50),
            latency_p95_ms=self._percentile(sorted_latencies, 95),
            latency_p99_ms=self._percentile(sorted_latencies, 99),
            latency_avg_ms=statistics.mean(self._total_latencies),
            stage_latencies=stage_summaries,
            avg_input_tokens=statistics.mean(self._input_tokens) if self._input_tokens else 0,
            avg_output_tokens=statistics.mean(self._output_tokens) if self._output_tokens else 0,
            total_input_tokens=sum(self._input_tokens),
            total_output_tokens=sum(self._output_tokens),
            avg_citation_score=statistics.mean(self._citation_scores) if self._citation_scores else 0,
        )

    def flush(self) -> None:
        """Persist current metrics snapshot to disk."""
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summarize()
        path = self.log_dir / f"metrics_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info("Metrics flushed to %s", path)

    @staticmethod
    def _percentile(sorted_data: list[float], p: float) -> float:
        """Compute percentile from pre-sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        d = k - f
        return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])
