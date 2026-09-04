"""
Experiment Tracker — logs parameters, metrics, and artifacts for each
experiment run.

Supports two backends:
  - JSON file (lightweight, zero dependencies)
  - MLflow (full UI, model registry)

Usage:
    tracker = ExperimentTracker(backend="json", results_dir="experiments/results/")
    run = tracker.start_run("hybrid_reranker_v2")
    run.log_params({"top_k": 20, "reranker": True, "chunk_size": 300})
    run.log_metrics({"recall@5": 0.89, "mrr": 0.82, "faithfulness": 0.91})
    run.end()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """A single experiment run with parameters, metrics, and artifacts."""

    run_id: str
    experiment_name: str
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    status: str = "running"  # running | completed | failed
    notes: str = ""

    def log_params(self, params: dict) -> None:
        """Log experiment parameters."""
        self.params.update(params)

    def log_metrics(self, metrics: dict) -> None:
        """Log experiment metrics."""
        self.metrics.update(metrics)

    def log_artifact(self, path: str) -> None:
        """Log an artifact path."""
        self.artifacts.append(path)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "notes": self.notes,
        }


class ExperimentTracker:
    """
    Tracks experiments with configurable backend.

    Args:
        backend: 'json' for file-based tracking, 'mlflow' for MLflow.
        results_dir: Directory for JSON backend results.
    """

    def __init__(
        self,
        backend: Literal["json", "mlflow"] = "json",
        results_dir: str | Path = "experiments/results/",
    ):
        self.backend = backend
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._active_run: Optional[ExperimentRun] = None

    def start_run(self, experiment_name: str, notes: str = "") -> ExperimentRun:
        """Start a new experiment run."""
        now = datetime.now(timezone.utc)
        run_id = f"{experiment_name}_{now.strftime('%Y%m%d_%H%M%S')}"

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            start_time=now.isoformat(),
            notes=notes,
        )

        self._active_run = run
        logger.info("Started experiment run: %s", run_id)

        if self.backend == "mlflow":
            self._start_mlflow_run(experiment_name, run_id)

        return run

    def end_run(self, run: ExperimentRun, status: str = "completed") -> None:
        """End an experiment run and persist results."""
        run.end_time = datetime.now(timezone.utc).isoformat()
        run.status = status

        if self.backend == "json":
            self._save_json(run)
        elif self.backend == "mlflow":
            self._end_mlflow_run(run)

        logger.info(
            "Experiment %s %s — metrics: %s",
            run.run_id,
            status,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in run.metrics.items()},
        )
        self._active_run = None

    def list_runs(self, experiment_name: Optional[str] = None) -> list[dict]:
        """List all recorded experiment runs."""
        runs = []
        for f in sorted(self.results_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    run = json.load(fh)
                if experiment_name and run.get("experiment_name") != experiment_name:
                    continue
                runs.append(run)
            except Exception:
                continue
        return runs

    def compare_runs(self, run_ids: list[str]) -> list[dict]:
        """Load specific runs for comparison."""
        runs = self.list_runs()
        return [r for r in runs if r["run_id"] in run_ids]

    # ── JSON Backend ───────────────────────────────────────────

    def _save_json(self, run: ExperimentRun) -> None:
        """Save run to a JSON file."""
        path = self.results_dir / f"{run.run_id}.json"
        with open(path, "w") as f:
            json.dump(run.to_dict(), f, indent=2)

    # ── MLflow Backend ─────────────────────────────────────────

    def _start_mlflow_run(self, experiment_name: str, run_id: str) -> None:
        """Start an MLflow run."""
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_id)
        except ImportError:
            logger.warning("MLflow not installed, falling back to JSON")
            self.backend = "json"

    def _end_mlflow_run(self, run: ExperimentRun) -> None:
        """End an MLflow run with logged params and metrics."""
        try:
            import mlflow

            mlflow.log_params(run.params)
            mlflow.log_metrics(run.metrics)
            mlflow.end_run()
        except Exception as e:
            logger.warning("MLflow logging failed: %s, saving as JSON", e)
            self._save_json(run)
