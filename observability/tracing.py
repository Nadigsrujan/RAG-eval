"""
Tracer — lightweight request-level tracing for the RAG pipeline.

Each request gets a unique trace_id. Each pipeline stage (retrieval,
reranking, generation, citation) becomes a Span within that trace.
Traces are persisted as JSON files for later analysis.

This is a local-first alternative to Langfuse/Jaeger.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single stage within a trace."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    status: str = "ok"  # ok | error
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "latency_ms": round(self.latency_ms, 2),
            "status": self.status,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class Trace:
    """A complete request trace with multiple spans."""

    trace_id: str
    question: str
    spans: list[Span] = field(default_factory=list)
    total_latency_ms: float = 0.0
    start_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }


class Tracer:
    """
    Creates and manages traces for RAG pipeline requests.

    Usage:
        tracer = Tracer(trace_dir="traces/")
        trace = tracer.start_trace("What is X?")

        with tracer.span(trace, "retrieval") as span:
            results = retriever.retrieve(query)
            span.metadata["num_results"] = len(results)

        tracer.end_trace(trace)
    """

    def __init__(self, trace_dir: str | Path = "traces/", enabled: bool = True):
        self.trace_dir = Path(trace_dir)
        self.enabled = enabled
        if enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def start_trace(self, question: str, metadata: dict | None = None) -> Trace:
        """Start a new trace for a request."""
        trace = Trace(
            trace_id=str(uuid.uuid4())[:12],
            question=question,
            start_time=time.time(),
            metadata=metadata or {},
        )
        return trace

    class _SpanContext:
        """Context manager for timing a span."""

        def __init__(self, trace: Trace, span: Span):
            self.trace = trace
            self.span = span

        def __enter__(self) -> Span:
            self.span.start_time = time.time()
            return self.span

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.span.end_time = time.time()
            self.span.latency_ms = (self.span.end_time - self.span.start_time) * 1000
            if exc_type:
                self.span.status = "error"
                self.span.error = str(exc_val)
            self.trace.spans.append(self.span)
            return False  # don't suppress exceptions

    def span(self, trace: Trace, name: str) -> _SpanContext:
        """Create a span context manager for timing a pipeline stage."""
        return self._SpanContext(trace, Span(name=name))

    def end_trace(self, trace: Trace) -> None:
        """Finalize and persist a trace."""
        trace.total_latency_ms = (time.time() - trace.start_time) * 1000

        if self.enabled:
            self._persist_trace(trace)

        logger.info(
            "Trace %s: %s — %.0fms (%d spans)",
            trace.trace_id,
            trace.question[:40],
            trace.total_latency_ms,
            len(trace.spans),
        )

    def _persist_trace(self, trace: Trace) -> None:
        """Save trace to a JSON file."""
        try:
            trace_file = self.trace_dir / f"{trace.trace_id}.json"
            with open(trace_file, "w") as f:
                json.dump(trace.to_dict(), f, indent=2)
        except Exception as e:
            logger.error("Failed to persist trace %s: %s", trace.trace_id, e)

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Load the most recent traces from disk."""
        if not self.trace_dir.exists():
            return []

        files = sorted(self.trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        traces = []
        for f in files[:limit]:
            try:
                with open(f) as fh:
                    traces.append(json.load(fh))
            except Exception:
                continue
        return traces
