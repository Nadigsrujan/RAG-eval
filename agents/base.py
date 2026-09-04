"""
Base structures and interfaces for the Agentic Orchestration Layer.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentThought:
    """Individual agent reasoning step or validation action."""

    agent: str
    step: str
    thought: str
    status: str = "ok"  # ok | warning | error
    latency_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "step": self.step,
            "thought": self.thought,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 1),
            "data": self.data,
        }


@dataclass
class OrchestrationState:
    """Execution state passed through the agent orchestration pipeline."""

    query: str
    top_k: int = 5
    # Router stage
    intent: str = "direct"  # direct | multi_hop | exploratory | invalid
    sub_queries: list[str] = field(default_factory=list)
    rewritten_query: str | None = None

    # Retrieval stage
    retrieved_chunks: list[Any] = field(default_factory=list)
    filtered_chunks: list[Any] = field(default_factory=list)
    retrieval_valid: bool = True
    retrieval_score: float = 0.0
    retrieval_diagnostics: dict[str, Any] = field(default_factory=dict)

    # Generation stage
    draft_answer: str = ""
    final_answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)

    # Reflection & Validation stage
    faithfulness_score: float = 1.0
    citation_accuracy: float = 1.0
    self_corrected: bool = False
    validation_reasons: list[str] = field(default_factory=list)

    # Execution telemetry
    thoughts: list[AgentThought] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def add_thought(
        self,
        agent: str,
        step: str,
        thought: str,
        status: str = "ok",
        latency_ms: float = 0.0,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append an agent reasoning or validation step."""
        self.thoughts.append(
            AgentThought(
                agent=agent,
                step=step,
                thought=thought,
                status=status,
                latency_ms=latency_ms,
                data=data or {},
            )
        )


class BaseAgent(ABC):
    """Abstract base agent for pipeline components."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def execute(self, state: OrchestrationState) -> OrchestrationState:
        """Execute the agent with timing and error isolation."""
        t0 = time.time()
        logger.debug("Starting agent: %s (%s)", self.name, self.role)
        try:
            state = self.run(state)
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error("Agent %s failed: %s", self.name, e, exc_info=True)
            state.add_thought(
                agent=self.name,
                step="error",
                thought=f"Encountered error: {e}",
                status="error",
                latency_ms=latency_ms,
            )
            state.error = str(e)
        return state

    @abstractmethod
    def run(self, state: OrchestrationState) -> OrchestrationState:
        """Run agent core logic on the state."""
        pass
