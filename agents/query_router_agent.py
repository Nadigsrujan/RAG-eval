"""
Query Router Agent — analyzes query intent, complexity, and decomposition.
"""

from __future__ import annotations

import re
import time

from agents.base import BaseAgent, OrchestrationState


class QueryRouterAgent(BaseAgent):
    """
    Analyzes and classifies incoming user questions.

    Responsibilities:
      1. Classify intent: direct | multi_hop | exploratory | invalid
      2. Decompose multi-hop or comparative questions into focused sub-queries
      3. Formulate targeted search variations (query rewriting)
    """

    def __init__(self, name: str = "QueryRouterAgent"):
        super().__init__(name=name, role="Query intent classification & decomposition")

    def run(self, state: OrchestrationState) -> OrchestrationState:
        t0 = time.time()
        query = state.query.strip()

        # Intent detection heuristics
        intent = self._classify_intent(query)
        sub_queries = []
        rewritten_query = query

        if intent == "multi_hop":
            sub_queries = self._decompose_query(query)
            state.add_thought(
                agent=self.name,
                step="decomposition",
                thought=f"Decomposed multi-hop question into {len(sub_queries)} sub-queries.",
                status="ok",
                data={"sub_queries": sub_queries},
            )
        elif intent == "exploratory":
            rewritten_query = self._expand_exploratory_query(query)
            state.add_thought(
                agent=self.name,
                step="query_expansion",
                thought="Expanded exploratory query for comprehensive coverage.",
                status="ok",
                data={"original": query, "expanded": rewritten_query},
            )
        else:
            state.add_thought(
                agent=self.name,
                step="intent_classification",
                thought=f"Classified query as '{intent}' direct retrieval.",
                status="ok",
                data={"intent": intent},
            )

        latency_ms = (time.time() - t0) * 1000
        state.intent = intent
        state.sub_queries = sub_queries
        state.rewritten_query = rewritten_query
        state.timings["router_ms"] = latency_ms
        return state

    def _classify_intent(self, query: str) -> str:
        lower = query.lower()

        # Multi-hop / comparative indicators
        comparative_markers = [
            " compare ", " vs ", " versus ", " difference between ",
            " differences between ", " compared to ", " and how does ",
            " as well as ", " both "
        ]
        if any(marker in lower for marker in comparative_markers) or ("?" in query and query.count("?") > 1):
            return "multi_hop"

        # Exploratory indicators
        exploratory_markers = [
            "summarize", "summary", "overview", "what are all",
            "outline", "explain the entire", "high level"
        ]
        if any(lower.startswith(m) or f" {m} " in f" {lower} " for m in exploratory_markers):
            return "exploratory"

        return "direct"

    def _decompose_query(self, query: str) -> list[str]:
        """Decompose comparative or dual-target queries into distinct sub-questions."""
        lower = query.lower()
        parts: list[str] = []

        if " vs " in lower or " versus " in lower:
            tokens = re.split(r"\b(?:vs|versus)\b", query, flags=re.IGNORECASE)
            if len(tokens) == 2:
                parts = [
                    f"What is {tokens[0].strip()}?",
                    f"What is {tokens[1].strip()}?",
                ]
        elif "difference between" in lower:
            match = re.search(r"difference between\s+([^?]+?)\s+and\s+([^?]+)", query, re.IGNORECASE)
            if match:
                parts = [
                    f"Details regarding {match.group(1).strip()}",
                    f"Details regarding {match.group(2).strip()}",
                ]
        elif "?" in query and query.count("?") > 1:
            raw_splits = [q.strip() + "?" for q in query.split("?") if q.strip()]
            if len(raw_splits) > 1:
                parts = raw_splits

        # Fallback if specific pattern didn't match
        if not parts:
            parts = [query]

        return parts

    def _expand_exploratory_query(self, query: str) -> str:
        """Slightly expand exploratory queries with key document summary terms."""
        clean = re.sub(r"^(?:please\s+)?(?:give\s+me\s+)?(?:a\s+)?(?:summary\s+of|summarize|overview\s+of)\s+", "", query, flags=re.IGNORECASE).strip()
        return f"{clean} main concepts key findings summary"
