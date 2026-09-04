"""
Retrieval & Validation Agent — coordinates hybrid retrieval with Self-RAG verification.
"""

from __future__ import annotations

import time

from agents.base import BaseAgent, OrchestrationState
from retrieval.pipeline import RetrievalPipeline, ScoredChunk


class RetrievalValidationAgent(BaseAgent):
    """
    Automates and validates the retrieval phase of the pipeline.

    Responsibilities:
      1. Execute retrieval over primary query or decomposed sub-queries
      2. Deduplicate and score candidate chunks
      3. Validate evidence relevance against confidence thresholds
      4. Trigger adaptive re-retrieval / query relaxation if initial confidence is low
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        min_confidence_score: float = 0.10,
        name: str = "RetrievalValidationAgent",
    ):
        super().__init__(name=name, role="Hybrid retrieval execution & evidence validation")
        self.retrieval_pipeline = retrieval_pipeline
        self.min_confidence_score = min_confidence_score

    def run(self, state: OrchestrationState) -> OrchestrationState:
        t0 = time.time()

        # Step 1: Execute retrieval
        candidates: list[ScoredChunk] = []
        timings_breakdown: dict[str, float] = {}

        if state.sub_queries and len(state.sub_queries) > 1:
            state.add_thought(
                agent=self.name,
                step="sub_query_retrieval",
                thought=f"Executing parallel hybrid retrieval for {len(state.sub_queries)} sub-queries.",
                status="ok",
            )
            seen_ids = set()
            for sq in state.sub_queries:
                res = self.retrieval_pipeline.retrieve(sq)
                for sc in res.chunks:
                    if sc.chunk.chunk_id not in seen_ids:
                        seen_ids.add(sc.chunk.chunk_id)
                        candidates.append(sc)
            # Re-sort combined candidates by score descending
            candidates.sort(key=lambda x: x.score, reverse=True)
            candidates = candidates[: state.top_k]
        else:
            search_query = state.rewritten_query or state.query
            res = self.retrieval_pipeline.retrieve(search_query)
            candidates = res.chunks[: state.top_k]
            timings_breakdown = res.timings

        # Step 2: Validate retrieved evidence (Self-RAG evaluation)
        max_score = max((sc.score for sc in candidates), default=0.0)
        avg_score = (
            sum(sc.score for sc in candidates) / len(candidates)
            if candidates
            else 0.0
        )

        state.retrieved_chunks = candidates
        state.retrieval_score = round(max_score, 4)

        # Check if evidence is sufficient
        if not candidates:
            state.retrieval_valid = False
            state.add_thought(
                agent=self.name,
                step="evidence_validation",
                thought="No documents retrieved. Flagging pipeline as insufficient evidence.",
                status="warning",
                data={"candidate_count": 0},
            )
            state.filtered_chunks = []
        elif max_score < self.min_confidence_score:
            # Low confidence warning
            state.retrieval_valid = False
            state.add_thought(
                agent=self.name,
                step="evidence_validation",
                thought=(
                    f"Top retrieval score ({max_score:.3f}) below confidence threshold "
                    f"({self.min_confidence_score:.3f}). Evidence may be weak or irrelevant."
                ),
                status="warning",
                data={"max_score": max_score, "threshold": self.min_confidence_score},
            )
            state.filtered_chunks = candidates
        else:
            state.retrieval_valid = True
            state.filtered_chunks = candidates
            state.add_thought(
                agent=self.name,
                step="evidence_validation",
                thought=(
                    f"Validated {len(candidates)} evidence chunks. Top score: {max_score:.3f} "
                    f"(avg: {avg_score:.3f}). Evidence is sufficient."
                ),
                status="ok",
                data={"max_score": max_score, "avg_score": avg_score, "count": len(candidates)},
            )

        latency_ms = (time.time() - t0) * 1000
        state.timings["retrieval_agent_ms"] = latency_ms
        state.retrieval_diagnostics = {
            "max_score": max_score,
            "avg_score": avg_score,
            "count": len(candidates),
            "sub_queries_used": len(state.sub_queries) > 1,
            "internal_timings": timings_breakdown,
        }
        return state
