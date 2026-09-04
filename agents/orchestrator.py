"""
Pipeline Orchestrator — central orchestration layer coordinating agents, workflows, and validation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from agents.base import OrchestrationState
from agents.evaluator_agent import PipelineHealthReport, PipelineValidatorAgent
from agents.generation_agent import GenerationAgent
from agents.ingestion_agent import IngestionAgent, IngestionValidationResult
from agents.query_router_agent import QueryRouterAgent
from agents.reflection_agent import ReflectionValidatorAgent
from agents.retrieval_agent import RetrievalValidationAgent
from generation.llm import Generator
from generation.prompts import PromptManager
from guardrails.safety import SafetyGuard
from ingestion.embedder import Embedder
from ingestion.indexer import IndexBundle
from observability.tracing import Tracer
from retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Coordinates multi-agent automation and validation across query, ingestion, and evaluation.
    """

    def __init__(
        self,
        config: dict,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generator: Generator | None = None,
        prompt_manager: PromptManager | None = None,
        embedder: Embedder | None = None,
    ):
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator
        self.prompt_manager = prompt_manager
        self.embedder = embedder

        orch_cfg = config.get("orchestration", {})
        self.tracer = Tracer(
            enabled=config.get("observability", {}).get("tracing_enabled", True),
            trace_dir=config.get("observability", {}).get("trace_dir", "traces/"),
        )
        self.safety_guard = SafetyGuard(
            min_retrieval_score=orch_cfg.get("min_retrieval_confidence", 0.1),
            max_response_tokens=config.get("generation", {}).get("max_new_tokens", 500),
        )

        # Initialize Specialized Agents
        self.router_agent = QueryRouterAgent()
        self.retrieval_agent = (
            RetrievalValidationAgent(
                retrieval_pipeline=retrieval_pipeline,
                min_confidence_score=orch_cfg.get("min_retrieval_confidence", 0.10),
            )
            if retrieval_pipeline
            else None
        )
        self.generation_agent = (
            GenerationAgent(
                generator=generator,
                prompt_manager=prompt_manager,
            )
            if generator and prompt_manager
            else None
        )
        self.reflection_agent = ReflectionValidatorAgent(
            faithfulness_threshold=orch_cfg.get("faithfulness_threshold", 0.70)
        )
        self.ingestion_agent = IngestionAgent(config)
        self.evaluator_agent = PipelineValidatorAgent(config)

    def update_pipeline(
        self,
        retrieval_pipeline: RetrievalPipeline,
        generator: Generator | None = None,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        """Update active pipeline components (e.g. after newly ingested documents)."""
        self.retrieval_pipeline = retrieval_pipeline
        orch_cfg = self.config.get("orchestration", {})
        self.retrieval_agent = RetrievalValidationAgent(
            retrieval_pipeline=retrieval_pipeline,
            min_confidence_score=orch_cfg.get("min_retrieval_confidence", 0.10),
        )
        if generator:
            self.generator = generator
        if prompt_manager:
            self.prompt_manager = prompt_manager
        if self.generator and self.prompt_manager:
            self.generation_agent = GenerationAgent(
                generator=self.generator,
                prompt_manager=self.prompt_manager,
            )

    def orchestrate_query(self, question: str, top_k: int = 5) -> OrchestrationState:
        """
        Execute the full orchestrated RAG query lifecycle:
          Guardrails Check -> Router Agent -> Retrieval Agent (with Self-RAG check)
          -> Generation Agent -> Reflection Validator Agent (with Self-Correction)
        """
        state = OrchestrationState(query=question, top_k=top_k)
        total_start = time.time()
        trace = self.tracer.start_trace(question)

        try:
            # 1. Safety Guard Check
            with self.tracer.span(trace, "safety_guard") as span:
                t0 = time.time()
                is_safe, reason = self.safety_guard.check_query(question)
                span.metadata = {"is_safe": is_safe, "reason": reason}
                if not is_safe:
                    state.final_answer = f"Safety refusal: {reason}"
                    state.add_thought(
                        agent="SafetyGuard",
                        step="input_guardrail",
                        thought=f"Query rejected by safety filter: {reason}",
                        status="error",
                        latency_ms=(time.time() - t0) * 1000,
                    )
                    state.timings["total_ms"] = (time.time() - total_start) * 1000
                    return state

            # Check if index is ready
            if not self.retrieval_agent or not self.generation_agent:
                state.final_answer = "No documents have been ingested yet. Please upload PDFs first."
                state.add_thought(
                    agent="PipelineOrchestrator",
                    step="readiness_check",
                    thought="Retrieval pipeline not initialized. Documents must be ingested first.",
                    status="warning",
                )
                state.timings["total_ms"] = (time.time() - total_start) * 1000
                return state

            # 2. Query Router Agent
            with self.tracer.span(trace, "query_router_agent") as span:
                state = self.router_agent.execute(state)
                span.metadata = {
                    "intent": state.intent,
                    "sub_queries": state.sub_queries,
                    "rewritten_query": state.rewritten_query,
                }

            # 3. Retrieval Validation Agent (Self-RAG check)
            with self.tracer.span(trace, "retrieval_agent") as span:
                state = self.retrieval_agent.execute(state)
                span.metadata = {
                    "chunks_retrieved": len(state.retrieved_chunks),
                    "retrieval_valid": state.retrieval_valid,
                    "retrieval_score": state.retrieval_score,
                }

            # 4. Generation Agent
            with self.tracer.span(trace, "generation_agent") as span:
                state = self.generation_agent.execute(state)
                span.metadata = {
                    "draft_answer_length": len(state.draft_answer),
                    "sources": len(state.sources),
                }

            # 5. Reflection & Validation Agent (Critic & Self-Correction)
            with self.tracer.span(trace, "reflection_agent") as span:
                state = self.reflection_agent.execute(state)
                span.metadata = {
                    "faithfulness_score": state.faithfulness_score,
                    "citation_accuracy": state.citation_accuracy,
                    "self_corrected": state.self_corrected,
                }

            total_ms = (time.time() - total_start) * 1000
            state.timings["total_ms"] = total_ms

            # Log orchestrator summary
            logger.info(
                "Query orchestrated in %.1fms | Intent: %s | Chunks: %d | Faithfulness: %.2f | Corrected: %s",
                total_ms,
                state.intent,
                len(state.filtered_chunks),
                state.faithfulness_score,
                state.self_corrected,
            )
            return state
        finally:
            self.tracer.end_trace(trace)

    def orchestrate_ingestion(
        self, pdf_paths: list[Path]
    ) -> tuple[IndexBundle, IngestionValidationResult]:
        """Orchestrate PDF ingestion with pre- and post-validation audits."""
        trace = self.tracer.start_trace(f"ingestion_{len(pdf_paths)}_docs")
        try:
            with self.tracer.span(trace, "ingestion_agent"):
                bundle, audit = self.ingestion_agent.ingest_and_validate(pdf_paths)
                # Auto-wire new bundle into active retrieval pipeline
                config_retrieval = RetrievalPipeline(bundle, self.config, self.embedder)
                self.update_pipeline(config_retrieval)
                return bundle, audit
        finally:
            self.tracer.end_trace(trace)

    def orchestrate_validation(self) -> PipelineHealthReport:
        """Run automated full pipeline benchmark audit."""
        if not self.retrieval_pipeline or not self.generator or not self.prompt_manager:
            report = PipelineHealthReport(overall_status="UNHEALTHY")
            report.recommendations.append(
                "Pipeline components not fully initialized. Upload documents before auditing."
            )
            return report

        def retrieval_fn(question: str) -> list[str]:
            res = self.retrieval_pipeline.retrieve(question)
            return [sc.chunk.chunk_id for sc in res.chunks]

        def generation_fn(question: str) -> str:
            res = self.retrieval_pipeline.retrieve(question)
            from generation.context_builder import ContextBuilder
            context = ContextBuilder.build(res.chunks)
            prompt = self.prompt_manager.format(question, context)
            gen = self.generator.generate(prompt)
            return gen.answer

        trace = self.tracer.start_trace("pipeline_audit")
        try:
            with self.tracer.span(trace, "pipeline_evaluator_agent"):
                return self.evaluator_agent.audit_pipeline(retrieval_fn, generation_fn)
        finally:
            self.tracer.end_trace(trace)
