"""
FastAPI application — the main backend for the document intelligence system.

Endpoints:
  POST /ingest   — Upload PDF(s) and build the retrieval index
  POST /query    — Ask a question and get an answer with citations
  GET  /health   — Health check
  GET  /metrics  — Pipeline performance metrics
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.orchestrator import PipelineOrchestrator
from api.schemas import (
    AgentThoughtInfo,
    AgentValidationReport,
    CitationInfo,
    EvalResponse,
    GenerationEvalResponse,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
    OrchestratedIngestResponse,
    OrchestratedQueryResponse,
    OrchestratedValidationResponse,
    QueryRequest,
    QueryResponse,
    RetrievalEvalResponse,
    SourceInfo,
    TimingInfo,
)
from generation.citation import CitationExtractor
from generation.context_builder import ContextBuilder
from generation.llm import Generator
from generation.prompts import PromptManager
from ingestion.embedder import Embedder
from ingestion.indexer import IndexBundle
from ingestion.pipeline import IngestionPipeline
from retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

# ── Global State ───────────────────────────────────────────────

_config: dict = {}
_index_bundle: IndexBundle | None = None
_retrieval_pipeline: RetrievalPipeline | None = None
_generator: Generator | None = None
_prompt_manager: PromptManager | None = None
_embedder: Embedder | None = None
_orchestrator: PipelineOrchestrator | None = None
_query_metrics: list[dict] = []


def _load_config() -> dict:
    """Load config from configs/default.yaml or environment."""
    config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
    if Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    logger.warning("Config not found at %s, using defaults", config_path)
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    global _config, _embedder, _generator, _prompt_manager, _orchestrator

    _config = _load_config()
    gen_cfg = _config.get("generation", {})
    retrieval_cfg = _config.get("retrieval", {})

    # Pre-initialize the embedder (shared between ingestion and retrieval)
    _embedder = Embedder(
        model_name=retrieval_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Pre-initialize the generator and trigger model load
    _generator = Generator(
        model_name=gen_cfg.get("model", "Qwen/Qwen3-4B-Instruct-2507"),
        max_new_tokens=gen_cfg.get("max_new_tokens", 500),
        max_input_tokens=gen_cfg.get("max_input_tokens", 2048),
        temperature=gen_cfg.get("temperature", 0.1),
    )
    # Eager load the model so first-request latency doesn't block
    try:
        _ = _generator.pipeline
    except Exception as e:
        logger.error("Failed to eager load generator model: %s", e)

    _prompt_manager = PromptManager(version=gen_cfg.get("prompt_version", "v1"))

    # Try to load pre-built index if it exists
    index_dir = Path("indices/")
    if index_dir.exists() and (index_dir / "faiss.index").exists():
        try:
            global _index_bundle, _retrieval_pipeline
            _index_bundle = IndexBundle.load(index_dir)
            _retrieval_pipeline = RetrievalPipeline(_index_bundle, _config, _embedder)
            logger.info("Pre-built index loaded: %d chunks", _index_bundle.num_chunks)
        except Exception as e:
            logger.error("Failed to load pre-built index: %s", e)

    # Initialize the Orchestration Layer
    _orchestrator = PipelineOrchestrator(
        config=_config,
        retrieval_pipeline=_retrieval_pipeline,
        generator=_generator,
        prompt_manager=_prompt_manager,
        embedder=_embedder,
    )

    logger.info("Application started with Orchestrator ready")
    yield
    logger.info("Application shutting down")


# ── App Creation ───────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Document Intelligence",
    description="Production-grade RAG system with hybrid retrieval, evaluation, and observability",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for the web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web frontend
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


# ── Endpoints ──────────────────────────────────────────────────


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the web frontend."""
    index_path = web_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Document Intelligence API. Visit /docs for API documentation."}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: list[UploadFile] = File(...)):
    """
    Upload PDF documents and build the retrieval index.

    Accepts one or more PDF files. Runs the full ingestion pipeline:
    parse → chunk → embed → index with automated index validation.
    """
    global _index_bundle, _retrieval_pipeline

    # Save uploaded files to temp directory
    pdf_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue
            tmp_path = Path(tmp_dir) / file.filename
            content = await file.read()
            tmp_path.write_bytes(content)
            pdf_paths.append(tmp_path)

        if not pdf_paths:
            return IngestResponse(
                status="error",
                documents_processed=0,
                total_chunks=0,
                message="No valid PDF files were uploaded.",
            )

        # Run ingestion pipeline through orchestrator if available
        if _orchestrator:
            _index_bundle, audit = _orchestrator.orchestrate_ingestion(pdf_paths)
            _retrieval_pipeline = _orchestrator.retrieval_pipeline
            return IngestResponse(
                status="success",
                documents_processed=audit.documents_count,
                total_chunks=audit.chunks_count,
                message=(
                    f"Successfully ingested and validated {audit.documents_count} "
                    f"document(s) into {audit.chunks_count} chunks. Index status: {audit.validation_status}."
                ),
            )
        else:
            pipeline = IngestionPipeline(_config)
            _index_bundle = pipeline.ingest(pdf_paths, save_dir="indices/")
            _retrieval_pipeline = RetrievalPipeline(_index_bundle, _config, _embedder)

    return IngestResponse(
        status="success",
        documents_processed=len(pdf_paths),
        total_chunks=_index_bundle.num_chunks,
        message=f"Successfully ingested {len(pdf_paths)} document(s) into {_index_bundle.num_chunks} chunks.",
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Ask a question about the ingested documents.

    When use_orchestrator is True, runs the multi-agent orchestration pipeline:
    SafetyGuard → QueryRouterAgent → RetrievalValidationAgent → GenerationAgent → ReflectionValidatorAgent.
    """
    if _retrieval_pipeline is None or _generator is None:
        return QueryResponse(
            answer="No documents have been ingested yet. Please upload PDFs first.",
            metadata={"error": "no_index"},
        )

    # Agentic Orchestrated Flow
    if request.use_orchestrator and _orchestrator:
        state = _orchestrator.orchestrate_query(request.question, top_k=request.top_k)

        timings = TimingInfo(
            retrieval_ms=round(state.timings.get("retrieval_agent_ms", 0), 1),
            reranking_ms=round(
                state.retrieval_diagnostics.get("internal_timings", {}).get("reranking_ms", 0), 1
            ),
            generation_ms=round(state.timings.get("generation_agent_ms", 0), 1),
            citation_ms=round(state.timings.get("reflection_agent_ms", 0), 1),
            total_ms=round(state.timings.get("total_ms", 0), 1),
        )

        _query_metrics.append({
            "total_ms": state.timings.get("total_ms", 0),
            "retrieval_ms": state.timings.get("retrieval_agent_ms", 0),
            "generation_ms": state.timings.get("generation_agent_ms", 0),
            "citation_score": state.citation_accuracy,
        })

        return QueryResponse(
            answer=state.final_answer,
            sources=[
                SourceInfo(document=s["document"], page=s["page"], score=s["score"])
                for s in state.sources
            ],
            citations=[
                CitationInfo(
                    document=c["document"],
                    page=c["page"],
                    is_valid=c["is_valid"],
                    reason=c.get("reason", ""),
                )
                for c in state.citations
            ],
            timings=timings,
            metadata={
                **state.metadata,
                "intent": state.intent,
                "sub_queries": state.sub_queries,
                "faithfulness_score": state.faithfulness_score,
                "citation_accuracy": state.citation_accuracy,
                "self_corrected": state.self_corrected,
                "agent_thoughts_count": len(state.thoughts),
                "retrieval_valid": state.retrieval_valid,
            },
        )

    # Legacy Direct Pipeline Fallback
    total_start = time.time()
    retrieval_result = _retrieval_pipeline.retrieve(request.question)
    context = ContextBuilder.build(retrieval_result.chunks)
    sources = ContextBuilder.get_source_list(retrieval_result.chunks)

    t0 = time.time()
    prompt = _prompt_manager.format(request.question, context)
    gen_result = _generator.generate(prompt)
    generation_ms = (time.time() - t0) * 1000

    t0 = time.time()
    raw_citations = CitationExtractor.extract(gen_result.answer)
    if not raw_citations and retrieval_result.chunks:
        top_chunk = retrieval_result.chunks[0].chunk
        gen_result.answer = f"{gen_result.answer} [Source: {top_chunk.document}, Page {top_chunk.page}]"
        raw_citations = CitationExtractor.extract(gen_result.answer)

    validated = CitationExtractor.validate(raw_citations, retrieval_result.chunks)
    citation_ms = (time.time() - t0) * 1000
    total_ms = (time.time() - total_start) * 1000

    timings = TimingInfo(
        retrieval_ms=round(retrieval_result.timings.get("retrieval_ms", 0), 1),
        reranking_ms=round(retrieval_result.timings.get("reranking_ms", 0), 1),
        generation_ms=round(generation_ms, 1),
        citation_ms=round(citation_ms, 1),
        total_ms=round(total_ms, 1),
    )

    _query_metrics.append({
        "total_ms": total_ms,
        "retrieval_ms": retrieval_result.timings.get("retrieval_ms", 0),
        "generation_ms": generation_ms,
        "citation_score": CitationExtractor.citation_score(validated),
    })

    return QueryResponse(
        answer=gen_result.answer,
        sources=[
            SourceInfo(document=s["document"], page=s["page"], score=s["score"])
            for s in sources
        ],
        citations=[
            CitationInfo(
                document=v.citation.document,
                page=v.citation.page,
                is_valid=v.is_valid,
                reason=v.reason,
            )
            for v in validated
        ],
        timings=timings,
        metadata={
            "input_tokens": gen_result.input_tokens,
            "output_tokens": gen_result.output_tokens,
            "model": gen_result.model_name,
            "retrieval_mode": retrieval_result.metadata.get("mode", ""),
            "num_candidates": retrieval_result.metadata.get("initial_candidates", 0),
        },
    )


# ── Explicit Orchestration Endpoints ───────────────────────────

@app.post("/orchestrate/query", response_model=OrchestratedQueryResponse)
def orchestrate_query(request: QueryRequest):
    """
    Execute a query through the full agentic orchestration layer.
    Returns agent thoughts, sub-queries, validation reports, and citations.
    """
    if _orchestrator is None or _retrieval_pipeline is None:
        return OrchestratedQueryResponse(
            answer="No documents have been ingested yet. Please upload PDFs first.",
            metadata={"error": "no_index"},
        )

    state = _orchestrator.orchestrate_query(request.question, top_k=request.top_k)
    timings = TimingInfo(
        retrieval_ms=round(state.timings.get("retrieval_agent_ms", 0), 1),
        reranking_ms=round(
            state.retrieval_diagnostics.get("internal_timings", {}).get("reranking_ms", 0), 1
        ),
        generation_ms=round(state.timings.get("generation_agent_ms", 0), 1),
        citation_ms=round(state.timings.get("reflection_agent_ms", 0), 1),
        total_ms=round(state.timings.get("total_ms", 0), 1),
    )

    return OrchestratedQueryResponse(
        answer=state.final_answer,
        intent=state.intent,
        sub_queries=state.sub_queries,
        sources=[
            SourceInfo(document=s["document"], page=s["page"], score=s["score"])
            for s in state.sources
        ],
        citations=[
            CitationInfo(
                document=c["document"],
                page=c["page"],
                is_valid=c["is_valid"],
                reason=c.get("reason", ""),
            )
            for c in state.citations
        ],
        agent_thoughts=[
            AgentThoughtInfo(
                agent=t.agent,
                step=t.step,
                thought=t.thought,
                status=t.status,
                latency_ms=t.latency_ms,
                data=t.data,
            )
            for t in state.thoughts
        ],
        validation=AgentValidationReport(
            retrieval_valid=state.retrieval_valid,
            retrieval_score=state.retrieval_score,
            faithfulness_score=state.faithfulness_score,
            citation_accuracy=state.citation_accuracy,
            self_corrected=state.self_corrected,
            validation_reasons=state.validation_reasons,
        ),
        timings=timings,
        metadata=state.metadata,
    )


@app.post("/orchestrate/ingest", response_model=OrchestratedIngestResponse)
async def orchestrate_ingest(files: list[UploadFile] = File(...)):
    """Upload PDFs with automated pre- and post-validation audits."""
    global _index_bundle, _retrieval_pipeline

    pdf_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue
            tmp_path = Path(tmp_dir) / file.filename
            content = await file.read()
            tmp_path.write_bytes(content)
            pdf_paths.append(tmp_path)

        if not pdf_paths:
            return OrchestratedIngestResponse(
                status="error",
                documents_processed=0,
                total_chunks=0,
                validation_status="error",
                checks=[],
                message="No valid PDF files uploaded.",
            )

        if _orchestrator is None:
            return OrchestratedIngestResponse(
                status="error",
                documents_processed=0,
                total_chunks=0,
                validation_status="error",
                checks=[],
                message="Orchestration engine is not initialized.",
            )

        _index_bundle, audit = _orchestrator.orchestrate_ingestion(pdf_paths)
        _retrieval_pipeline = _orchestrator.retrieval_pipeline

    return OrchestratedIngestResponse(
        status="success",
        documents_processed=audit.documents_count,
        total_chunks=audit.chunks_count,
        validation_status=audit.validation_status,
        checks=audit.checks,
        message=f"Ingested and validated {audit.documents_count} document(s) into {audit.chunks_count} chunks.",
    )


@app.post("/orchestrate/validate", response_model=OrchestratedValidationResponse)
def orchestrate_validation():
    """Run automated pipeline SLA audit and emit health status with recommendations."""
    if _orchestrator is None or _retrieval_pipeline is None:
        return OrchestratedValidationResponse(
            overall_status="UNHEALTHY",
            retrieval_status="UNHEALTHY",
            generation_status="UNHEALTHY",
            message="No documents ingested. Upload PDFs before auditing.",
        )

    report = _orchestrator.orchestrate_validation()
    return OrchestratedValidationResponse(
        overall_status=report.overall_status,
        retrieval_status=report.retrieval_status,
        generation_status=report.generation_status,
        retrieval_metrics=report.retrieval_metrics,
        generation_metrics=report.generation_metrics,
        sla_checks=report.sla_checks,
        recommendations=report.recommendations,
        elapsed_seconds=report.elapsed_seconds,
        message=f"Pipeline Health: {report.overall_status}. {len(report.sla_checks)} SLA checks evaluated.",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and readiness."""
    return HealthResponse(
        status="healthy",
        index_loaded=_index_bundle is not None,
        num_chunks=_index_bundle.num_chunks if _index_bundle else 0,
        model_loaded=_generator is not None and _generator._pipeline is not None,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get aggregate pipeline performance metrics."""
    if not _query_metrics:
        return MetricsResponse()

    import statistics

    totals = [m["total_ms"] for m in _query_metrics]
    retrievals = [m["retrieval_ms"] for m in _query_metrics]
    generations = [m["generation_ms"] for m in _query_metrics]
    citations = [m["citation_score"] for m in _query_metrics]

    sorted_totals = sorted(totals)
    p95_idx = int(len(sorted_totals) * 0.95)

    return MetricsResponse(
        total_queries=len(_query_metrics),
        avg_latency_ms=round(statistics.mean(totals), 1),
        p95_latency_ms=round(sorted_totals[min(p95_idx, len(sorted_totals) - 1)], 1),
        avg_retrieval_ms=round(statistics.mean(retrievals), 1),
        avg_generation_ms=round(statistics.mean(generations), 1),
        avg_citation_score=round(statistics.mean(citations), 4),
    )


@app.post("/evaluate", response_model=EvalResponse)
def run_evaluation():
    """
    Run the full RAG evaluation pipeline.

    Loads the eval dataset, runs retrieval + generation evaluation
    against the live pipeline, and returns structured metrics.
    Requires documents to be ingested first.
    """
    if _retrieval_pipeline is None or _generator is None:
        return EvalResponse(
            status="error",
            message="No documents ingested. Upload PDFs before running evaluation.",
        )

    import json

    from evaluation.generation_eval import GenerationEvaluator
    from evaluation.retrieval_eval import RetrievalEvaluator

    eval_cfg = _config.get("evaluation", {})
    dataset_path = Path(eval_cfg.get("dataset", "evaluation/datasets/eval_set.jsonl"))

    if not dataset_path.exists():
        return EvalResponse(
            status="error",
            message=f"Eval dataset not found: {dataset_path}",
        )

    # Load evaluation dataset
    eval_data = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))

    if not eval_data:
        return EvalResponse(
            status="error",
            message="Eval dataset is empty.",
        )

    eval_start = time.time()

    # --- Retrieval Evaluation ---
    retrieval_evaluator = RetrievalEvaluator()

    def retrieval_fn(question: str) -> list[str]:
        result = _retrieval_pipeline.retrieve(question)
        return [sc.chunk.chunk_id for sc in result.chunks]

    retrieval_metrics = retrieval_evaluator.evaluate(eval_data, retrieval_fn)

    # --- Generation Evaluation ---
    generation_evaluator = GenerationEvaluator()

    def generation_fn(question: str) -> str:
        retrieval_result = _retrieval_pipeline.retrieve(question)
        context = ContextBuilder.build(retrieval_result.chunks)
        prompt = _prompt_manager.format(question, context)
        gen_result = _generator.generate(prompt)
        return gen_result.answer

    generation_metrics = generation_evaluator.evaluate(eval_data, generation_fn)

    elapsed = time.time() - eval_start

    return EvalResponse(
        status="success",
        retrieval=RetrievalEvalResponse(
            recall_at_1=round(retrieval_metrics.recall_at_1, 4),
            recall_at_5=round(retrieval_metrics.recall_at_5, 4),
            recall_at_10=round(retrieval_metrics.recall_at_10, 4),
            mrr=round(retrieval_metrics.mrr, 4),
            ndcg_at_5=round(retrieval_metrics.ndcg_at_5, 4),
            precision_at_5=round(retrieval_metrics.precision_at_5, 4),
            num_queries=retrieval_metrics.num_queries,
        ),
        generation=GenerationEvalResponse(
            exact_match=round(generation_metrics.exact_match, 4),
            fuzzy_f1=round(generation_metrics.fuzzy_f1, 4),
            contains_match=round(generation_metrics.contains_match, 4),
            avg_answer_length=round(generation_metrics.avg_answer_length, 1),
            num_queries=generation_metrics.num_queries,
        ),
        elapsed_seconds=round(elapsed, 1),
        message=f"Evaluation complete: {len(eval_data)} items in {elapsed:.1f}s",
    )
