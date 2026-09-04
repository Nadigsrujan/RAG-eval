"""
Pydantic schemas for API request/response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Requests ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of evidence chunks to retrieve")
    use_orchestrator: bool = Field(
        True, description="Enable multi-agent orchestration and self-reflection validation"
    )


# ── Core Component Schemas ─────────────────────────────────────

class SourceInfo(BaseModel):
    """A source document reference."""

    document: str
    page: int
    score: float


class CitationInfo(BaseModel):
    """A validated citation from the answer."""

    document: str
    page: int
    is_valid: bool
    reason: str = ""


class TimingInfo(BaseModel):
    """Latency breakdown for the pipeline."""

    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    citation_ms: float = 0.0
    total_ms: float = 0.0


# ── Agentic & Orchestration Schemas ───────────────────────────

class AgentThoughtInfo(BaseModel):
    """A recorded reasoning or validation step by an autonomous agent."""

    agent: str
    step: str
    thought: str
    status: str = "ok"
    latency_ms: float = 0.0
    data: dict = {}


class AgentValidationReport(BaseModel):
    """Automated validation metrics produced by the agentic pipeline."""

    retrieval_valid: bool = True
    retrieval_score: float = 0.0
    faithfulness_score: float = 1.0
    citation_accuracy: float = 1.0
    self_corrected: bool = False
    validation_reasons: list[str] = []


class OrchestratedQueryResponse(BaseModel):
    """Enhanced response containing full agentic rationale and validation audit."""

    answer: str
    intent: str = "direct"
    sub_queries: list[str] = []
    sources: list[SourceInfo] = []
    citations: list[CitationInfo] = []
    agent_thoughts: list[AgentThoughtInfo] = []
    validation: AgentValidationReport = Field(default_factory=AgentValidationReport)
    timings: TimingInfo = Field(default_factory=TimingInfo)
    metadata: dict = {}


class OrchestratedIngestResponse(BaseModel):
    """Response body for orchestrated document intake and index validation."""

    status: str
    documents_processed: int
    total_chunks: int
    validation_status: str
    checks: list[dict] = []
    message: str


class OrchestratedValidationResponse(BaseModel):
    """Automated end-to-end pipeline health and SLA audit response."""

    overall_status: str
    retrieval_status: str
    generation_status: str
    retrieval_metrics: dict[str, float] = {}
    generation_metrics: dict[str, float] = {}
    sla_checks: list[dict] = []
    recommendations: list[str] = []
    elapsed_seconds: float = 0.0
    message: str = ""


# ── Responses ──────────────────────────────────────────────────


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    sources: list[SourceInfo] = []
    citations: list[CitationInfo] = []
    timings: TimingInfo = TimingInfo()
    metadata: dict = {}


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    status: str
    documents_processed: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    index_loaded: bool
    num_chunks: int = 0
    model_loaded: bool = False


class MetricsResponse(BaseModel):
    """Response body for the /metrics endpoint."""

    total_queries: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_retrieval_ms: float = 0.0
    avg_generation_ms: float = 0.0
    avg_citation_score: float = 0.0


# ── Evaluation Responses ───────────────────────────────────────

class RetrievalEvalResponse(BaseModel):
    """Retrieval evaluation metrics."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    precision_at_5: float = 0.0
    num_queries: int = 0


class GenerationEvalResponse(BaseModel):
    """Generation evaluation metrics."""

    exact_match: float = 0.0
    fuzzy_f1: float = 0.0
    contains_match: float = 0.0
    avg_answer_length: float = 0.0
    num_queries: int = 0


class CitationEvalResponse(BaseModel):
    """Citation evaluation metrics."""

    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    avg_citations_per_answer: float = 0.0
    num_queries: int = 0


class EvalResponse(BaseModel):
    """Response body for the /evaluate endpoint."""

    status: str = "success"
    retrieval: RetrievalEvalResponse = RetrievalEvalResponse()
    generation: GenerationEvalResponse = GenerationEvalResponse()
    citation: CitationEvalResponse = CitationEvalResponse()
    elapsed_seconds: float = 0.0
    message: str = ""
