---
name: agentic-patterns
description: "Architectural patterns and best practices for building autonomous, self-correcting, and validating AI agent systems in RAG and Document Intelligence pipelines."
---

# Agentic Patterns & Architectures for Document Intelligence

This skill provides architectural specifications and operational patterns for orchestrating multi-agent systems that automate and validate RAG (Retrieval-Augmented Generation) and document processing workflows.

---

## 1. Core Architectural Patterns

### 1.1 The Router / Dispatcher Pattern
- **Purpose**: Analyze incoming user queries or tasks before execution to classify intent, complexity, and safety.
- **Decision Space**:
  - `DIRECT`: Single-hop factual lookup. Routed to standard hybrid retrieval + generation.
  - `MULTI_HOP`: Complex or comparative inquiry requiring multiple sub-queries. Routed to the Planner / Query Decomposer.
  - `EXPLORATORY`: Broad summary or overview request requiring wide context aggregation.
  - `INVALID / ADVERSARIAL`: Out-of-bounds, prompt injection, or unsupported query. Routed to rejection or clarification.
- **Output Schema**:
  ```python
  class RouteDecision(BaseModel):
      intent: str  # direct | multi_hop | exploratory | invalid
      confidence: float
      reasoning: str
      sub_queries: list[str] = []
      rewritten_query: str | None = None
  ```

### 1.2 Planner & Query Decomposition Pattern
- **Purpose**: Break complex queries into atomic, verifiable sub-queries.
- **Execution**:
  - Sub-queries are retrieved concurrently or sequentially.
  - Results are aggregated and deduplicated with reciprocal rank scoring or provenance tracking.

### 1.3 Self-RAG / Retrieval Validation Pattern (Evaluator)
- **Purpose**: Validate retrieval output before invoking generation to prevent hallucinations caused by empty or noisy context.
- **Criteria**:
  - **Relevance Confidence**: Maximum and average similarity scores must meet minimum thresholds (e.g., cross-encoder score > 0.1).
  - **Coverage**: Top chunks must contain semantic overlap with key entities in the query.
- **Feedback Loop**:
  - If confidence is insufficient, trigger query expansion, alternative retriever (e.g. dense if BM25 had 0 hits, or vice-versa), or gracefully refuse with an informative notification.

### 1.4 Reflection & Self-Correction Pattern (Critic)
- **Purpose**: Post-generation inspection to guarantee factual faithfulness and citation validity.
- **Validation Passes**:
  1. **Faithfulness Check**: Every statement in the generated answer must be grounded in the retrieved chunks.
  2. **Citation Validation**: Citations must correspond to existing document names, valid page numbers, and chunk text.
- **Self-Healing Loop**:
  - If hallucinated content or broken citations are detected, the critic passes structured feedback back to the generator for a corrective retry.

### 1.5 Automated Ingestion & Index Integrity Pattern
- **Purpose**: Validate data quality at intake and verify index readiness.
- **Validation Passes**:
  1. **Document Health**: Check for zero-byte files, non-extractable scanned pages (OCR trigger), and empty text.
  2. **Index Audit**: Confirm non-zero FAISS vectors, matching dimension, valid BM25 vocabulary size, and run an automated canary query.

### 1.6 Continuous Pipeline Health & Evaluator Pattern
- **Purpose**: Automatically benchmark the end-to-end pipeline against test suites and monitor regression against SLAs.
- **Metrics Tracked**:
  - Retrieval: `Recall@K`, `MRR`, `NDCG@K`, `Precision@K`.
  - Generation: `Exact Match`, `Token F1`, `Contains Match`.
  - Citation: `Citation Precision`, `Citation Recall`, `Citation F1`.
  - Operational: `p50/p95/p99 Latency`, `Error Rate`.

---

## 2. Orchestration Layer Architecture

The **Orchestration Layer** governs the agent state machine:

```
[Query Input]
      │
      ▼
1. Safety Guard Check
      │
      ▼
2. QueryRouterAgent ──────► [Intent / Decomposition]
      │
      ▼
3. RetrievalValidationAgent ─► [Self-RAG Score Check] ─► (Fallback query expansion if low)
      │
      ▼
4. GenerationAgent ────────► [Draft Response with Citations]
      │
      ▼
5. ReflectionValidatorAgent ─► [Faithfulness & Citation Audit]
      │
      ├─► Pass ──► [Final Grounded Response + Provenance]
      └─► Fail ──► (Self-Correction Loop: max 1 retry) ──► [Corrected Response]
```

### 2.1 State Management
All steps operate on an immutable or thread-safe `OrchestrationState` context containing:
- Query, intent, and sub-queries.
- Raw retrieved chunks and filtered validated chunks.
- Draft answer, reflection critique, and final answer.
- Step-by-step agent thought logs (`AgentThought`) with timestamps and latencies.
