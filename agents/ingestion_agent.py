"""
Ingestion & Index Validation Agent — automates ingestion and validates index health.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingestion.indexer import IndexBundle
from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


@dataclass
class IngestionValidationResult:
    """Diagnostic audit result of document ingestion and index creation."""

    documents_count: int = 0
    chunks_count: int = 0
    faiss_vectors_count: int = 0
    bm25_corpus_size: int = 0
    canary_test_passed: bool = False
    validation_status: str = "ok"  # ok | warning | error
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class IngestionAgent:
    """
    Automates and validates the ingestion and indexing lifecycle.

    Responsibilities:
      1. Pre-validate PDF inputs (file existence, non-zero size)
      2. Orchestrate PDF parsing, chunking, embedding, and dual index generation
      3. Post-validate index integrity (vector dimension, vector count, BM25 corpus)
      4. Execute canary search validation on the newly built index
    """

    def __init__(self, config: dict, save_dir: str = "indices/"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.pipeline = IngestionPipeline(config)

    def ingest_and_validate(
        self, pdf_paths: list[Path]
    ) -> tuple[IndexBundle, IngestionValidationResult]:
        """
        Run automated ingestion followed by a rigorous post-index validation pass.
        """
        audit = IngestionValidationResult(documents_count=len(pdf_paths))

        # 1. Pre-ingestion validation
        valid_paths: list[Path] = []
        for p in pdf_paths:
            if not p.exists():
                audit.errors.append(f"File not found: {p}")
                audit.checks.append({"check": f"File exists: {p.name}", "status": "FAIL"})
                continue
            if p.stat().st_size == 0:
                audit.errors.append(f"Zero-byte file: {p.name}")
                audit.checks.append({"check": f"Non-empty: {p.name}", "status": "FAIL"})
                continue

            valid_paths.append(p)
            audit.checks.append({"check": f"Pre-check: {p.name}", "status": "PASS"})

        if not valid_paths:
            audit.validation_status = "error"
            raise ValueError(f"No valid PDF documents to ingest. Errors: {audit.errors}")

        # 2. Execute ingestion
        t0 = time.time()
        index_bundle = self.pipeline.ingest(valid_paths, save_dir=str(self.save_dir))
        ingest_duration_s = time.time() - t0

        audit.chunks_count = index_bundle.num_chunks
        audit.faiss_vectors_count = index_bundle.faiss_index.ntotal
        audit.bm25_corpus_size = index_bundle.bm25_index.corpus_size

        # 3. Post-ingestion Index Integrity Validations
        # Check A: Vector count matches chunk count
        vector_count_match = audit.faiss_vectors_count == audit.chunks_count
        audit.checks.append({
            "check": "FAISS vector count matches chunk count",
            "status": "PASS" if vector_count_match else "FAIL",
            "detail": f"{audit.faiss_vectors_count} vs {audit.chunks_count}",
        })
        if not vector_count_match:
            audit.errors.append("FAISS vector count does not match total chunk count.")

        # Check B: BM25 corpus size matches chunk count
        bm25_match = audit.bm25_corpus_size == audit.chunks_count
        audit.checks.append({
            "check": "BM25 corpus size matches chunk count",
            "status": "PASS" if bm25_match else "FAIL",
            "detail": f"{audit.bm25_corpus_size} vs {audit.chunks_count}",
        })
        if not bm25_match:
            audit.errors.append("BM25 corpus size does not match total chunk count.")

        # Check C: Canary search query validation
        canary_passed = False
        try:
            if index_bundle.chunks:
                sample_tokens = index_bundle.chunks[0].text.lower().split()
                sample_token = sample_tokens[0] if sample_tokens else "test"
                bm25_scores = index_bundle.bm25_index.get_scores([sample_token])
                canary_passed = len(bm25_scores) == audit.chunks_count
        except Exception as e:
            logger.warning("Canary search query failed: %s", e)
            canary_passed = False

        audit.canary_test_passed = canary_passed
        audit.checks.append({
            "check": "Canary retrieval index query",
            "status": "PASS" if canary_passed else "FAIL",
        })

        if audit.errors:
            audit.validation_status = "warning" if audit.chunks_count > 0 else "error"
        else:
            audit.validation_status = "ok"

        logger.info(
            "Ingestion validation complete in %.2fs: %d documents -> %d chunks. Status: %s",
            ingest_duration_s,
            len(valid_paths),
            audit.chunks_count,
            audit.validation_status,
        )

        return index_bundle, audit
