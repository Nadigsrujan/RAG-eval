"""
Ingestion Pipeline — orchestrates parse → chunk → embed → index.

This is the single entry point for ingesting PDF documents into the
retrieval system.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from ingestion.indexer import IndexBuilder, IndexBundle
from ingestion.pdf_parser import PDFParser

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end PDF ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline(config)
        bundle = pipeline.ingest(["/path/to/doc.pdf"])
        bundle.save("indices/")
    """

    def __init__(self, config: dict):
        ingestion_cfg = config.get("ingestion", {})
        retrieval_cfg = config.get("retrieval", {})

        self.parser = PDFParser(backend=ingestion_cfg.get("pdf_parser", "pdfplumber"))
        self.chunker = Chunker(
            chunk_size=ingestion_cfg.get("chunk_size", 300),
            chunk_overlap=ingestion_cfg.get("chunk_overlap", 50),
        )
        self.embedder = Embedder(
            model_name=retrieval_cfg.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

    def ingest(
        self,
        pdf_paths: list[str | Path],
        save_dir: Optional[str | Path] = None,
    ) -> IndexBundle:
        """
        Run the full ingestion pipeline.

        Args:
            pdf_paths: List of PDF file paths to ingest.
            save_dir: If provided, save the index bundle to this directory.

        Returns:
            IndexBundle containing all indices and chunk metadata.
        """
        start = time.time()

        # 1. Parse PDFs
        logger.info("═══ STEP 1: Parsing %d PDF(s) ═══", len(pdf_paths))
        documents = self.parser.parse_many(pdf_paths)
        if not documents:
            raise ValueError("No documents were successfully parsed.")

        total_pages = sum(doc.total_pages for doc in documents)
        logger.info("Parsed %d documents, %d total pages", len(documents), total_pages)

        # 2. Chunk documents
        logger.info("═══ STEP 2: Chunking ═══")
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError("No chunks were produced from the documents.")
        logger.info("Produced %d chunks", len(chunks))

        # 3. Embed chunks
        logger.info("═══ STEP 3: Embedding ═══")
        embeddings = self.embedder.embed_chunks(chunks)
        logger.info("Embeddings shape: %s", embeddings.shape)

        # 4. Build indices
        logger.info("═══ STEP 4: Building indices ═══")
        metadata = {
            "num_documents": len(documents),
            "num_pages": total_pages,
            "num_chunks": len(chunks),
            "embedding_model": self.embedder.model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "documents": [doc.filename for doc in documents],
        }
        bundle = IndexBuilder.build_all(chunks, embeddings, metadata)

        # 5. Optionally save
        if save_dir:
            bundle.save(save_dir)

        elapsed = time.time() - start
        logger.info(
            "═══ Ingestion complete in %.1fs: %d docs → %d chunks → %d vectors ═══",
            elapsed,
            len(documents),
            len(chunks),
            bundle.faiss_index.ntotal,
        )

        return bundle
