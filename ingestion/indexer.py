"""
Indexer — builds and manages FAISS and BM25 indices for retrieval.

Stores chunk metadata alongside indices so retrieval returns full
provenance information.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class IndexBundle:
    """
    Contains all index data needed for retrieval.

    This is the artifact produced by ingestion and consumed by retrieval.
    """

    faiss_index: object  # faiss.Index
    bm25_index: object  # BM25Okapi
    chunks: list[Chunk]
    embeddings: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def save(self, directory: str | Path) -> None:
        """Persist the index bundle to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        import faiss
        faiss.write_index(self.faiss_index, str(directory / "faiss.index"))

        # Save BM25 index
        with open(directory / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25_index, f)

        # Save chunks as JSON
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(directory / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2)

        # Save embeddings
        np.save(directory / "embeddings.npy", self.embeddings)

        # Save metadata
        with open(directory / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Index bundle saved to %s (%d chunks)", directory, len(self.chunks))

    @classmethod
    def load(cls, directory: str | Path) -> IndexBundle:
        """Load an index bundle from disk."""
        directory = Path(directory)

        # Load FAISS index
        import faiss
        faiss_index = faiss.read_index(str(directory / "faiss.index"))

        # Load BM25 index
        with open(directory / "bm25.pkl", "rb") as f:
            bm25_index = pickle.load(f)

        # Load chunks
        with open(directory / "chunks.json") as f:
            chunks_data = json.load(f)
        chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                document=c["document"],
                page=c["page"],
                start_char=c["start_char"],
                end_char=c["end_char"],
                metadata=c.get("metadata", {}),
            )
            for c in chunks_data
        ]

        # Load embeddings
        embeddings = np.load(directory / "embeddings.npy")

        # Load metadata
        metadata = {}
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        logger.info("Index bundle loaded from %s (%d chunks)", directory, len(chunks))
        return cls(
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
        )


class IndexBuilder:
    """
    Builds FAISS (dense) and BM25 (sparse) indices from chunks and embeddings.
    """

    @staticmethod
    def build_faiss_index(embeddings: np.ndarray) -> object:
        """
        Build a FAISS index from embedding vectors.

        Uses IndexFlatIP (inner product) since embeddings are L2-normalized,
        making inner product equivalent to cosine similarity.
        """
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
        index.add(embeddings.astype(np.float32))

        logger.info(
            "FAISS index built: %d vectors, dim=%d",
            index.ntotal,
            dim,
        )
        return index

    @staticmethod
    def build_bm25_index(chunks: list[Chunk]) -> object:
        """
        Build a BM25 index from chunk texts.

        Tokenizes by whitespace + lowercasing for simplicity.
        """
        from rank_bm25 import BM25Okapi

        tokenized = [chunk.text.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized)

        logger.info("BM25 index built: %d documents", len(chunks))
        return bm25

    @classmethod
    def build_all(
        cls,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> IndexBundle:
        """Build both indices and bundle them together."""
        faiss_index = cls.build_faiss_index(embeddings)
        bm25_index = cls.build_bm25_index(chunks)

        return IndexBundle(
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata or {},
        )
