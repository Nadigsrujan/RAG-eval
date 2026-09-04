"""
Embedder — wraps SentenceTransformer to produce dense vectors for chunks and queries.

Supports batched embedding for efficiency and caches the model on first load.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    Dense embedding wrapper around SentenceTransformers.

    Args:
        model_name: HuggingFace model identifier for the embedding model.
        device: Device to run on ('cpu', 'cuda', 'mps'). None = auto-detect.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self._device)
            logger.info(
                "Embedding model loaded (dim=%d, device=%s)",
                self.embedding_dim,
                self._model.device,
            )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of text strings into dense vectors.

        Args:
            texts: List of text strings to embed.
            show_progress: Show a progress bar during encoding.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        logger.info("Embedding %d texts (batch_size=%d)", len(texts), self.batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )

        logger.info("Embedding complete: shape=%s", embeddings.shape)
        return embeddings

    def embed_chunks(self, chunks: list) -> np.ndarray:
        """Embed Chunk objects (extracts .text field)."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0]
