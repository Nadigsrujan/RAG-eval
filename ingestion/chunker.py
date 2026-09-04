"""
Chunker — splits parsed PDF pages into overlapping text chunks
with sentence-boundary awareness.

Each chunk carries provenance metadata (document, page, character offsets)
so we can trace any retrieved chunk back to its source.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ingestion.pdf_parser import Page, ParsedDocument

logger = logging.getLogger(__name__)

# Sentence boundary pattern: split on period/question/exclamation
# followed by whitespace and a capital letter, or end of string.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Chunk:
    """A text chunk with full provenance metadata."""

    chunk_id: str
    text: str
    document: str
    page: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (words ≈ 0.75 tokens for English)."""
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document": self.document,
            "page": self.page,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class Chunker:
    """
    Splits document text into overlapping chunks with sentence-boundary awareness.

    Args:
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of overlapping words between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Split an entire document into chunks, preserving page provenance."""
        all_chunks: list[Chunk] = []

        for page in document.pages:
            if not page.has_content:
                continue

            # Build page text including linearized tables
            page_text = page.text
            if page.tables:
                table_text = self._linearize_tables(page.tables)
                page_text = f"{page_text}\n\n{table_text}" if page_text else table_text

            page_chunks = self._split_text(
                text=page_text,
                document=document.filename,
                page=page.page_number,
            )
            all_chunks.extend(page_chunks)

        logger.info(
            "Chunked %s: %d pages → %d chunks",
            document.filename,
            document.total_pages,
            len(all_chunks),
        )
        return all_chunks

    def chunk_documents(self, documents: list[ParsedDocument]) -> list[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _split_text(self, text: str, document: str, page: int) -> list[Chunk]:
        """Split text into overlapping chunks respecting sentence boundaries."""
        if not text.strip():
            return []

        # Split into sentences first
        sentences = _SENTENCE_BOUNDARY.split(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_words: list[str] = []
        current_start = 0
        char_offset = 0

        for sentence in sentences:
            sentence_words = sentence.split()

            # If adding this sentence exceeds chunk_size and we have content,
            # emit a chunk and start new one with overlap
            if (
                len(current_words) + len(sentence_words) > self.chunk_size
                and current_words
            ):
                chunk_text = " ".join(current_words)
                chunk_id = self._make_chunk_id(document, page, current_start)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        document=document,
                        page=page,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                    )
                )

                # Overlap: keep the last chunk_overlap words
                overlap_words = current_words[-self.chunk_overlap :]
                current_words = overlap_words
                current_start = char_offset - len(" ".join(overlap_words))

            current_words.extend(sentence_words)
            char_offset += len(sentence) + 1  # +1 for the space between sentences

        # Emit the final chunk
        if current_words:
            chunk_text = " ".join(current_words)
            chunk_id = self._make_chunk_id(document, page, current_start)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    document=document,
                    page=page,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                )
            )

        return chunks

    @staticmethod
    def _make_chunk_id(document: str, page: int, start_char: int) -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{document}::p{page}::c{start_char}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _linearize_tables(tables: list[list[list[str]]]) -> str:
        """Convert table data to pipe-delimited text for embedding."""
        lines = []
        for table in tables:
            for row in table:
                lines.append(" | ".join(cell for cell in row))
            lines.append("")  # blank line between tables
        return "\n".join(lines)
