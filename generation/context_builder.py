"""
Context Builder — constructs LLM-ready context from retrieved chunks.

Formats chunks with source attribution markers so the LLM can cite
specific documents and pages in its response.
"""

from __future__ import annotations

import logging
from retrieval.bm25_retriever import ScoredChunk

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Formats retrieved chunks into a context string for the LLM.

    Each chunk is tagged with its source document and page number
    so the LLM can produce citations like [Source: doc.pdf, Page 7].
    """

    @staticmethod
    def build(chunks: list[ScoredChunk], max_tokens: int = 250) -> str:
        """
        Build a formatted context string from scored chunks.

        Args:
            chunks: Ranked list of retrieved chunks.
            max_tokens: Approximate token budget for the context.

        Returns:
            Formatted context string with source markers.
        """
        if not chunks:
            return "No relevant information was found in the documents."

        context_parts = []
        total_words = 0

        for i, scored_chunk in enumerate(chunks, 1):
            chunk = scored_chunk.chunk
            source_tag = f"[Source: {chunk.document}, Page {chunk.page}]"
            block = f"--- Evidence {i} {source_tag} ---\n{chunk.text}\n"

            word_count = len(block.split())
            if total_words + word_count > max_tokens:
                break

            context_parts.append(block)
            total_words += word_count

        context = "\n".join(context_parts)
        logger.debug("Built context: %d chunks, ~%d words", len(context_parts), total_words)
        return context

    @staticmethod
    def get_source_list(chunks: list[ScoredChunk]) -> list[dict]:
        """
        Extract a deduplicated list of source documents and pages.

        Returns:
            List of {"document": str, "page": int, "score": float} dicts.
        """
        sources = []
        seen = set()

        for scored_chunk in chunks:
            chunk = scored_chunk.chunk
            key = (chunk.document, chunk.page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": chunk.document,
                    "page": chunk.page,
                    "score": round(scored_chunk.score, 4),
                })

        return sources
