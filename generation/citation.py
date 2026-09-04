"""
Citation Extractor & Validator — extracts and verifies source citations
from LLM-generated answers.

This is the "citation correctness" evaluation layer. The LLM claims
its answer comes from specific sources; this module checks whether
those sources actually support the claims.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from retrieval.bm25_retriever import ScoredChunk

logger = logging.getLogger(__name__)

# Pattern to match [Source: filename, Page N]
_CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]",
    re.IGNORECASE,
)


@dataclass
class Citation:
    """A citation extracted from the generated answer."""

    document: str
    page: int
    raw_text: str  # the original citation string from the answer


@dataclass
class ValidatedCitation:
    """A citation with its validation result."""

    citation: Citation
    is_valid: bool
    supporting_chunk: Optional[ScoredChunk] = None
    reason: str = ""


class CitationExtractor:
    """Extracts and validates citations from generated answers."""

    @staticmethod
    def extract(answer: str) -> list[Citation]:
        """
        Extract all [Source: filename, Page N] citations from an answer.

        Args:
            answer: LLM-generated answer text.

        Returns:
            List of Citation objects found in the answer.
        """
        citations = []
        for match in _CITATION_PATTERN.finditer(answer):
            document = match.group(1).strip()
            page = int(match.group(2))
            citations.append(
                Citation(
                    document=document,
                    page=page,
                    raw_text=match.group(0),
                )
            )

        logger.debug("Extracted %d citations from answer", len(citations))
        return citations

    @staticmethod
    def validate(
        citations: list[Citation],
        retrieved_chunks: list[ScoredChunk],
    ) -> list[ValidatedCitation]:
        """
        Validate whether extracted citations match retrieved evidence.

        A citation is valid if there exists a retrieved chunk from the
        same document and page.

        Args:
            citations: Citations extracted from the answer.
            retrieved_chunks: Chunks that were retrieved and used as context.

        Returns:
            List of ValidatedCitation with validity flags.
        """
        # Build lookup: (document, page) → list of chunks
        chunk_lookup: dict[tuple[str, int], list[ScoredChunk]] = {}
        for scored_chunk in retrieved_chunks:
            key = (scored_chunk.chunk.document, scored_chunk.chunk.page)
            chunk_lookup.setdefault(key, []).append(scored_chunk)

        results = []
        for citation in citations:
            key = (citation.document, citation.page)
            matching_chunks = chunk_lookup.get(key, [])

            if matching_chunks:
                results.append(
                    ValidatedCitation(
                        citation=citation,
                        is_valid=True,
                        supporting_chunk=matching_chunks[0],
                        reason="Citation matches retrieved evidence.",
                    )
                )
            else:
                results.append(
                    ValidatedCitation(
                        citation=citation,
                        is_valid=False,
                        reason=(
                            f"No retrieved chunk from {citation.document} "
                            f"Page {citation.page}."
                        ),
                    )
                )

        valid_count = sum(1 for v in results if v.is_valid)
        logger.info(
            "Citation validation: %d/%d valid",
            valid_count,
            len(results),
        )
        return results

    @staticmethod
    def citation_score(validated: list[ValidatedCitation]) -> float:
        """
        Compute citation correctness score.

        Returns:
            Fraction of citations that are valid (0.0 to 1.0).
        """
        if not validated:
            return 0.0
        return sum(1 for v in validated if v.is_valid) / len(validated)
