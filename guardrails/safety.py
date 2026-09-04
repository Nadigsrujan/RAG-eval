"""
Guardrails — safety checks for the RAG pipeline.

Prevents hallucination, prompt injection, and unbounded responses.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Common prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+.*?(instructions|prompts|rules)",
    r"forget\s+.*?(everything|all|previous|instructions)",
    r"you\s+are\s+now\s+",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"pretend\s+you\s+(are|were)",
    r"do\s+not\s+follow\s+(the|your)\s+(instructions|rules)",
    r"override\s+(your|the)\s+(instructions|rules|prompt)",
    r"new\s+instructions?\s*:",
    r"jailbreak",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


class SafetyGuard:
    """
    Safety checks for RAG pipeline inputs and outputs.

    Checks:
      1. Query injection detection
      2. Retrieval confidence threshold (no hallucination on empty evidence)
      3. Response length bounds
      4. PII pattern detection in responses
    """

    def __init__(
        self,
        min_retrieval_score: float = 0.1,
        max_response_tokens: int = 500,
        min_query_length: int = 3,
        max_query_length: int = 1000,
    ):
        self.min_retrieval_score = min_retrieval_score
        self.max_response_tokens = max_response_tokens
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length

    def check_query(self, query: str) -> tuple[bool, str]:
        """
        Validate a user query for safety.

        Returns:
            (is_safe, reason) tuple.
        """
        if len(query.strip()) < self.min_query_length:
            return False, "Query is too short."

        if len(query) > self.max_query_length:
            return False, "Query exceeds maximum length."

        if self.detect_prompt_injection(query):
            logger.warning("Prompt injection detected: %s...", query[:100])
            return False, "Query contains potentially unsafe patterns."

        return True, "OK"

    def check_retrieval_confidence(
        self, scores: list[float]
    ) -> tuple[bool, str]:
        """
        Check if retrieved evidence meets the confidence threshold.

        If no chunk scores above the threshold, the system should refuse
        to answer rather than hallucinate.

        Returns:
            (has_evidence, reason) tuple.
        """
        if not scores:
            return False, "No evidence was retrieved."

        max_score = max(scores)
        if max_score < self.min_retrieval_score:
            return False, (
                f"No sufficiently relevant evidence found "
                f"(best score: {max_score:.3f}, threshold: {self.min_retrieval_score})."
            )

        return True, "OK"

    def check_response(self, response: str) -> tuple[bool, str]:
        """
        Validate a generated response for safety.

        Returns:
            (is_safe, reason) tuple.
        """
        word_count = len(response.split())
        if word_count > self.max_response_tokens:
            return False, f"Response exceeds maximum length ({word_count} > {self.max_response_tokens})."

        if self._contains_pii(response):
            return False, "Response may contain personally identifiable information."

        return True, "OK"

    @staticmethod
    def detect_prompt_injection(text: str) -> bool:
        """Check if text contains prompt injection patterns."""
        for pattern in _COMPILED_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def _contains_pii(text: str) -> bool:
        """Basic PII detection (emails, phone numbers, SSNs)."""
        # Email
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
            return True
        # US SSN
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
            return True
        # US phone
        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text):
            return True
        return False

    def get_refusal_message(self, reason: str) -> str:
        """Generate a safe refusal message."""
        return (
            f"I cannot provide an answer. {reason} "
            "Please try rephrasing your question or uploading more relevant documents."
        )
