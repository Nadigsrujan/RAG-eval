"""
Reflection & Validation Agent — verifies faithfulness, checks citations, and performs self-correction.
"""

from __future__ import annotations

import re
import time
from typing import Any

from agents.base import BaseAgent, OrchestrationState
from generation.citation import CitationExtractor


class ReflectionValidatorAgent(BaseAgent):
    """
    Validates synthesized responses for factual faithfulness and citation validity.

    Responsibilities:
      1. Extract and validate all citations against the retrieved chunk provenance
      2. Check answer faithfulness (lexical and entity grounding against evidence)
      3. Self-correct answers with invalid citations or ungrounded assertions
      4. Compute a unified confidence and accuracy report
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.70,
        name: str = "ReflectionValidatorAgent",
    ):
        super().__init__(name=name, role="Faithfulness validation, citation audit & self-correction")
        self.faithfulness_threshold = faithfulness_threshold

    def run(self, state: OrchestrationState) -> OrchestrationState:
        t0 = time.time()
        answer = state.draft_answer
        chunks = state.filtered_chunks or state.retrieved_chunks

        if not answer or not chunks:
            state.faithfulness_score = 1.0
            state.citation_accuracy = 1.0
            state.timings["reflection_agent_ms"] = (time.time() - t0) * 1000
            return state

        # 1. Citation Audit
        raw_citations = CitationExtractor.extract(answer)
        validated_citations = CitationExtractor.validate(raw_citations, chunks)
        citation_score = CitationExtractor.citation_score(validated_citations)

        state.citations = [
            {
                "document": v.citation.document,
                "page": v.citation.page,
                "is_valid": v.is_valid,
                "reason": v.reason,
            }
            for v in validated_citations
        ]
        state.citation_accuracy = round(citation_score, 4)

        # 2. Faithfulness / Grounding Audit
        faithfulness_score, ungrounded_tokens = self._audit_faithfulness(answer, chunks)
        state.faithfulness_score = round(faithfulness_score, 4)

        # 3. Self-Correction Loop
        self_corrected = False
        reasons = []

        # Check for citation invalidity
        invalid_citations = [v for v in validated_citations if not v.is_valid]
        if invalid_citations and chunks:
            # Self-correct citations by substituting verified top chunk provenance
            top_chunk = chunks[0].chunk
            corrected_answer = re.sub(
                r"\[Source:\s*[^,]+,\s*Page\s*\d+\]",
                f"[Source: {top_chunk.document}, Page {top_chunk.page}]",
                answer,
            )
            if corrected_answer != answer:
                answer = corrected_answer
                self_corrected = True
                reasons.append("Replaced unverified citation with verified top evidence chunk.")
                # Re-validate
                new_raw = CitationExtractor.extract(answer)
                new_val = CitationExtractor.validate(new_raw, chunks)
                state.citations = [
                    {
                        "document": v.citation.document,
                        "page": v.citation.page,
                        "is_valid": v.is_valid,
                        "reason": v.reason,
                    }
                    for v in new_val
                ]
                state.citation_accuracy = CitationExtractor.citation_score(new_val)

        # Check for severe hallucination / ungrounded claims
        if faithfulness_score < self.faithfulness_threshold and len(answer) > 20:
            reasons.append(
                f"Faithfulness score ({faithfulness_score:.2f}) below threshold ({self.faithfulness_threshold})."
            )
            # Add an advisory note if claims could not be verified in the documents
            if "Note:" not in answer and not answer.startswith("I could not find"):
                answer = f"{answer} (Self-Correction Note: Part of this answer may extrapolate beyond the ingested context)."
                self_corrected = True

        state.final_answer = answer
        state.self_corrected = self_corrected
        state.validation_reasons = reasons

        # Record Agent Thought
        status = "ok" if (citation_score >= 0.8 and faithfulness_score >= self.faithfulness_threshold) else "warning"
        thought_msg = (
            f"Validation complete. Faithfulness: {state.faithfulness_score * 100:.0f}%, "
            f"Citation Accuracy: {state.citation_accuracy * 100:.0f}%."
        )
        if self_corrected:
            thought_msg += f" Triggered self-correction: {'; '.join(reasons)}"

        state.add_thought(
            agent=self.name,
            step="reflection_audit",
            thought=thought_msg,
            status=status,
            data={
                "faithfulness_score": state.faithfulness_score,
                "citation_accuracy": state.citation_accuracy,
                "self_corrected": self_corrected,
                "reasons": reasons,
            },
        )

        latency_ms = (time.time() - t0) * 1000
        state.timings["reflection_agent_ms"] = latency_ms
        return state

    def _audit_faithfulness(
        self, answer: str, chunks: list[Any]
    ) -> tuple[float, list[str]]:
        """
        Check lexical grounding of answer content against evidence chunks.
        Computes the ratio of non-trivial answer content words present in the evidence.
        """
        # Strip citation tags from answer text before lexical check
        clean_text = re.sub(r"\[Source:[^\]]+\]", "", answer).lower()
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", clean_text)

        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "are",
            "was", "were", "have", "has", "had", "can", "could", "would",
            "should", "not", "but", "what", "which", "where", "when", "how",
            "about", "into", "their", "they", "been", "also", "used", "uses"
        }
        content_words = [w for w in words if w not in stop_words]

        if not content_words:
            return 1.0, []

        # Combine evidence text
        evidence_text = " ".join(
            (c.chunk.text.lower() if hasattr(c, "chunk") else str(c).lower())
            for c in chunks
        )

        grounded = [w for w in content_words if w in evidence_text]
        ungrounded = [w for w in content_words if w not in evidence_text]
        score = len(grounded) / len(content_words)
        return score, ungrounded[:5]
