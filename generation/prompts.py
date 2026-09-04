"""
Prompt Manager — versioned prompt templates for RAG generation.

Prompt engineering is critical for RAG quality. Versioning templates
lets us track which prompt produced which evaluation results.
"""

from __future__ import annotations

PROMPT_TEMPLATES = {
    "v1": {
        "system": (
            "Read the following context from documents and answer the question briefly and accurately based on the facts provided."
        ),
        "user": (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    },
    "v2": {
        "system": (
            "You are an expert document analyst. Your task is to provide accurate, "
            "well-cited answers based exclusively on the provided evidence.\n\n"
            "Rules:\n"
            "1. Only use information from the evidence below.\n"
            "2. Cite every factual claim with [Source: filename, Page N].\n"
            "3. If evidence is insufficient, explicitly state what is missing.\n"
            "4. Be concise but thorough."
        ),
        "user": (
            "EVIDENCE:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "Provide a comprehensive answer with citations:"
        ),
    },
}


class PromptManager:
    """
    Manages versioned prompt templates.

    Args:
        version: Which prompt template version to use.
    """

    def __init__(self, version: str = "v1"):
        if version not in PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown prompt version '{version}'. "
                f"Available: {list(PROMPT_TEMPLATES.keys())}"
            )
        self.version = version
        self.template = PROMPT_TEMPLATES[version]

    def format(self, question: str, context: str) -> str:
        """
        Build the final prompt string for the LLM.

        For encoder-decoder models (e.g., FLAN-T5), we combine system +
        user into a single prompt. For decoder-only models, these could
        be separated into system/user messages.

        Args:
            question: The user's question.
            context: Formatted context from ContextBuilder.

        Returns:
            Complete prompt string ready for the LLM.
        """
        system = self.template["system"]
        user = self.template["user"].format(question=question, context=context)
        return f"{system}\n\n{user}"

    def format_as_messages(self, question: str, context: str) -> list[dict]:
        """Format as chat messages (for chat-style models)."""
        return [
            {"role": "system", "content": self.template["system"]},
            {
                "role": "user",
                "content": self.template["user"].format(
                    question=question, context=context
                ),
            },
        ]
