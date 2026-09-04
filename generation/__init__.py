"""Generation package — LLM generation, prompts, context building, and citations."""

from generation.context_builder import ContextBuilder
from generation.prompts import PromptManager
from generation.llm import Generator, GenerationResult
from generation.citation import CitationExtractor, Citation, ValidatedCitation

__all__ = [
    "ContextBuilder",
    "PromptManager",
    "Generator",
    "GenerationResult",
    "CitationExtractor",
    "Citation",
    "ValidatedCitation",
]
