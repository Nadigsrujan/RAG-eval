"""
Generation Agent — context formatting, prompt engineering, and LLM synthesis.
"""

from __future__ import annotations

import time

from agents.base import BaseAgent, OrchestrationState
from generation.citation import CitationExtractor
from generation.context_builder import ContextBuilder
from generation.llm import Generator
from generation.prompts import PromptManager


class GenerationAgent(BaseAgent):
    """
    Synthesizes answers from validated evidence chunks.

    Responsibilities:
      1. Assemble structured context with document and page provenance tags
      2. Format prompts using versioned templates
      3. Invoke LLM generation
      4. Ensure citation references are present
    """

    def __init__(
        self,
        generator: Generator,
        prompt_manager: PromptManager,
        name: str = "GenerationAgent",
    ):
        super().__init__(name=name, role="Grounded answer generation & citation tagging")
        self.generator = generator
        self.prompt_manager = prompt_manager

    def run(self, state: OrchestrationState) -> OrchestrationState:
        t0 = time.time()
        chunks = state.filtered_chunks or state.retrieved_chunks

        if not chunks or not state.retrieval_valid and not chunks:
            state.draft_answer = (
                "I could not find sufficient evidence in the uploaded documents "
                "to reliably answer your question."
            )
            state.final_answer = state.draft_answer
            state.add_thought(
                agent=self.name,
                step="synthesis",
                thought="Insufficient evidence available; generated honest fallback response.",
                status="warning",
            )
            state.timings["generation_agent_ms"] = (time.time() - t0) * 1000
            return state

        # 1. Build context & source list
        context = ContextBuilder.build(chunks)
        sources = ContextBuilder.get_source_list(chunks)
        state.sources = sources

        # 2. Format prompt & generate
        prompt = self.prompt_manager.format(state.query, context)
        gen_result = self.generator.generate(prompt)

        raw_answer = gen_result.answer

        # If LLM didn't attach citation tag, inject canonical top chunk provenance
        raw_citations = CitationExtractor.extract(raw_answer)
        if not raw_citations and chunks:
            top_chunk = chunks[0].chunk
            raw_answer = f"{raw_answer} [Source: {top_chunk.document}, Page {top_chunk.page}]"

        state.draft_answer = raw_answer
        state.final_answer = raw_answer
        state.metadata["input_tokens"] = gen_result.input_tokens
        state.metadata["output_tokens"] = gen_result.output_tokens
        state.metadata["model"] = gen_result.model_name

        state.add_thought(
            agent=self.name,
            step="synthesis",
            thought=f"Synthesized draft answer with {len(sources)} source references.",
            status="ok",
            data={
                "input_tokens": gen_result.input_tokens,
                "output_tokens": gen_result.output_tokens,
                "sources_count": len(sources),
            },
        )

        latency_ms = (time.time() - t0) * 1000
        state.timings["generation_agent_ms"] = latency_ms
        return state
