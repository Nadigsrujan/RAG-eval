"""Tests for generation components — prompts, context building, citations."""

import pytest

from generation.citation import Citation, CitationExtractor
from generation.context_builder import ContextBuilder
from generation.prompts import PromptManager
from ingestion.chunker import Chunk
from retrieval.bm25_retriever import ScoredChunk

# ── PromptManager Tests ────────────────────────────────────────


class TestPromptManager:
    def test_v1_format(self):
        pm = PromptManager(version="v1")
        prompt = pm.format("What is X?", "Evidence about X.")
        assert "What is X?" in prompt
        assert "Evidence about X." in prompt

    def test_v2_format(self):
        pm = PromptManager(version="v2")
        prompt = pm.format("What is Y?", "Evidence about Y.")
        assert "What is Y?" in prompt
        assert "comprehensive" in prompt.lower()

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt version"):
            PromptManager(version="v99")

    def test_format_as_messages(self):
        pm = PromptManager(version="v1")
        messages = pm.format_as_messages("Question?", "Context.")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ── ContextBuilder Tests ───────────────────────────────────────


class TestContextBuilder:
    def _make_scored_chunks(self) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="c1", text="First chunk content.",
                    document="paper.pdf", page=3, start_char=0, end_char=20,
                ),
                score=0.95,
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="c2", text="Second chunk content.",
                    document="paper.pdf", page=5, start_char=0, end_char=21,
                ),
                score=0.82,
            ),
        ]

    def test_build_includes_source_tags(self):
        chunks = self._make_scored_chunks()
        context = ContextBuilder.build(chunks)
        assert "[Source: paper.pdf, Page 3]" in context
        assert "[Source: paper.pdf, Page 5]" in context

    def test_build_empty_chunks(self):
        context = ContextBuilder.build([])
        assert "No relevant information" in context

    def test_source_list_deduplication(self):
        chunks = self._make_scored_chunks()
        # Add a duplicate (same doc, same page)
        chunks.append(ScoredChunk(
            chunk=Chunk(
                chunk_id="c3", text="Another chunk.",
                document="paper.pdf", page=3, start_char=20, end_char=40,
            ),
            score=0.70,
        ))
        sources = ContextBuilder.get_source_list(chunks)
        # paper.pdf page 3 should appear only once
        page3_sources = [s for s in sources if s["page"] == 3]
        assert len(page3_sources) == 1


# ── CitationExtractor Tests ───────────────────────────────────


class TestCitationExtractor:
    def test_extract_citations(self):
        answer = (
            "The model uses AdamW [Source: paper.pdf, Page 4]. "
            "It was trained on CIFAR-10 [Source: paper.pdf, Page 5]."
        )
        citations = CitationExtractor.extract(answer)
        assert len(citations) == 2
        assert citations[0].document == "paper.pdf"
        assert citations[0].page == 4
        assert citations[1].page == 5

    def test_extract_no_citations(self):
        answer = "The model uses AdamW and was trained on CIFAR-10."
        citations = CitationExtractor.extract(answer)
        assert len(citations) == 0

    def test_validate_valid_citation(self):
        citations = [Citation(document="paper.pdf", page=4, raw_text="[Source: paper.pdf, Page 4]")]
        chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="c1", text="Uses AdamW optimizer.",
                    document="paper.pdf", page=4, start_char=0, end_char=20,
                ),
                score=0.9,
            )
        ]
        validated = CitationExtractor.validate(citations, chunks)
        assert len(validated) == 1
        assert validated[0].is_valid is True

    def test_validate_invalid_citation(self):
        citations = [Citation(document="paper.pdf", page=99, raw_text="[Source: paper.pdf, Page 99]")]
        chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="c1", text="Some content.",
                    document="paper.pdf", page=4, start_char=0, end_char=12,
                ),
                score=0.9,
            )
        ]
        validated = CitationExtractor.validate(citations, chunks)
        assert validated[0].is_valid is False

    def test_citation_score(self):
        from generation.citation import ValidatedCitation
        validated = [
            ValidatedCitation(citation=Citation("a.pdf", 1, ""), is_valid=True),
            ValidatedCitation(citation=Citation("a.pdf", 2, ""), is_valid=True),
            ValidatedCitation(citation=Citation("a.pdf", 3, ""), is_valid=False),
        ]
        score = CitationExtractor.citation_score(validated)
        assert score == pytest.approx(2 / 3, abs=0.01)


# ── Generator Tests ───────────────────────────────────────────


class TestGenerator:
    def test_alias_normalization(self):
        from generation.llm import Generator
        g1 = Generator(model_name="Qwen3-4B-Instruct-2507")
        assert g1.model_name == "Qwen/Qwen3-4B-Instruct-2507"

        g2 = Generator(model_name="flan-t5")
        assert g2.model_name == "google/flan-t5-base"

        g3 = Generator(model_name="custom/my-model")
        assert g3.model_name == "custom/my-model"

    def test_causal_lm_reasoning_and_stop_token_extraction(self):
        from unittest.mock import MagicMock

        from generation.llm import Generator

        gen = Generator(model_name="Qwen/Qwen3-4B-Instruct-2507")
        gen._task = "text-generation"

        mock_tok = MagicMock()
        mock_tok.encode.side_effect = lambda x, **kw: MagicMock(shape=(1, 10)) if kw.get("return_tensors") else [1] * len(x.split())
        mock_tok.decode.side_effect = lambda ids, **kw: "decoded"
        mock_tok.pad_token_id = 0
        gen._tokenizer = mock_tok

        mock_pipe = MagicMock()
        mock_pipe.return_value = [{
            "generated_text": "<think>\nThinking about the context...\n</think>\nThe answer is 42.<|im_end|>"
        }]
        gen._pipeline = mock_pipe

        res = gen.generate("What is the meaning of life?")
        assert res.answer == "The answer is 42."
        assert res.metadata.get("reasoning") == "Thinking about the context..."
        assert res.model_name == "Qwen/Qwen3-4B-Instruct-2507"

    def test_groq_generation(self):
        from unittest.mock import MagicMock

        from generation.llm import Generator

        gen = Generator(model_name="llama-3.3-70b-versatile", provider="groq", api_key="gsk_test")
        assert gen.is_groq is True

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Groq says hello!"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        gen._groq_client = mock_client
        res = gen.generate("Hello?")
        assert res.answer == "Groq says hello!"
        assert res.input_tokens == 15
        assert res.output_tokens == 5
        assert "groq/llama-3.3-70b-versatile" in res.model_name

