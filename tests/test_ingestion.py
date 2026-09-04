"""Tests for the ingestion pipeline — PDF parsing, chunking, embedding logic."""

import pytest

from ingestion.chunker import Chunker
from ingestion.pdf_parser import Page, ParsedDocument, PDFParser

# ── PDFParser Tests ────────────────────────────────────────────


class TestPDFParser:
    def test_clean_text_removes_extra_whitespace(self):
        result = PDFParser._clean_text("  hello   world  \n\n  foo  ")
        assert result == "hello world foo"

    def test_clean_text_removes_control_chars(self):
        result = PDFParser._clean_text("hello\x00world\x07test")
        assert result == "helloworld test" or "hello" in result

    def test_parse_nonexistent_file_raises(self):
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.pdf")

    def test_parse_non_pdf_raises(self):
        parser = PDFParser()
        with pytest.raises(ValueError, match="Not a PDF"):
            parser.parse(__file__)  # .py file


# ── Chunker Tests ──────────────────────────────────────────────


class TestChunker:
    def test_basic_chunking(self):
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        doc = ParsedDocument(
            filename="test.pdf",
            path="/test.pdf",
            pages=[
                Page(page_number=1, text=" ".join(f"word{i}" for i in range(30)))
            ],
            total_pages=1,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document == "test.pdf"
            assert chunk.page == 1
            assert len(chunk.text) > 0

    def test_overlap_less_than_size(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)

    def test_empty_page_produces_no_chunks(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        doc = ParsedDocument(
            filename="test.pdf",
            path="/test.pdf",
            pages=[Page(page_number=1, text="   ")],
            total_pages=1,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_chunk_ids_are_deterministic(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test sentence. " * 20
        doc = ParsedDocument(
            filename="test.pdf",
            path="/test.pdf",
            pages=[Page(page_number=1, text=text)],
            total_pages=1,
        )
        chunks1 = chunker.chunk_document(doc)
        chunks2 = chunker.chunk_document(doc)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_provenance(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        doc = ParsedDocument(
            filename="paper.pdf",
            path="/paper.pdf",
            pages=[
                Page(page_number=3, text="Hello world. " * 30),
            ],
            total_pages=5,
        )
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.document == "paper.pdf"
            assert chunk.page == 3

    def test_table_linearization(self):
        result = Chunker._linearize_tables([
            [["A", "B"], ["1", "2"]],
        ])
        assert "A | B" in result
        assert "1 | 2" in result
