"""
PDF Parser — extracts text, tables, and metadata from PDF documents.

Supports two backends:
  - pdfplumber (default): better table extraction, layout awareness
  - PyPDF2: faster, lighter, text-only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Represents a single extracted page from a PDF."""

    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip()) or bool(self.tables)


@dataclass
class ParsedDocument:
    """Complete parsed PDF document."""

    filename: str
    path: str
    pages: list[Page]
    total_pages: int
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class PDFParser:
    """
    Extracts structured content from PDF files.

    Args:
        backend: Which PDF library to use ('pdfplumber' or 'pypdf2').
    """

    def __init__(self, backend: Literal["pdfplumber", "pypdf2"] = "pdfplumber"):
        self.backend = backend

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF file and return structured content."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        logger.info("Parsing %s with %s backend", pdf_path.name, self.backend)

        if self.backend == "pdfplumber":
            return self._parse_pdfplumber(pdf_path)
        elif self.backend == "pypdf2":
            return self._parse_pypdf2(pdf_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def parse_many(self, pdf_paths: list[str | Path]) -> list[ParsedDocument]:
        """Parse multiple PDFs."""
        documents = []
        for path in pdf_paths:
            try:
                doc = self.parse(path)
                documents.append(doc)
            except Exception as e:
                logger.error("Failed to parse %s: %s", path, e)
        return documents

    # ── pdfplumber backend ─────────────────────────────────────────

    def _parse_pdfplumber(self, pdf_path: Path) -> ParsedDocument:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            doc_metadata = pdf.metadata or {}
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = self._extract_tables_plumber(page)
                pages.append(
                    Page(
                        page_number=i + 1,
                        text=self._clean_text(text),
                        tables=tables,
                        metadata={"width": page.width, "height": page.height},
                    )
                )

        return ParsedDocument(
            filename=pdf_path.name,
            path=str(pdf_path),
            pages=pages,
            total_pages=len(pages),
            metadata=doc_metadata,
        )

    @staticmethod
    def _extract_tables_plumber(page) -> list[list[list[str]]]:
        """Extract tables from a pdfplumber page."""
        try:
            raw_tables = page.extract_tables()
            if not raw_tables:
                return []
            cleaned = []
            for table in raw_tables:
                cleaned_table = []
                for row in table:
                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                    cleaned_table.append(cleaned_row)
                cleaned.append(cleaned_table)
            return cleaned
        except Exception:
            return []

    # ── PyPDF2 backend ─────────────────────────────────────────────

    def _parse_pypdf2(self, pdf_path: Path) -> ParsedDocument:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = []
        doc_metadata = {}

        if reader.metadata:
            doc_metadata = {
                k: str(v) for k, v in reader.metadata.items() if v
            }

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
                logger.warning("Could not extract text from page %d of %s", i + 1, pdf_path.name)

            pages.append(
                Page(
                    page_number=i + 1,
                    text=self._clean_text(text),
                    tables=[],  # PyPDF2 has no native table extraction
                )
            )

        return ParsedDocument(
            filename=pdf_path.name,
            path=str(pdf_path),
            pages=pages,
            total_pages=len(pages),
            metadata=doc_metadata,
        )

    # ── Text cleaning ──────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and remove control characters."""
        import re

        # Replace multiple whitespace/newlines with single space
        text = re.sub(r"\s+", " ", text)
        # Remove control chars except newline
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()
