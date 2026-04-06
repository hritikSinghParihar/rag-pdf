import os
import logging
from pathlib import Path
from typing import List, Dict, Any

import pymupdf4llm
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from config import config

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".html", ".htm", ".docx"}

# ─────────────────────────────────────────────
# Per-format loaders
# ─────────────────────────────────────────────

def load_pdf(path: str) -> List[Dict[str, Any]]:
    """Extract text per page as Markdown using PyMuPDF4LLM."""
    md_pages = pymupdf4llm.to_markdown(path, page_chunks=True)
    return [
        {"text": page_data["text"], "page": i + 1}
        for i, page_data in enumerate(md_pages)
    ]


def load_txt(path: str) -> List[Dict[str, Any]]:
    """Read plain-text file and split into virtual pages (~3 000 chars each)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    page_size = 3000
    pages = []
    for i, start in enumerate(range(0, max(len(text), 1), page_size)):
        chunk = text[start : start + page_size].strip()
        if chunk:
            pages.append({"text": chunk, "page": i + 1})
    return pages


def load_html(path: str) -> List[Dict[str, Any]]:
    """Parse HTML with BeautifulSoup, strip boilerplate, return virtual pages."""
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    # Remove noisy/non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)

    page_size = 3000
    pages = []
    for i, start in enumerate(range(0, max(len(text), 1), page_size)):
        chunk = text[start : start + page_size].strip()
        if chunk:
            pages.append({"text": chunk, "page": i + 1})
    return pages


def load_docx(path: str) -> List[Dict[str, Any]]:
    """
    Extract text from DOCX paragraphs and tables; group into virtual pages
    of ~20 paragraphs each.
    """
    doc = DocxDocument(path)

    # Collect all text blocks in document order
    blocks: List[str] = []

    # Paragraphs
    for para in doc.paragraphs:
        txt = para.text.strip()
        if txt:
            blocks.append(txt)

    # Tables (each cell becomes its own block)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                blocks.append(row_text)

    # Group into virtual pages
    page_size = 20
    pages = []
    for i in range(0, max(len(blocks), 1), page_size):
        chunk = "\n".join(blocks[i : i + page_size]).strip()
        if chunk:
            pages.append({"text": chunk, "page": i // page_size + 1})
    return pages


# ─────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────

_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".html": load_html,
    ".htm": load_html,
    ".docx": load_docx,
}


def ingest_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Ingest multiple files of any supported type and return a unified list of
    page-level dicts: [{text, page, source}, ...].
    """
    all_pages: List[Dict[str, Any]] = []
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning("File not found, skipping: %s", path)
            continue

        ext = Path(path).suffix.lower()
        loader = _LOADERS.get(ext)
        if loader is None:
            logger.warning("Unsupported file type '%s', skipping: %s", ext, path)
            continue

        try:
            logger.info("Ingesting %s", path)
            pages = loader(path)
            for p in pages:
                p["source"] = os.path.basename(path)
            all_pages.extend(pages)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to ingest %s: %s", path, exc, exc_info=True)

    return all_pages


# ─────────────────────────────────────────────
# Backward-compatibility alias
# ─────────────────────────────────────────────

def ingest_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Legacy alias — delegates to ingest_files."""
    return ingest_files(pdf_paths)
