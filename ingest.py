import os
import logging
from typing import List, Dict, Any
import pymupdf4llm

from config import config

logger = logging.getLogger(__name__)

def extract_text_from_pdf(path: str) -> List[Dict[str, Any]]:
    """Extract text per page as Markdown with metadata."""
    md_pages = pymupdf4llm.to_markdown(path, page_chunks=True)
    pages = []
    # pymupdf4llm.to_markdown(page_chunks=True) returns a list of dictionaries.
    for i, page_data in enumerate(md_pages):
        pages.append(
            {
                "text": page_data["text"],
                "page": i + 1,
            }
        )
    return pages

def ingest_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Ingest multiple PDFs and return list of page-level dicts."""
    all_pages = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logger.warning(f"File not found: {pdf_path}")
            continue
        logger.info(f"Ingesting {pdf_path}")
        pages = extract_text_from_pdf(pdf_path)
        for p in pages:
            p["source"] = os.path.basename(pdf_path)
        all_pages.extend(pages)
    return all_pages
