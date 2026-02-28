import os
import logging
from typing import List, Dict, Any
import fitz  # PyMuPDF

from config import config

logger = logging.getLogger(__name__)

def extract_text_from_pdf(path: str) -> List[Dict[str, Any]]:
    """Extract text per page with metadata."""
    doc = fitz.open(path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append(
            {
                "text": text,
                "page": page_num + 1,
            }
        )
    doc.close()
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
