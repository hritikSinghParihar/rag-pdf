import json
import logging
from typing import List, Dict, Any, Tuple
import os

logger = logging.getLogger(__name__)

class DocumentStructure:
    """Represents a structured document extracted from OCR."""
    def __init__(self, title: str, sections: List[Dict[str, Any]], metadata: Dict[str, Any]):
        self.title = title
        self.sections = sections
        self.metadata = metadata

    def get_chunks(self, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """Perform layout-aware chunking."""
        chunks = []
        for i, section in enumerate(self.sections):
            heading = section.get("heading", "")
            content = section.get("content", "")
            section_type = section.get("type", "text")
            
            # Metadata for each chunk
            chunk_metadata = {
                "source_title": self.title,
                "section_index": i,
                "heading": heading,
                "type": section_type,
                **self.metadata
            }

            if section_type == "table":
                # Tables are best kept whole if small, or row-by-row if large
                table_data = section.get("table_data", [])
                table_str = "\n".join([" | ".join(map(str, row)) for row in table_data])
                chunks.append({
                    "text": f"Table: {heading}\n{table_str}",
                    "metadata": chunk_metadata
                })
            else:
                # Text sections
                if len(content) > max_chunk_size:
                    # Simple split for now, but layout-aware
                    # You could use sentence splitting here
                    words = content.split()
                    for j in range(0, len(words), max_chunk_size):
                        sub_content = " ".join(words[j:j+max_chunk_size])
                        chunks.append({
                            "text": f"Section: {heading}\n{sub_content}",
                            "metadata": chunk_metadata
                        })
                else:
                    chunks.append({
                        "text": f"Section: {heading}\n{content}",
                        "metadata": chunk_metadata
                    })
        return chunks

class StructureBuilder:
    @staticmethod
    def build_from_ocr(ocr_data: Dict[str, Any]) -> DocumentStructure:
        """Convert OCR JSON response into DocumentStructure."""
        title = ocr_data.get("title", "Unknown Document")
        sections = ocr_data.get("sections", [])
        metadata = ocr_data.get("metadata", {})
        return DocumentStructure(title, sections, metadata)
