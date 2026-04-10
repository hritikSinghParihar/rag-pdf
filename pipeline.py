import os
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ocr_modules import VisionOCR
from structure_builder import StructureBuilder
from vector_store import get_vector_store
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.ocr = VisionOCR()
        self.embedding_model = SentenceTransformer(config.embedding_model_name)
        # Assuming dimension from the model (MiniLM-L6-v2 is 384)
        self.vector_store = get_vector_store(384) 

    def process_image(self, image_path: str):
        """End-to-end processing of an image into the vector store."""
        logger.info(f"Processing image: {image_path}")
        
        # 1. OCR & Layout Extraction
        ocr_data = self.ocr.extract_structured_data(image_path)
        if "error" in ocr_data:
            logger.error(f"OCR failed: {ocr_data['error']}")
            return
        
        # 2. Structure Building & Chunking
        doc_structure = StructureBuilder.build_from_ocr(ocr_data)
        chunks = doc_structure.get_chunks()
        
        logger.info(f"Extracted {len(chunks)} chunks from {image_path}")
        
        # 3. Embedding & Storage
        texts = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            m = c["metadata"].copy()
            m["text"] = c["text"] # Ensure text is stored in metadata
            metadatas.append(m)
        
        embeddings = self.embedding_model.encode(texts)
        self.vector_store.add(np.array(embeddings), metadatas)
        self.vector_store.save()
        
        logger.info(f"Successfully ingested {image_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path>")
        sys.exit(1)
    
    pipeline = RAGPipeline()
    pipeline.process_image(sys.argv[1])
