import os
import logging
from ingest import ingest_files
from vector_store import get_vector_store
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unified_ingest():
    # 1. Prepare test files
    test_files = [
        "data/wisipay_financial_report_2024.txt", # Standard TXT
        "test_ocr_image.png",                     # Image OCR
    ]
    
    # Check if files exist
    test_files = [f for f in test_files if os.path.exists(f)]
    
    if not test_files:
        logger.error("No test files found!")
        return

    logger.info(f"Starting unified ingestion for: {test_files}")
    
    # 2. Run ingestion
    all_chunks = ingest_files(test_files)
    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    
    for i, chunk in enumerate(all_chunks[:5]):
        logger.info(f"Chunk {i} from {chunk.get('source')} (Page {chunk.get('page')}): {chunk.get('text')[:100]}...")

    # 3. Verify embedding & storage
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = get_vector_store(384)
    
    texts = [c["text"] for c in all_chunks]
    metadatas = [c for c in all_chunks] # Already formatted
    
    embeddings = embedding_model.encode(texts)
    vector_store.add(np.array(embeddings), metadatas)
    vector_store.save()
    
    logger.info("Verification complete. Vector store updated.")

if __name__ == "__main__":
    test_unified_ingest()
