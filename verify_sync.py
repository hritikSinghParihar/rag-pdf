import os
import numpy as np
from config import config
from utils import get_all_supported_files
from ingest import ingest_files
from embed import chunk_pages, embed_texts, get_embedding_model
from vector_store import get_vector_store

def test_sync():
    print(f"Scanning directory: {config.scrapper_dir}")
    scrapper_files = get_all_supported_files(config.scrapper_dir)
    print(f"Found {len(scrapper_files)} files in scraper directory.")
    
    if not scrapper_files:
        print("No files discovered. Aborting test.")
        return

    # Load existing store
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    store = get_vector_store(dim)
    
    indexed_sources = {m.get("source") for m in store.metadata} if store.metadata else set()
    new_files = [f for f in scrapper_files if f not in indexed_sources]
    
    print(f"Already indexed: {len(indexed_sources)}")
    print(f"New files to index: {len(new_files)}")
    
    if not new_files:
        print("No new files to index. Sync works (already up to date).")
        return

    # Process a few files for testing
    test_files = new_files[:2]
    print(f"Testing with first 2 new files: {test_files}")
    
    pages = ingest_files(test_files)
    chunks = chunk_pages(pages)
    print(f"Extracted {len(chunks)} chunks.")
    
    if chunks:
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        for meta, text in zip(metadatas, texts):
            meta["text"] = text

        embeddings = embed_texts(texts)
        embeddings = np.array(embeddings)
        
        if store.index is None:
            store.create_new()
        
        store.add(embeddings, metadatas)
        print("Successfully added chunks to vector store.")
        # We won't save in the test to avoid messing up the actual index
        # store.save()
        print("Test passed: Sync logic functional.")
    else:
        print("Test failed: No chunks extracted.")

if __name__ == "__main__":
    test_sync()
