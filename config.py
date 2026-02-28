import os
from dataclasses import dataclass

def _load_api_key(key_name: str = "keys.txt"):
    """Load API key from environment variable or keys.txt file"""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key
    
    # Try to load from keys.txt
    try:
        with open(key_name, "r") as f:
            api_key = f.read().strip()
            if api_key:
                return api_key
    except FileNotFoundError:
        pass
    
    return ""

@dataclass
class RAGConfig:
    # Paths
    data_dir: str = "data"
    index_dir: str = "index"
    index_file: str = "faiss_index.bin"
    metadata_file: str = "metadata.json"

    # Chunking
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 100

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 5

    # Gemini
    gemini_api_key: str = _load_api_key()
    gemini_model: str = "gemini-2.5-flash"

config = RAGConfig()
