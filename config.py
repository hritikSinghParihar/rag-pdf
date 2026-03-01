import os
from dataclasses import dataclass

def _load_api_keys():
    """Load API keys from environment variables or keys.txt file.
    
    Returns a dict with 'gemini' and 'openai' keys.
    """
    keys = {"gemini": "", "openai": ""}
    
    # Try environment variables first
    keys["gemini"] = os.getenv("GEMINI_API_KEY", "").strip()
    keys["openai"] = os.getenv("OPENAI_API_KEY", "").strip()
    
    # If not in env, try to load from keys.txt
    if not keys["gemini"] or not keys["openai"]:
        try:
            with open("keys.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        keys["gemini"] = line.split("=", 1)[1].strip()
                    elif line.startswith("OPENAI_API_KEY="):
                        keys["openai"] = line.split("=", 1)[1].strip()
        except FileNotFoundError:
            pass
    
    return keys

_loaded_keys = _load_api_keys()

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
    gemini_api_key: str = _loaded_keys["gemini"]
    gemini_model: str = "gemini-2.5-flash"

    # OpenAI
    openai_api_key: str = _loaded_keys["openai"]
    openai_model: str = "gpt-4o-mini"

    # Provider selection ("gemini" or "openai")
    provider: str = "gemini"

config = RAGConfig()
