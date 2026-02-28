import logging
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from config import config

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.embedding_model_name)
    return _tokenizer

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.embedding_model_name}")
        _model = SentenceTransformer(config.embedding_model_name)
    return _model

def chunk_text(
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> List[str]:
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size_tokens, n)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == n:
            break
        start = max(0, end - chunk_overlap_tokens)
    return chunks

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    if chunk_size_tokens is None:
        chunk_size_tokens = config.chunk_size_tokens
    if chunk_overlap_tokens is None:
        chunk_overlap_tokens = config.chunk_overlap_tokens

    chunks = []
    for page in pages:
        page_text = page["text"]
        base_meta = {
            "source": page.get("source"),
            "page": page.get("page"),
        }
        page_chunks = chunk_text(page_text, chunk_size_tokens, chunk_overlap_tokens)
        for i, ch in enumerate(page_chunks):
            meta = base_meta.copy()
            meta["chunk_id"] = f"{meta['source']}_p{meta['page']}_c{i}"
            chunks.append(
                {
                    "text": ch,
                    "metadata": meta,
                }
            )
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings
