import logging
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    if chunk_size_tokens is None:
        chunk_size_tokens = config.chunk_size_tokens
    if chunk_overlap_tokens is None:
        chunk_overlap_tokens = config.chunk_overlap_tokens

    # Set up splitters
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    tokenizer = get_tokenizer()
    def token_len(text: str) -> int:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        length_function=token_len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for page in pages:
        page_text = page["text"]
        base_meta = {
            "source": page.get("source"),
            "page": page.get("page"),
        }
        
        # 1. Split by Markdown headers (keeps headings intact)
        md_docs = markdown_splitter.split_text(page_text)
        
        # 2. Split larger sections while preserving layout (tables will generally stay intact if they fit in the chunk)
        split_docs = text_splitter.split_documents(md_docs)
        
        for i, split_doc in enumerate(split_docs):
            meta = base_meta.copy()
            # Add header metadata found by MarkdownHeaderTextSplitter
            for key, val in split_doc.metadata.items():
                meta[key] = val
                
            meta["chunk_id"] = f"{meta['source']}_p{meta['page']}_c{i}"
            chunks.append(
                {
                    "text": split_doc.page_content,
                    "metadata": meta,
                }
            )
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings
