# Local-First RAG System - AI Agent Instructions

## Architecture Overview

This is a **local-first Retrieval-Augmented Generation (RAG)** system over PDFs. The design philosophy prioritizes privacy and offline-first processing—PDFs are chunked and embedded locally; only top-k retrieved chunks plus the user question are sent to OpenAI.

### Core Data Flow
1. **Ingest** (`ingest.py`): Extract text from PDFs using PyMuPDF (fitz), tracking source and page numbers
2. **Chunk & Embed** (`embed.py`): Token-based chunking with overlap, generate embeddings via sentence-transformers
3. **Store** (`vector_store.py`): Persist FAISS index + JSON metadata to disk
4. **Retrieve** (`retrieve.py`): Query embedding lookup in FAISS, return top-k chunks with metadata
5. **Generate** (`generate.py`): Build prompt from chunks, call OpenAI with system instruction to answer only from context

### File Responsibilities
- `config.py`: Single config dataclass (RAGConfig) with paths, embedding model, chunk params, OpenAI settings
- `main.py`: CLI entry point—takes question arg, retrieves chunks, generates answer, prints sources
- `ui.py`: Streamlit dashboard with PDF upload, adjustable chunk/overlap/top-k sliders, and query interface
- `vector_store.py`: FAISSVectorStore class managing index I/O and metadata JSON (parallel to .bin file)

## Key Patterns & Conventions

### Global Model Caching
`embed.py` uses module-level globals (`_model`, `_tokenizer`) with lazy initialization:
```python
def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.embedding_model_name)
    return _model
```
**Why**: Models are heavy; load once per process. Call `get_embedding_model()` wherever embeddings are needed.

### Chunk Metadata Tracking
Each chunk dict maintains: `{"text": "...", "source": "filename.pdf", "page": 3}`. This metadata flows through the entire pipeline and appears in the final answer's source attribution.

### Configuration-Driven Parameters
All non-trivial values live in `config.RAGConfig`: chunk sizes, model names, OpenAI settings. UI and CLI read from config and allow runtime overrides (e.g., `top_k` in `main.py --top_k 10`).

### Vector Store Persistence
FAISS index (`.bin`) and metadata (`.json`) are saved separately. The metadata is **essential**—it links embedding vectors back to original text and sources.

## Developer Workflows

### Local Development
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Building an Index
Typically done via Streamlit UI (upload PDFs → auto-ingest, chunk, embed, save). Alternatively, write a script using `ingest_pdfs()` → `chunk_pages()` → `embed_texts()` → `vector_store.save()`.

### Running Queries
- **CLI**: `python main.py "What is X?" --top_k 5`
- **Web UI**: `streamlit run ui.py` (localhost:8501)

### Dependencies
Key packages: `torch`, `sentence-transformers` (embeddings), `faiss-cpu` (search), `PyMuPDF` (PDF parsing), `openai` (LLM), `streamlit` (UI). See `requirements.txt`.

## Critical Integration Points

### OpenAI Dependency
- API key must be in `OPENAI_API_KEY` env var; config raises RuntimeError if missing
- Only used in `generate_answer()` to call `gpt-4.1-mini` (configurable in `config.openai_model`)
- Prompt prompt includes system instruction: "Answer ONLY using the provided context"

### Embedding Model Lock
Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dims). Changing this requires rebuilding the FAISS index (dimension mismatch will fail at query time).

### PDF Parsing Assumptions
- Uses PyMuPDF's `get_text("text")` which extracts per-page; assumes UTF-8 compatibility
- No layout/table handling—plain text extraction only
- Page numbering is 1-indexed for user display

## When Modifying This Codebase
- **Chunking changes**: Regenerate index (old chunks invalid with new tokenizer)
- **Adding features**: Extend config.py first, then wire through components
- **Changing vector dims**: Rebuild entire index; metadata alone won't transfer
