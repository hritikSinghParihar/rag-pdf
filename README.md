# Local-First RAG over PDFs (Python + Streamlit)

## Features
- Local PDF ingestion (PyMuPDF)
- Local chunking + embeddings (sentence-transformers)
- Local FAISS vector store
- Only top-k chunks + question go to OpenAI
- Streamlit UI + CLI

## Setup

```bash
git clone <your-repo> rag_system
cd rag_system
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


## Configuration

1. Put your LLM API key in `keys.txt` or set the `GEMINI_API_KEY` (or
	 `OPENAI_API_KEY` if you revert to OpenAI) environment variable.
2. The `config.py` loader will pick up the key from the env var first,
	 then fall back to `keys.txt`.

## GitHub preparation

- A `.gitignore` file is included to exclude sensitive keys, virtual
	environments, and editor settings (see below).
- Before pushing to a public repository, ensure `keys.txt` does **not**
	contain a real API key or remove it entirely.
