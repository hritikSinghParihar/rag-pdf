# Local-First RAG over PDFs (Python + Streamlit)

## Features
- Local PDF ingestion (PyMuPDF)
- Local chunking + embeddings (sentence-transformers)
- Local FAISS vector store
- Only top-k chunks + question go to OpenAI
- Streamlit UI + CLI

## Overview

This repository is a *local-first* retrieval-augmented generation (RAG)
prototype specifically tailored for PDF documents. The idea is to keep the
heavy lifting (parsing, indexing, embedding) on the user’s machine, only
sending a small, relevant slice of text to an external language model for
answering questions. The typical workflow is:

1. **Upload/ingest** one or more PDF files via the Streamlit web UI.
2. **Chunk** each page into token-bounded pieces and compute embeddings using
	a SentenceTransformer model (`all-MiniLM-L6-v2` by default).
3. **Persist** the vectors in a FAISS index along with simple metadata
	(source filename, page number, chunk id, and the chunk text itself). The
	index lives under `index/` and can be reloaded between sessions.
4. **Query**: when you type a question, the system embeds the question and
	performs an L2 search in the FAISS index, returning the top‑k closest
	chunks.
5. **Generate**: the retrieved chunks are concatenated and prefixed with a
	strict system instruction; the resulting prompt is sent to Gemini (or
	OpenAI) to produce the final answer. Only the prompt and reply leave the
	local machine.

Because all data never leaves the user’s environment except for the snippets
used in the prompt, this pattern provides better privacy and bandwidth
efficiency compared to uploading entire documents to the cloud.

## Setup

```bash
git clone <your-repo> rag_system
cd rag_system
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


## Configuration
Additional config options are available in `config.RAGConfig`:

```python
# chunking
chunk_size_tokens: int = 600
chunk_overlap_tokens: int = 100

# retrieval
top_k: int = 5

# embedding
embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
```

1. Put your LLM API key in `keys.txt` or set the `GEMINI_API_KEY` (or
	 `OPENAI_API_KEY` if you revert to OpenAI) environment variable.
2. The `config.py` loader will pick up the key from the env var first,
	 then fall back to `keys.txt`.

## GitHub preparation

- A `.gitignore` file is included to exclude sensitive keys, virtual
	environments, and editor settings (see below).
- Before pushing to a public repository, ensure `keys.txt` does **not**
	contain a real API key or remove it entirely.

### Running on a public host

1. Push the repository to GitHub as described earlier.
2. Use Streamlit Community Cloud or any container/VM host (Heroku, Docker,
	 etc.) to run `streamlit run ui.py` remotely. See the previous section for
	 more details.

### Packaging for local execution

If you want to distribute a standalone executable, you can containerize the
application or use PyInstaller. Both approaches simply bundle Python and the
code; the app still launches a local web server and opens the browser.

## Troubleshooting

- **Empty search results**: make sure you have indexed documents. Use the
	“Reload index” button or re-upload PDFs.
- **Model responds "I don’t know"**: the strict prompt requires the answer to
	appear *verbatim* in the retrieved chunks; adjust your question or provide
	more documents.
- **`FutureWarning` about `google.generativeai`**: the library is deprecated
	(see the warning). You can switch to `google.genai` by updating
	`generate.py` accordingly.

## Contributing

This is a small demo project. If you’d like to add features (e.g. alternate
embedding models, PDF preprocessing, more sophisticated retrieval), feel free
to fork and submit pull requests.

## License

Specify an appropriate open-source license (MIT, Apache 2.0, etc.) before
pushing to GitHub.
