import os
import logging
from typing import List, Dict, Any

# guard against import errors so the app shows a message instead of a white
# page. Streamlit Cloud hides import failures in its build logs, so we want to
# catch them early and display them in the UI.
import_streamlit_error = None
try:
    import numpy as np
    import streamlit as st

    from config import config
    from ingest import ingest_pdfs
    from embed import chunk_pages, embed_texts, get_embedding_model
    from vector_store import get_vector_store
    from retrieve import retrieve_relevant_chunks
    from generate import generate_answer
except Exception as e:  # pylint: disable=broad-except
    import_streamlit_error = e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Local RAG PDF QA", layout="wide")

# if imports failed, surface the exception and abort
if import_streamlit_error is not None:
    try:
        st.error("Failed to start app due to import error:")
        st.exception(import_streamlit_error)
    except Exception:
        print("Import error while starting app:", import_streamlit_error)
    raise SystemExit(import_streamlit_error)

st.title("📚 Local-First RAG over PDFs")

with st.sidebar:
    st.header("Settings")

    # provider selection
    st.subheader("AI Model Provider")
    provider = st.radio(
        "Select LLM provider:",
        options=["gemini", "openai"],
        format_func=lambda x: "Google Gemini" if x == "gemini" else "OpenAI",
        horizontal=True,
    )
    config.provider = provider

    # show API key status for both providers
    st.markdown("---")
    st.subheader("API Keys Status")
    
    gemini_status = "✅ Set" if config.gemini_api_key else "❌ Not set"
    openai_status = "✅ Set" if config.openai_api_key else "❌ Not set"
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Gemini**: {gemini_status}")
    with col2:
        st.write(f"**OpenAI**: {openai_status}")

    st.markdown("---")
    st.subheader("Chunking Settings")
    
    chunk_size = st.number_input(
        "Chunk size (tokens)", min_value=200, max_value=1200,
        value=config.chunk_size_tokens, step=50,
    )
    chunk_overlap = st.number_input(
        "Chunk overlap (tokens)", min_value=0, max_value=400,
        value=config.chunk_overlap_tokens, step=25,
    )
    top_k = st.number_input(
        "Top-k retrieval", min_value=1, max_value=20,
        value=config.top_k, step=1,
    )

    st.markdown("---")
    st.markdown("**Index Directory:**")
    st.code(os.path.abspath(config.index_dir))

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
if "num_docs" not in st.session_state:
    st.session_state["num_docs"] = 0
if "num_chunks" not in st.session_state:
    st.session_state["num_chunks"] = 0

col1, col2, col3 = st.columns(3)
with col1:
    index_button = st.button("📥 Index Documents")
with col2:
    reload_button = st.button("🔁 Reload Existing Index")
with col3:
    clear_button = st.button("🧹 Clear Index (local files)")

status_placeholder = st.empty()

if index_button:
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Indexing documents... This may take a while."):
            os.makedirs(config.data_dir, exist_ok=True)
            saved_paths = []
            for uf in uploaded_files:
                save_path = os.path.join(config.data_dir, uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.read())
                saved_paths.append(save_path)

            pages = ingest_pdfs(saved_paths)
            chunks = chunk_pages(
                pages,
                chunk_size_tokens=int(chunk_size),
                chunk_overlap_tokens=int(chunk_overlap),
            )

            texts = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            
            # Add text back to metadata for later retrieval
            for meta, text in zip(metadatas, texts):
                meta["text"] = text

            model = get_embedding_model()
            dim = model.get_sentence_embedding_dimension()
            store = get_vector_store(dim)
            store.create_new()

            embeddings = embed_texts(texts)
            embeddings = np.array(embeddings)
            store.add(embeddings, metadatas)
            store.save()

            st.session_state["indexed"] = True
            st.session_state["num_docs"] = len(saved_paths)
            st.session_state["num_chunks"] = len(chunks)

            status_placeholder.success(
                f"Indexed {len(saved_paths)} documents into {len(chunks)} chunks."
            )

if reload_button:
    with st.spinner("Reloading index..."):
        model = get_embedding_model()
        dim = model.get_sentence_embedding_dimension()
        store = get_vector_store(dim)
        st.session_state["indexed"] = store.index is not None and store.index.ntotal > 0
        st.session_state["num_docs"] = len({m.get("source") for m in store.metadata}) if store.metadata else 0
        st.session_state["num_chunks"] = len(store.metadata)
        status_placeholder.info(
            f"Loaded index with {st.session_state['num_docs']} docs and {st.session_state['num_chunks']} chunks."
        )

if clear_button:
    if os.path.exists(config.index_dir):
        for f in os.listdir(config.index_dir):
            os.remove(os.path.join(config.index_dir, f))
        status_placeholder.warning("Cleared local index files.")
        st.session_state["indexed"] = False
        st.session_state["num_docs"] = 0
        st.session_state["num_chunks"] = 0
    else:
        st.info("No index directory found.")

st.markdown("---")

st.subheader("Ask a question")

question = st.text_input("Your question about the indexed PDFs")
ask_button = st.button("❓ Ask")

if st.session_state.get("indexed"):
    st.caption(
        f"Indexed documents: {st.session_state['num_docs']} | "
        f"Total chunks: {st.session_state['num_chunks']}"
    )
else:
    st.caption("No index loaded yet.")

if ask_button:
    if not question.strip():
        st.error("Please type a question.")
    elif not st.session_state.get("indexed"):
        st.error("Please index documents or reload an index first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            chunks = retrieve_relevant_chunks(question, top_k=int(top_k))
            # build prompt separately so we can display it in the UI
            from generate import build_prompt
            prompt = build_prompt(chunks, question)
            try:
                answer = generate_answer(chunks, question, provider=provider)
            except RuntimeError as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

            st.markdown("### Answer")
            st.write(answer)

            # show debug info: the prompt that was sent to the LLM
            st.markdown("### Prompt sent to model")
            st.code(prompt, language="text")

            if chunks:
                st.markdown("### Retrieved Context (Preview)")
                st.caption(f"Top-{len(chunks)} chunks used:")
                for i, ch in enumerate(chunks, start=1):
                    with st.expander(
                        f"Chunk {i} — {ch.get('source')} (page {ch.get('page')})"
                    ):
                        st.write(ch.get("text", "")[:2000])

                st.markdown("### Sources")
                for ch in chunks:
                    src = ch.get("source")
                    page = ch.get("page")
                    st.write(f"- {src}, page {page}")
