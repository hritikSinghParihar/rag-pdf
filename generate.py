import logging
from typing import List, Dict, Any

from config import config
import google.generativeai as genai

logger = logging.getLogger(__name__)

def build_prompt(chunks: List[Dict[str, Any]], question: str) -> str:
    context_blocks = []
    for c in chunks:
        source = c.get("source")
        page = c.get("page")
        text = c.get("text")
        meta_str = f"[Source: {source}, Page: {page}]"
        context_blocks.append(f"{meta_str}\n{text}")
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context."
    # strict instruction to confine answers to the provided context.  the model
    # will respond "I don't know" if the information isn't present verbatim.
    instruction = (
        "You are a helpful assistant. Answer ONLY using the provided context.\n"
        "If the answer is not in the context, say you don't know."
    )
    prompt = f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    return prompt

def generate_answer(chunks: List[Dict[str, Any]], question: str) -> str:
    if not config.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel(config.gemini_model)

    prompt = build_prompt(chunks, question)

    logger.info("Calling Gemini with retrieved chunks only")

    resp = model.generate_content(prompt)

    answer = resp.text
    return answer
