import logging
from typing import List, Dict, Any

from config import config

# try to import both clients; we'll use whichever is selected at runtime.
try:
    import google.generativeai as genai
except ImportError as exc:
    genai = None  # type: ignore
    _genai_import_error = exc

try:
    from openai import OpenAI
except ImportError as exc:
    OpenAI = None  # type: ignore
    _openai_import_error = exc

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
    
    # Analysis-friendly instruction: maintains strict context but allows logical synthesis
    instruction = (
        "You are a helpful and analytical assistant. Your task is to answer the user's question "
        "using ONLY the provided context.\n"
        "1. If the information is spread across multiple chunks, synthesize it to form a complete answer.\n"
        "2. If the question requires a comparison or logical deduction (e.g., 'Is A more than B?'), "
        "perform that analysis explicitly using the facts in the context.\n"
        "3. cite the specific source and page for each fact you use.\n"
        "4. If and only if the context completely lacks the required information, say you don't know."
    )
    
    prompt = f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnalysis and Answer:"
    return prompt

def generate_answer(chunks: List[Dict[str, Any]], question: str, provider: str | None = None) -> str:
    # use the provided provider or fall back to config default
    if provider is None:
        provider = config.provider

    prompt = build_prompt(chunks, question)

    if provider == "gemini":
        if not config.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        if genai is None:
            raise RuntimeError(
                "google-generativeai package is missing; please add it to requirements."
            )
        logger.info("Calling Gemini with retrieved chunks only")
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.gemini_model)
        resp = model.generate_content(prompt)
        answer = resp.text
    elif provider == "openai":
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        if OpenAI is None:
            raise RuntimeError(
                "openai package is missing; please add it to requirements."
            )
        logger.info("Calling OpenAI with retrieved chunks only")
        client = OpenAI(api_key=config.openai_api_key)
        resp = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers ONLY from the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'.")

    return answer
