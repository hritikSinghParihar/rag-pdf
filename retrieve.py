import logging
from typing import List, Dict, Any

import numpy as np

from config import config
from embed import embed_texts, get_embedding_model
from vector_store import get_vector_store

logger = logging.getLogger(__name__)

def retrieve_relevant_chunks(
    query: str,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = config.top_k

    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()

    store = get_vector_store(dim)

    query_emb = embed_texts([query])
    query_emb = np.array(query_emb).reshape(1, -1)

    results = store.search(query_emb, top_k=top_k)
    return results
