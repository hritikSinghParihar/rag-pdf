import os
import json
import logging
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

from config import config

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, dim: int, index_path: str, metadata_path: str):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def create_new(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index from {self.index_path}")
        else:
            logger.warning("Index file not found, creating new index")
            self.create_new()
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata from {self.metadata_path}")
        else:
            logger.warning("Metadata file not found, starting empty metadata")
            self.metadata = []

    def save(self):
        if self.index is not None:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
        if self.metadata is not None:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata to {self.metadata_path}")

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        if self.index is None:
            self.create_new()
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Empty index")
            return []
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["distance"] = float(dist)
            results.append(item)
        return results

def get_vector_store(embedding_dim: int) -> FAISSVectorStore:
    index_path = os.path.join(config.index_dir, config.index_file)
    metadata_path = os.path.join(config.index_dir, config.metadata_file)
    store = FAISSVectorStore(embedding_dim, index_path, metadata_path)
    store.load()
    return store
