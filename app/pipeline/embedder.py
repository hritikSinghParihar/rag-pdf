import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger("rag_app")

class Embedder:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self.initialize()
        return self._model

    def initialize(self):
        """Pre-load the model to avoid lazy-loading delays."""
        if self._model is not None:
            return
            
        logger.info(f"Loading SentenceTransformer model: {settings.EMBEDDING_MODEL_NAME}")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

embedder = Embedder()
