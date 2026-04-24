from typing import List
from app.core.config import settings

class Embedder:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

embedder = Embedder()
