from qdrant_client import QdrantClient as QClient
from qdrant_client.http import models
from app.core.config import settings
from app.core.logging import logger

class QdrantVectorClient:
    def __init__(self):
        self.client = QClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60,  # Increase timeout for slow networks
        )
        self.collection = settings.QDRANT_COLLECTION

    def ensure_collection(self, vector_size: int):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            logger.info(f"Creating collection {self.collection}")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def upsert_vectors(self, vectors: list, payloads: list, ids: list):
        self.client.upsert(
            collection_name=self.collection,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )
        )

    def search(self, query_vector: list, limit: int = 10):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit
        )
        return response.points

vector_client = QdrantVectorClient()
