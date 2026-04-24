from qdrant_client import QdrantClient
from app.core.config import settings

def test_qdrant():
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        collections = client.get_collections()
        print(f"Successfully connected to Qdrant. Collections: {collections}")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")

if __name__ == "__main__":
    test_qdrant()
