from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Project Info
    PROJECT_NAME: str = "Local RAG PDF"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "development_secret_key_change_me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days
    ALGORITHM: str = "HS256"

    # Admin Info
    FIRST_SUPERUSER: str = "admin@example.com"
    FIRST_SUPERUSER_PASSWORD: str = "adminpassword"

    # PostgreSQL
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "rag_db"
    POSTGRES_PORT: int = 5432
    POSTGRES_SSLMODE: str = "disable"
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @property
    def database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        url = f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        if self.POSTGRES_SSLMODE:
            url += f"?sslmode={self.POSTGRES_SSLMODE}"
        return url

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "documents"

    # Cloudflare R2 (S3)
    R2_BUCKET: str = "rag-docs"
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_ENDPOINT_URL: str = ""  # e.g., https://<id>.r2.cloudflarestorage.com

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    # LLM Providers
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    LLM_PROVIDER: str = "openai"  # "openai" or "gemini"

    # RAG Config
    CHUNK_SIZE_TOKENS: int = 600
    CHUNK_OVERLAP_TOKENS: int = 100
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K: int = 10

    # RBI Scrapper
    RBI_SCRAPPER_BASE_URL: str = "http://localhost:8001"
    RBI_SCRAPPER_API_KEY: Optional[str] = None

settings = Settings()
