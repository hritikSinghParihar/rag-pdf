from fastapi import APIRouter
from app.api.v1.routes import ingest, query, documents, health, auth

api_router = APIRouter()
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
