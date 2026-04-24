from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router
from app.schemas.response import ErrorResponse
from app.models import Base, engine
from app.models.user import User  # Import models to ensure they are registered with Base
from app.models.document import Document, Chunk

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    import logging
    logger = logging.getLogger("rag_app")
    logger.info("Starting database table creation...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database table creation completed.")
    except Exception as e:
        logger.error(f"Error during database table creation: {e}")
        raise e
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Centralized Error Handling Middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="An unexpected error occurred",
            details=str(exc)
        ).model_dump()
    )

@app.get("/")
async def root_redirect():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "ok"}

app.include_router(api_router, prefix=settings.API_V1_STR)
