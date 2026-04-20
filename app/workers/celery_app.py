from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "rag_workers",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
)

celery_app.conf.task_routes = {
    "app.workers.ingest_worker.process_document_task": "main-queue",
}

celery_app.autodiscover_tasks(["app.workers"])
