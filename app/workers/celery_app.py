from celery import Celery
from app.core.config import settings

broker_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
if settings.REDIS_PASSWORD:
    broker_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"

celery_app = Celery(
    "rag_workers",
    broker=broker_url,
    backend=broker_url,
)

# celery_app.conf.task_routes = {
#     "app.workers.ingest_worker.process_document_task": "main-queue",
# }

# Explicitly import tasks to ensure they are registered
import app.workers.ingest_worker

celery_app.autodiscover_tasks(["app.workers"])
