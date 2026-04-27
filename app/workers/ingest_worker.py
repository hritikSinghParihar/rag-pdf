from app.workers.celery_app import celery_app
from app.models import SessionLocal
from app.pipeline.orchestrator import process_document_pipeline
from app.pipeline.embedder import embedder

# Pre-load model in worker process
embedder.initialize()

@celery_app.task
def process_document_task(doc_id: int, file_path: str):
    db = SessionLocal()
    try:
        process_document_pipeline(db, doc_id, file_path)
    finally:
        db.close()
        # Cleanup temporary file
        import os
        if os.path.exists(file_path):
            os.remove(file_path)
