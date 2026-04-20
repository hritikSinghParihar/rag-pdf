import os
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.document import Document
from app.integrations.storage.r2_client import storage_client
# from app.pipeline.orchestrator import process_document_pipeline  # Optional

logger = logging.getLogger(__name__)

class IngestionService:
    @staticmethod
    def process_upload(db: Session, file_path: str, user_id: int):
        filename = os.path.basename(file_path)
        
        # 1. Create DB record
        doc = Document(
            file_name=filename,
            status="processing"
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        # 2. Upload to R2 (R2 path not stored in DB - handled externally if needed)
        r2_key = f"docs/{user_id}/{doc.id}/{filename}"
        storage_client.upload_file(file_path, r2_key)
        
        # 3. Trigger processing (should be Celery task in future)
        # For now, synchronous or direct call
        return doc

ingestion_service = IngestionService()
