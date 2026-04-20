import os
import tempfile
import logging
from sqlalchemy.orm import Session
from app.integrations.rbi_client import rbi_client
from app.models.document import Document, SyncJob
from app.services.ingestion_service import ingestion_service
from app.pipeline.orchestrator import process_document_pipeline

logger = logging.getLogger(__name__)

class RBIService:
    def sync_rbi_documents(self, db: Session, user_id: int, job_id: str = None):
        """Sync documents from RBI Scrapper to RAG system."""
        logger.info(f"Starting RBI document sync (Job: {job_id})...")
        
        # 1. Fetch list of files
        files = rbi_client.list_files()
        if not files:
            logger.info("No files found or error fetching list.")
            if job_id:
                job = db.query(SyncJob).filter(SyncJob.id == job_id).first()
                if job:
                    job.status = "completed"
                    job.message = "No files found or error fetching list."
                    db.commit()
            return {"synced": 0, "skipped": 0, "errors": 0}
            
        total_files = len(files)
        if job_id:
            job = db.query(SyncJob).filter(SyncJob.id == job_id).first()
            if job:
                job.total_files = str(total_files)
                db.commit()

        # 2. Get existing document names
        existing_docs = db.query(Document.file_name).all()
        existing_names = {doc.file_name for doc in existing_docs}
        
        synced_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, file_path in enumerate(files):
            filename = os.path.basename(file_path)
            
            # Skip if already exists
            if filename in existing_names:
                skipped_count += 1
                continue
                
            # 3. Download file
            file_content = rbi_client.download_file(file_path)
            if not file_content:
                error_count += 1
                continue
                
            # 4. Save to temp file and process
            tmp_path = f"/tmp/rbi_sync_{filename}"
            with open(tmp_path, "wb") as f:
                f.write(file_content)
                
            try:
                # Process document
                doc = ingestion_service.process_upload(db, tmp_path, user_id)
                process_document_pipeline(db, doc.id, tmp_path)
                synced_count += 1
                logger.info(f"Successfully synced: {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                error_count += 1
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                
                # Update progress periodically or every file
                if job_id and (i % 5 == 0 or i == total_files - 1):
                    job = db.query(SyncJob).filter(SyncJob.id == job_id).first()
                    if job:
                        job.synced_files = str(synced_count)
                        job.error_files = str(error_count)
                        db.commit()
                    
        if job_id:
            job = db.query(SyncJob).filter(SyncJob.id == job_id).first()
            if job:
                job.status = "completed"
                job.synced_files = str(synced_count)
                job.error_files = str(error_count)
                job.message = f"Sync completed. {synced_count} synced, {skipped_count} skipped, {error_count} errors."
                db.commit()

        return {
            "synced": synced_count,
            "skipped": skipped_count,
            "errors": error_count
        }

rbi_service = RBIService()
