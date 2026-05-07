import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.document import Document, SyncJob
from app.services.ingestion_service import ingestion_service
from app.pipeline.orchestrator import process_document_pipeline
from app.integrations.rbi.scraper import RBIScraper

logger = logging.getLogger("rag_app")

class RBIService:
    def sync_rbi_documents(self, db: Session, user_id: int, job_id: str = None):
        """Sync documents from RBI website to RAG system."""
        logger.info(f"Starting internal RBI document sync (Job: {job_id})...")
        
        synced_total = 0
        skipped_total = 0
        error_total = 0
        
        doc_categories = ["circular", "notification", "master_direction", "master_circular"]
        
        try:
            with RBIScraper() as scraper:
                for category in doc_categories:
                    try:
                        logger.info(f"Syncing category: {category}")
                        
                        # 1. Fetch links based on category
                        if category in ["circular", "notification"]:
                            # Periodic documents: loop through years/months
                            start_year = settings.RBI_SYNC_START_YEAR
                            current_year = datetime.now().year
                            
                            for year in range(start_year, current_year + 1):
                                for month in range(1, 13):
                                    if year == current_year and month > datetime.now().month:
                                        break
                                        
                                    try:
                                        links = scraper.get_links(category, year=year, month=month)
                                        self._process_links(db, scraper, links, user_id, job_id)
                                    except Exception as e:
                                        logger.error(f"Error syncing {category} for {year}-{month}: {e}")
                                        continue
                        else:
                            # Consolidated documents: fetch all at once
                            try:
                                links = scraper.get_links(category)
                                self._process_links(db, scraper, links, user_id, job_id)
                            except Exception as e:
                                logger.error(f"Error syncing {category}: {e}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"Unexpected error in category {category}: {e}")
                        continue
                        
            # Update job status at the end
            if job_id:
                job = db.query(SyncJob).get(job_id)
                if job:
                    job.status = "completed"
                    job.message = f"Sync completed across all categories."
                    db.commit()

        except Exception as e:
            logger.error(f"Critical error during RBI sync: {e}")
            if job_id:
                job = db.query(SyncJob).get(job_id)
                if job:
                    job.status = "failed"
                    job.message = f"Error: {str(e)}"
                    db.commit()

    def _process_links(self, db: Session, scraper: RBIScraper, links: list, user_id: int, job_id: str):
        """Processes a list of links: download, save, index."""
        if not links:
            return

        # Get existing document URLs to avoid duplicates
        existing_urls = {doc.source_url for doc in db.query(Document.source_url).all() if doc.source_url}
        
        for link_info in links:
            url = link_info["url"]
            name = link_info["name"]
            
            if url in existing_urls:
                logger.debug(f"Skipping (already exists): {name}")
                continue
                
            # Download
            logger.info(f"Downloading: {name}")
            content = scraper.download_pdf(url)
            if not content:
                continue
                
            # Save to temporary file
            safe_name = self._sanitize_filename(name)
            filename = f"rbi_{int(time.time())}_{safe_name}.pdf"
            tmp_path = os.path.join("uploads", filename)
            os.makedirs("uploads", exist_ok=True)
            
            with open(tmp_path, "wb") as f:
                f.write(content)
                
            try:
                # Ingest and Process
                doc = ingestion_service.process_upload(db, tmp_path, user_id)
                # Store the source URL for de-duplication
                doc.source_url = url
                db.commit()
                
                process_document_pipeline(db, doc.id, tmp_path)
                logger.info(f"Successfully processed: {name}")
            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
            # Update job progress
            if job_id:
                job = db.query(SyncJob).get(job_id)
                if job:
                    job.synced_files = str(int(job.synced_files or 0) + 1)
                    db.commit()

    def _sanitize_filename(self, name: str) -> str:
        """Removes illegal characters from filename."""
        import re
        clean_name = re.sub(r'[\\/*?:"<>|]', '_', name)
        return clean_name.replace(" ", "_").strip()

rbi_service = RBIService()
import time 
