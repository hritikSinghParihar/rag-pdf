import os
import time
import logging
import re
from sqlalchemy.orm import Session
from app.models.document import Document, SyncJob
from app.services.ingestion_service import ingestion_service
from app.pipeline.orchestrator import process_document_pipeline
from app.integrations.npci.scraper import NPCIScraper
from app.integrations.npci.parser import parse_npci_press_releases, parse_npci_media_coverages

logger = logging.getLogger("rag_app")

class NPCIService:
    def sync_npci_documents(self, db: Session, user_id: int, job_id: str = None):
        """Sync documents from NPCI website to RAG system."""
        logger.info(f"Starting NPCI document sync (Job: {job_id})...")
        
        try:
            with NPCIScraper() as scraper:
                # 1. Fetch Press Releases by Year
                start_year = 2024  # Or from settings
                current_year = 2026 # As seen in screenshot
                for year in range(start_year, current_year + 1):
                    try:
                        pr_data = scraper.get_press_release_details(year=year)
                        pr_links = parse_npci_press_releases(pr_data)
                        self._process_links(db, scraper, pr_links, user_id, job_id)
                    except Exception as e:
                        logger.error(f"Error syncing NPCI press releases for {year}: {e}")
                
                # 2. Fetch Media Coverages
                mc_data = scraper.get_media_coverages()
                mc_links = parse_npci_media_coverages(mc_data)
                self._process_links(db, scraper, mc_links, user_id, job_id)
                
            # Update job status at the end
            if job_id:
                job = db.query(SyncJob).get(job_id)
                if job:
                    job.status = "completed"
                    job.message = "NPCI Sync completed."
                    db.commit()

        except Exception as e:
            logger.error(f"Critical error during NPCI sync: {e}")
            if job_id:
                job = db.query(SyncJob).get(job_id)
                if job:
                    job.status = "failed"
                    job.message = f"Error: {str(e)}"
                    db.commit()

    def _process_links(self, db: Session, scraper: NPCIScraper, links: list, user_id: int, job_id: str):
        """Processes a list of links: download, save, index."""
        if not links:
            return

        # Get existing document URLs to avoid duplicates
        existing_urls = {doc.source_url for doc in db.query(Document.source_url).all() if doc.source_url}
        
        for link_info in links:
            url = link_info["url"]
            name = link_info["name"]
            doc_type = link_info["type"]
            
            # NPCI URLs are relative in API
            full_url = url if url.startswith("http") else f"https://www.npci.org.in{url}"
            
            if full_url in existing_urls:
                logger.debug(f"Skipping (already exists): {name}")
                continue
                
            # Download
            logger.info(f"Downloading {doc_type}: {name}")
            content = scraper.download_pdf(full_url)
            if not content:
                continue
                
            # Save to temporary file
            safe_name = self._sanitize_filename(name)
            filename = f"npci_{doc_type}_{int(time.time())}_{safe_name}.pdf"
            tmp_path = os.path.join("uploads", filename)
            os.makedirs("uploads", exist_ok=True)
            
            with open(tmp_path, "wb") as f:
                f.write(content)
                
            try:
                # Ingest and Process
                doc = ingestion_service.process_upload(db, tmp_path, user_id)
                doc.source_url = full_url
                db.commit()
                
                process_document_pipeline(db, doc.id, tmp_path)
                logger.info(f"Successfully processed NPCI {doc_type}: {name}")
            except Exception as e:
                logger.error(f"Error processing NPCI {name}: {e}")
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
        clean_name = re.sub(r'[\\/*?:"<>|]', '_', name)
        return clean_name.replace(" ", "_").strip()

npci_service = NPCIService()
