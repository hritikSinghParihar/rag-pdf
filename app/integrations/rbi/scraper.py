import time
import random
import logging
from typing import Optional, Dict, List
from curl_cffi import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings
from app.integrations.rbi.parser import parse_hidden_form_data, parse_document_links, parse_pdf_link

logger = logging.getLogger("rag_app")

class RBIScraper:
    def __init__(self):
        self.session = self._setup_session()

    def _setup_session(self) -> requests.Session:
        return requests.Session(
            impersonate=settings.RBI_CHROME_IMPERSONATION,
            timeout=30, 
        )

    def ensure_session(self):
        """Hits homepage to establish cookies if not present."""
        if not self.session.cookies:
            try:
                logger.info("Establishing RBI session cookies...")
                resp = self.session.get(settings.RBI_HOME_URL)
                logger.info(f"RBI Homepage status: {resp.status_code}")
                time.sleep(random.uniform(1.0, 2.0))
            except Exception as e:
                logger.warning(f"Failed to load RBI homepage: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_links(self, doc_type: str, year: Optional[int] = None, month: Optional[int] = None) -> List[Dict[str, str]]:
        """Fetches document links for a specific type and period."""
        self.ensure_session()
        
        urls = {
            "circular": settings.RBI_INDEX_URL,
            "notification": settings.RBI_NOTIFICATIONS_URL,
            "master_direction": settings.RBI_MASTER_DIRECTIONS_URL,
            "master_circular": settings.RBI_MASTER_CIRCULARS_URL
        }
        
        url = urls.get(doc_type)
        if not url:
            logger.error(f"Unknown document type: {doc_type}")
            return []
            
        logger.info(f"Fetching {doc_type} index...")
        
        try:
            r_get = self.session.get(url)
            if r_get.status_code != 200:
                logger.error(f"Failed to load {doc_type} index. Status: {r_get.status_code}")
                return []
                
            # For Circulars and Notifications, we need to handle the ASP.NET form post for year/month
            if doc_type in ["circular", "notification"] and year and month:
                form_data = parse_hidden_form_data(r_get.text)
                form_data["hdnYear"] = str(year)
                form_data["hdnMonth"] = str(month)
                
                # Small delay before post
                time.sleep(random.uniform(0.5, 1.5))
                
                r_post = self.session.post(url, data=form_data)
                if r_post.status_code != 200:
                    logger.error(f"Failed to post form for {doc_type} {year}-{month}. Status: {r_post.status_code}")
                    return []
                return parse_document_links(r_post.text, url, doc_type=doc_type)
            
            # For Master Directions/Circulars, we just parse the GET response
            return parse_document_links(r_get.text, url, doc_type=doc_type)
            
        except Exception as e:
            logger.error(f"Error fetching links for {doc_type}: {e}")
            raise e # Raise to trigger retry

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_pdf(self, url: str) -> Optional[bytes]:
        """Downloads a PDF from a document detail page."""
        try:
            # First hit the detail page to find the PDF link
            page = self.session.get(url)
            if page.status_code != 200:
                return None
                
            pdf_url = parse_pdf_link(page.text, url)
            if not pdf_url:
                logger.warning(f"No PDF link found on page: {url}")
                return None

            # Small delay before downloading PDF
            time.sleep(random.uniform(0.5, 1.0))
            
            resp = self.session.get(pdf_url)
            if resp.status_code == 200 and resp.content:
                return resp.content
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            raise e # Raise to trigger retry

    def close(self):
        self.session.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
