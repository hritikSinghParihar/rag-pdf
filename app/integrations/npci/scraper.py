import time
import random
import logging
from typing import Optional, Dict, List
from curl_cffi import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings

logger = logging.getLogger("rag_app")

class NPCIScraper:
    def __init__(self):
        self.session = self._setup_session()
        self.base_url = "https://www.npci.org.in"

    def _setup_session(self) -> requests.Session:
        return requests.Session(
            impersonate=settings.NPCI_CHROME_IMPERSONATION,
            timeout=30,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_press_release_details(self, year: int, page: int = 1) -> Dict:
        """Fetches press releases for a specific year from NPCI API."""
        url = f"{settings.NPCI_PRESS_RELEASE_DETAILS_API}"
        params = {
            "tabSlug": "press-releases",
            "year": year,
            "page": page
        }
        logger.info(f"Fetching NPCI press releases for year {year} (Page {page})...")
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_media_coverages(self, page: int = 1, page_size: int = 25) -> Dict:
        """Fetches media coverages from NPCI API."""
        url = f"{settings.NPCI_MEDIA_COVERAGE_API}"
        params = {
            "populate[details][populate]": "*",
            "pagination[page]": page,
            "pagination[pageSize]": page_size,
            "sort": "createdAt:desc"
        }
        logger.info(f"Fetching NPCI media coverages (Page {page})...")
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """Downloads a PDF from NPCI."""
        if not pdf_url.startswith("http"):
            pdf_url = f"{self.base_url}{pdf_url}"
            
        logger.info(f"Downloading NPCI PDF: {pdf_url}")
        time.sleep(random.uniform(0.5, 1.5))
        resp = self.session.get(pdf_url)
        if resp.status_code == 200 and resp.content:
            return resp.content
        return None

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
