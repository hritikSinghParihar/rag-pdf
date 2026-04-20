import httpx
from typing import List, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RBIClient:
    def __init__(self):
        self.base_url = settings.RBI_SCRAPPER_BASE_URL.rstrip("/")
        self.api_key = settings.RBI_SCRAPPER_API_KEY
        self.headers = {"X-API-Key": self.api_key} if self.api_key else {}

    def list_files(self) -> List[str]:
        """Fetch list of available PDF files from RBI Scrapper."""
        url = f"{self.base_url}/api/v1/downloads/list"
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error listing files from RBI Scrapper: {e}")
            return []

    def download_file(self, file_path: str) -> Optional[bytes]:
        """Download a specific file content from RBI Scrapper."""
        # Note: file_path might contain slashes if it's nested
        url = f"{self.base_url}/api/v1/downloads/file/{file_path}"
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error(f"Error downloading file {file_path} from RBI Scrapper: {e}")
            return None

rbi_client = RBIClient()
