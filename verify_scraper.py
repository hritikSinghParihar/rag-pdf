import sys
import os

# Add the project root to sys.path
sys.path.append("/home/carl/Desktop/wisipay/rag-pdf")

from app.integrations.rbi_client import rbi_client
from app.core.config import settings

def test_connection():
    print(f"Testing connection to: {settings.RBI_SCRAPPER_BASE_URL}")
    print(f"Using API Key: {settings.RBI_SCRAPPER_API_KEY[:5]}...{settings.RBI_SCRAPPER_API_KEY[-5:]}")
    
    files = rbi_client.list_files()
    if files:
        print(f"Successfully fetched {len(files)} files.")
        for f in files[:5]:
            print(f" - {f}")
    else:
        print("Failed to fetch files or no files found.")

if __name__ == "__main__":
    test_connection()
