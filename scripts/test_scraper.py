import os
import sys
import logging
from datetime import datetime

# Set PYTHONPATH
os.environ["PYTHONPATH"] = os.getcwd()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

try:
    from app.integrations.rbi.scraper import RBIScraper
    
    with RBIScraper() as scraper:
        # Test categories
        categories = ["circular", "notification", "master_direction", "master_circular"]
        
        for cat in categories:
            print(f"\n--- Testing Category: {cat} ---")
            if cat in ["circular", "notification"]:
                links = scraper.get_links(cat, year=2024, month=1)
            else:
                links = scraper.get_links(cat)
                
            print(f"Found {len(links)} links.")
            if links:
                print(f"First link: {links[0]['name']} -> {links[0]['url']}")
                
                # Test download of first link
                print(f"Testing download of first link...")
                content = scraper.download_pdf(links[0]['url'])
                if content:
                    print(f"Download SUCCESSFUL ({len(content)} bytes)")
                else:
                    print("Download FAILED")
            else:
                print(f"No links found for {cat}")

    print("\nVerification script COMPLETED!")
except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
