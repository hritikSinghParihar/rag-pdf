import sys
import os
from unittest.mock import MagicMock, patch

# Add app to path
sys.path.append(os.path.abspath("."))

def test_rbi_sync():
    # Mock config to avoid env errors
    with patch("app.core.config.settings.RBI_SCRAPPER_BASE_URL", "http://mock"), \
         patch("app.core.config.settings.RBI_SCRAPPER_API_KEY", "mock-key"):
        
        # Import after patches if possible, but here we can just mock the dependencies
        from app.services.rbi_service import rbi_service
        from sqlalchemy.orm import Session
        
        db = MagicMock(spec=Session)
        user_id = 1
        
        # Mock Document query
        mock_query = db.query.return_value
        mock_query.all.return_value = []
        
        # Mock rbi_client
        with patch("app.services.rbi_service.rbi_client") as mock_client:
            mock_client.list_files.return_value = ["2025/test.pdf"]
            mock_client.download_file.return_value = b"Fake PDF Content"
            
            # Mock pipeline and ingestion service
            with patch("app.services.rbi_service.ingestion_service") as mock_ingest, \
                 patch("app.services.rbi_service.process_document_pipeline") as mock_pipeline:
                
                mock_doc = MagicMock()
                mock_doc.id = "doc-uuid"
                mock_doc.file_name = "test.pdf"
                mock_ingest.process_upload.return_value = mock_doc
                
                result = rbi_service.sync_rbi_documents(db, user_id)
                
                print(f"Sync Results: {result}")
                if result["synced"] == 1:
                    print("✓ Correct synced count")
                else:
                    print(f"✗ Unexpected synced count: {result['synced']}")
                    sys.exit(1)
                    
                if mock_client.list_files.called:
                    print("✓ list_files called")
                if mock_client.download_file.called:
                    print("✓ download_file called")
                if mock_ingest.process_upload.called:
                    print("✓ process_upload called")
                if mock_pipeline.called:
                    print("✓ process_document_pipeline called")

if __name__ == "__main__":
    print("Running RBI Sync Integration Test...")
    try:
        test_rbi_sync()
        print("\nIntegration test PASSED")
    except Exception as e:
        print(f"\nIntegration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
