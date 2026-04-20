import sys
import os

# Add the project root to sys.path
sys.path.append("/home/carl/Desktop/wisipay/rag-pdf")

from app.models import SessionLocal
from app.services.rbi_service import rbi_service
from app.repositories.user_repo import user_repo

def run_sync():
    db = SessionLocal()
    try:
        # Get the first superuser
        user = user_repo.get_by_email(db, email="admin@example.com")
        if not user:
            print("Superuser not found. Please run initial_data.py first.")
            return
        
        print(f"Starting sync for user: {user.email} (ID: {user.id})")
        stats = rbi_service.sync_rbi_documents(db, user.id)
        print(f"Sync complete: {stats}")
    finally:
        db.close()

if __name__ == "__main__":
    run_sync()
