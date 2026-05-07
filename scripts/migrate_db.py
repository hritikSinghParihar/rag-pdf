from sqlalchemy import text
import os
import sys

# Set PYTHONPATH
os.environ["PYTHONPATH"] = os.getcwd()

try:
    from app.models import engine
    
    with engine.connect() as conn:
        print("Checking if source_url column exists...")
        # Check if column exists
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='documents' AND column_name='source_url';"))
        exists = result.fetchone()
        
        if not exists:
            print("Adding source_url column to documents table...")
            # Use raw SQL for migration
            conn.execute(text("ALTER TABLE documents ADD COLUMN source_url VARCHAR;"))
            conn.execute(text("CREATE INDEX ix_documents_source_url ON documents (source_url);"))
            conn.commit()
            print("Column added successfully!")
        else:
            print("Column already exists.")
            
except Exception as e:
    print(f"Error during migration: {e}")
    sys.exit(1)
print("Migration script finished.")
