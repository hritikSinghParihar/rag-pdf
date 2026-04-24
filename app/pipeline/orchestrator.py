import os
from sqlalchemy.orm import Session
from app.models.document import Document, Chunk
from app.pipeline.chunker import chunker
from app.pipeline.embedder import embedder
from app.integrations.vector_db.qdrant_client import vector_client
import uuid

def process_document_pipeline(db: Session, doc_id: int, file_path: str):
    # This is a simplified version of the full ingestion pipeline
    # In a real app, this would be a Celery task.
    
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return
    
    try:
        # 1. Extraction using the new unified FileParser
        from app.pipeline.parser import file_parser
        parsed_pages = file_parser.parse(file_path)
        
        all_chunks_data = []
        for i, page_data in enumerate(parsed_pages):
            page_text = page_data["text"]
            metadata = {"source_id": doc.id, "page": page_data["page"]}
            chunks = chunker.split_page(page_text, metadata)
            all_chunks_data.extend(chunks)
            
        # 2. Embedding & Vector Search Store
        texts = [c["text"] for c in all_chunks_data]
        embeddings = embedder.embed_texts(texts)
        
        vector_ids = [str(uuid.uuid4()) for _ in texts]
        payloads = [c["metadata"] for c in all_chunks_data]
        # Add text to payload for retrieval
        for payload, text in zip(payloads, texts):
            payload["text"] = text
            
        vector_client.ensure_collection(embedder.get_dimension())
        vector_client.upsert_vectors(embeddings, payloads, vector_ids)
        
        # 3. Update DB
        for i, (text, vid, payload) in enumerate(zip(texts, vector_ids, payloads)):
            chunk = Chunk(
                document_id=doc.id,
                chunk_text=text,
                # vector_id and page_number removed - not in live schema
            )
            db.add(chunk)
            
        doc.status = "completed"
        db.commit()
    except Exception as e:
        doc.status = "error"
        db.commit()
        raise e
