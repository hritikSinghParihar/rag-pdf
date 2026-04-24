from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.models.document import Document
from app.models import get_db
from app.integrations.vector_db.qdrant_client import vector_client
from app.pipeline.embedder import embedder
from app.integrations.openai_client import openai_client
from app.core.config import settings
from app.schemas.response import SuccessResponse

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/", response_model=SuccessResponse)
async def query_documents(
    body: QueryRequest,
    db: Session = Depends(get_db)
):
    question = body.question

    # 1. Embed question
    query_vector = embedder.embed_texts([question])[0]
    
    # 2. Search Qdrant
    results = vector_client.search(query_vector, limit=settings.TOP_K)
    
    # 3. Build context
    context_parts = []
    for hit in results:
        payload = hit.payload
        context_parts.append(f"[Source: {payload.get('source_id')}, Page: {payload.get('page')}]\n{payload.get('text')}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # 4. Generate answer
    instruction = "You are a helpful assistant. Answer based ONLY on the context."
    prompt = f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    answer = openai_client.generate_chat_completion(messages)
    
    # 5. Format sources with human-readable names and de-duplicate
    formatted_sources = []
    seen_sources = set()
    
    for hit in results:
        payload = hit.payload
        doc_id = payload.get("source_id")
        page = payload.get("page")
        
        # Create a unique key for de-duplication (normalize to strings/ints)
        source_key = (str(doc_id), int(page) if page is not None else 0)
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            
            # Resolve file name from DB
            doc = db.query(Document).filter(Document.id == doc_id).first()
            file_name = doc.file_name if doc else "Unknown Document"
            
            formatted_sources.append({
                "file_name": file_name,
                "page": page,
                "doc_id": str(doc_id)
            })

    return SuccessResponse(
        message="Answer generated",
        data={
            "answer": answer,
            "sources": formatted_sources
        }
    )
