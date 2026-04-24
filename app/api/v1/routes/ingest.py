import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.models import get_db
from app.services.ingestion_service import ingestion_service
from app.services.rbi_service import rbi_service
from app.workers.ingest_worker import process_document_task
from app.models.document import SyncJob
from app.core.dependencies import get_current_user
from app.schemas.response import SuccessResponse

router = APIRouter()

@router.post("/upload", response_model=SuccessResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # Validate file extension
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.html', '.htm', '.txt', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Save file to uploads directory
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        doc = ingestion_service.process_upload(db, file_path, current_user.id)
        # Trigger background processing via Celery
        process_document_task.delay(doc.id, file_path)
        
        return SuccessResponse(
            message="Document uploaded and processing started in background",
            data={"document_id": str(doc.id), "filename": doc.file_name, "status": "processing"}
        )
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e

@router.post("/rbi-sync", response_model=SuccessResponse)
async def sync_rbi_documents(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    # Create a sync job record
    job = SyncJob(status="running")
    db.add(job)
    db.commit()
    db.refresh(job)
    
    background_tasks.add_task(rbi_service.sync_rbi_documents, db, current_user.id, str(job.id))
    return SuccessResponse(
        message="RBI documents sync started in background",
        data={"job_id": str(job.id), "status": "started"}
    )

@router.get("/sync-status/{job_id}", response_model=SuccessResponse)
async def get_sync_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    job = db.query(SyncJob).filter(SyncJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")
        
    return SuccessResponse(
        message="Sync job status retrieved",
        data={
            "id": str(job.id),
            "status": job.status,
            "total_files": job.total_files,
            "synced_files": job.synced_files,
            "error_files": job.error_files,
            "message": job.message,
            "updated_at": str(job.updated_at)
        }
    )
