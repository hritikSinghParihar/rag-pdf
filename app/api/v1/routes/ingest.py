import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.models import get_db
from app.services.ingestion_service import ingestion_service
from app.services.rbi_service import rbi_service
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
    # Save file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        doc = ingestion_service.process_upload(db, temp_path, current_user.id)
        # In a real app, this would be a background task
        from app.pipeline.orchestrator import process_document_pipeline
        process_document_pipeline(db, doc.id, temp_path)
        
        return SuccessResponse(
            message="Document uploaded and processed successfully",
            data={"document_id": str(doc.id), "filename": doc.file_name}
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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
