from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def list_docs():
    return {"message": "List documents"}
