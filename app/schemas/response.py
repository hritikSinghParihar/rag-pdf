from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
    status: str = "success"
    message: str = ""
    data: Optional[T] = None

class SuccessResponse(BaseResponse[T]):
    status: str = "success"

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    code: Optional[str] = None
    details: Optional[Any] = None
