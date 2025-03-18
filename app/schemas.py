from pydantic import BaseModel
from typing import Optional


class SummaryRequest(BaseModel):
    text: str
    max_length: Optional[int] = None


class SummaryResponse(BaseModel):
    summary: str
