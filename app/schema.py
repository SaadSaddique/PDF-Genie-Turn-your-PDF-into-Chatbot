from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: str
    score: Optional[float] = None

class Answer(BaseModel):
    answer: str = Field(..., description="Markdown answer with [n] citations.")
    citations: List[Citation] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0, le=1.0)
