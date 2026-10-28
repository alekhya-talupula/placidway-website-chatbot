from pydantic import BaseModel, Field, HttpUrl

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=100)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)

class Citation(BaseModel):
    title: str
    url: HttpUrl

class ChatResponse(BaseModel):
    answer: str
    sources: list[Citation]
    grounded: bool

class SourceStatus(BaseModel):
    title: str
    configured: bool
    ingested: bool = False
