"""Request and response shapes for the HTTP API (validated by Pydantic)."""

from pydantic import BaseModel, ConfigDict, Field

ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=20000)  # the tighter configurable limit is checked in the service
    app_id: str = Field("default", min_length=1, max_length=64, pattern=ID_PATTERN)
    session_id: str | None = Field(None, max_length=128, pattern=ID_PATTERN)


class Citation(BaseModel):
    ref: int  # the [n] marker used in the answer
    chunk_id: str
    source: str  # knowledge-base file name
    title: str
    section: str
    distance: float | None  # raw ChromaDB distance (lower = closer); not a percentage. None = neighbour added for context
    metric: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]  # chunks the answer cites
    sources: list[Citation]  # every chunk retrieved for this question
    app_id: str
    session_id: str | None
    request_id: str
    model: str
    duration_ms: int


class AppInfo(BaseModel):
    app_id: str
    name: str
    welcome_message: str
    example_questions: list[str]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str | None = None
