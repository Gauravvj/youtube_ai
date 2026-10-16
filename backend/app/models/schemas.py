"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    """Payload for the POST /videos endpoint."""

    url: str = Field(
        ...,
        description="YouTube video URL (watch, youtu.be, or Shorts)",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )


# ── Responses ─────────────────────────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    """A single captioned segment from the video."""

    text: str
    start: float = Field(description="Start time in seconds")
    duration: float = Field(description="Duration of the segment in seconds")


class VideoResponse(BaseModel):
    """Successful response from POST /videos."""

    video_id: str
    title: str | None = Field(
        default=None,
        description="Video title (when available from transcript metadata)",
    )
    language: str = Field(description="Language code of the transcript used")
    segment_count: int = Field(description="Total number of transcript segments")
    chunk_count: int | None = Field(
        default=None,
        description="Number of chunks indexed for RAG",
    )
    transcript: list[TranscriptSegment]


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = "ok"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """Standard error envelope returned on failure."""

    detail: str


# ── RAG / Chat ────────────────────────────────────────────────────────────────

class VideoProcessedResponse(BaseModel):
    """Returned by POST /videos after transcript extraction AND indexing."""

    video_id: str
    language: str
    segment_count: int
    chunk_count: int = Field(description="Number of chunks indexed for RAG")
    message: str = "Video processed and indexed successfully."


class ChatRequest(BaseModel):
    """Payload for the POST /chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        description="The user's question about the video",
        examples=["What does the speaker say about machine learning?"],
    )


class TimestampReference(BaseModel):
    """A reference to a time range in the video."""

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text_preview: str = Field(description="Excerpt of transcript at this position")


class ChatResponse(BaseModel):
    """Response from POST /chat."""

    answer: str
    model: str = Field(description="LLM model that generated the answer")
    timestamps: list[TimestampReference] = Field(
        default_factory=list,
        description="Relevant video positions supporting the answer",
    )
