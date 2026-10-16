"""API route definitions."""

import logging

from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    TimestampReference,
    TranscriptSegment,
    VideoRequest,
    VideoResponse,
)
from backend.app.services.chunking_service import TranscriptChunk, chunk_segments
from backend.app.services.rag_service import RAGServiceError
from backend.app.services.rag_service import ask as rag_ask
from backend.app.services.vector_service import VectorIndex, build_index
from backend.app.services.youtube_service import (
    YouTubeServiceError,
    extract_video_id,
    fetch_transcript,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── In-memory store (single active video) ─────────────────────────────────────

class _VideoStore:
    """Holds the transcript and FAISS index for the currently loaded video."""

    def __init__(self) -> None:
        self.video_id: str | None = None
        self.index: VectorIndex | None = None
        self.language: str | None = None
        self.segment_count: int = 0
        self.chunk_count: int = 0
        self.segments: list[dict] = []
        self.chunks: list[TranscriptChunk] = []

    def clear(self) -> None:
        self.video_id = None
        self.index = None
        self.language = None
        self.segment_count = 0
        self.chunk_count = 0
        self.segments = []
        self.chunks = []


_store = _VideoStore()


def get_video_store() -> _VideoStore:
    """Return the active in-memory video store."""
    return _store


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return a simple health-check response."""
    return HealthResponse()


@router.post(
    "/videos",
    response_model=VideoResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL or video ID"},
        403: {"model": ErrorResponse, "description": "Restricted video"},
        404: {"model": ErrorResponse, "description": "Video / transcript not found"},
        502: {"model": ErrorResponse, "description": "Upstream YouTube error"},
    },
    tags=["Videos"],
    summary="Extract transcript and build RAG index for a YouTube video",
)
async def extract_transcript(body: VideoRequest) -> VideoResponse:
    """Accept a YouTube URL, extract the transcript, chunk it, and build
    a FAISS vector index for RAG queries."""
    try:
        video_id = extract_video_id(body.url)
    except YouTubeServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    try:
        result = fetch_transcript(video_id)
    except YouTubeServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    segments = result["segments"]

    # Chunk and index
    chunks = chunk_segments(segments)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Transcript was empty – nothing to index.",
        )

    index = build_index(chunks)

    # Update the single-video in-memory store
    _store.video_id = result["video_id"]
    _store.index = index
    _store.language = result["language"]
    _store.segment_count = len(segments)
    _store.chunk_count = len(chunks)
    _store.segments = segments
    _store.chunks = chunks

    logger.info(
        "Indexed video %s: %d segments → %d chunks",
        video_id, len(segments), len(chunks),
    )

    transcript_models = [
        TranscriptSegment(
            text=seg["text"],
            start=seg["start"],
            duration=seg["duration"],
        )
        for seg in segments
    ]

    return VideoResponse(
        video_id=result["video_id"],
        language=result["language"],
        segment_count=len(segments),
        chunk_count=len(chunks),
        transcript=transcript_models,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "No video indexed yet"},
        503: {"model": ErrorResponse, "description": "Ollama service unavailable"},
    },
    tags=["Chat"],
    summary="Ask a question about the currently indexed video",
)
async def chat(body: ChatRequest) -> ChatResponse:
    """Answer a user question using RAG over the active video's transcript."""
    if _store.index is None or _store.video_id is None:
        raise HTTPException(
            status_code=409,
            detail="No video has been indexed yet. "
                   "Please POST /videos first to load a video.",
        )

    try:
        result = rag_ask(body.question, _store.index)
    except RAGServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    timestamps = [
        TimestampReference(
            start=ts.start,
            end=ts.end,
            text_preview=ts.text_preview,
        )
        for ts in result.timestamps
    ]

    return ChatResponse(
        answer=result.answer,
        model=result.model,
        timestamps=timestamps,
    )
