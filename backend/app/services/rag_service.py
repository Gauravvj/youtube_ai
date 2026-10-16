"""RAG service – retrieval-augmented generation over video transcripts.

Ties together the vector index, Ollama LLM, and prompt engineering to
answer user questions with evidence drawn from the currently loaded
transcript.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import ollama as ollama_client

from backend.app.core.config import settings
from backend.app.services.chunking_service import TranscriptChunk
from backend.app.services.vector_service import VectorIndex

logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    """Raised when the RAG service encounters an unrecoverable error."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TimestampRef:
    """A reference to a position in the video."""

    start: float
    end: float
    text_preview: str


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    """Value object returned by :func:`ask`."""

    answer: str
    timestamps: list[TimestampRef] = field(default_factory=list)
    model: str = ""


# ── Prompt construction ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about a YouTube video based ONLY on the provided transcript excerpts.

Rules:
1. Base your answer strictly on the transcript evidence provided below.
2. If the available transcript does not contain sufficient information to answer the question, state: "The available transcript does not contain sufficient information to answer this question."
3. When referencing information, mention relevant timestamps in [MM:SS] format where possible.
4. Keep your answer concise, accurate, and factual. Do not speculate or introduce outside facts."""


def _build_user_prompt(
    question: str,
    evidence: list[tuple[TranscriptChunk, float]],
) -> str:
    """Build the user-turn prompt from retrieved evidence."""
    lines = ["=== Transcript Evidence ===\n"]
    for chunk, _score in evidence:
        start_ts = _fmt_ts(chunk.start_time)
        end_ts = _fmt_ts(chunk.end_time)
        lines.append(f"[{start_ts} – {end_ts}]  {chunk.text}\n")

    lines.append("\n=== Question ===")
    lines.append(question)
    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ── Public API ────────────────────────────────────────────────────────────────


def ask(
    question: str,
    index: VectorIndex,
    *,
    top_k: int | None = None,
    _generate_fn=None,
) -> ChatAnswer:
    """Ask a question and get a RAG-grounded answer.

    Args:
        question: The user's natural-language question.
        index: The vector index for the currently loaded video.
        top_k: Number of evidence chunks to retrieve.
        _generate_fn: (Testing) Optional callable replacing Ollama call.

    Returns:
        A :class:`ChatAnswer` containing the answer text and timestamps.
    """
    evidence = index.search(question, top_k=top_k)

    if not evidence:
        return ChatAnswer(
            answer="The available transcript does not contain sufficient information to answer this question.",
            timestamps=[],
            model=settings.OLLAMA_MODEL,
        )

    user_prompt = _build_user_prompt(question, evidence)

    generate = _generate_fn or _ollama_chat
    try:
        raw_answer, model_name = generate(user_prompt)
    except RAGServiceError:
        raise
    except Exception as exc:
        logger.error("Error communicating with Ollama: %s", exc)
        raise RAGServiceError(
            f"Failed to generate answer from Ollama: {exc}",
            status_code=503,
        ) from exc

    # Build timestamp references from the evidence chunks.
    ts_refs = [
        TimestampRef(
            start=chunk.start_time,
            end=chunk.end_time,
            text_preview=chunk.text[:120],
        )
        for chunk, _ in evidence
    ]

    return ChatAnswer(answer=raw_answer, timestamps=ts_refs, model=model_name)


def _ollama_chat(user_prompt: str) -> tuple[str, str]:
    """Call the local Ollama instance.

    Returns:
        A tuple of (response_text, model_name).
    """
    model = settings.OLLAMA_MODEL
    host = settings.OLLAMA_BASE_URL
    try:
        client = ollama_client.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response["message"]["content"]
        return content, model
    except Exception as exc:
        raise RAGServiceError(
            f"Ollama connection error at {host}: {exc}",
            status_code=503,
        ) from exc
