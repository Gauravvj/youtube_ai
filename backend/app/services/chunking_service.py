"""Chunking service – splits transcript segments into overlapping chunks.

Each chunk carries the start timestamp of its first segment and the end
timestamp (start + duration) of its last segment so we can link answers
back to the video timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.config import settings


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """A chunk of transcript text with its time span."""

    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    chunk_index: int


def chunk_segments(
    segments: list[dict],
    *,
    max_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[TranscriptChunk]:
    """Split transcript segments into overlapping text chunks.

    Args:
        segments: List of dicts with ``text``, ``start``, and ``duration``.
        max_chars: Maximum character length per chunk (default from settings).
        overlap_chars: Number of overlapping characters between consecutive
            chunks (default from settings).

    Returns:
        A list of :class:`TranscriptChunk` instances.
    """
    if max_chars is None:
        max_chars = settings.CHUNK_SIZE
    if overlap_chars is None:
        overlap_chars = settings.CHUNK_OVERLAP

    if not segments:
        return []

    # Build a flat list of (word, seg_start, seg_end) tuples so we can
    # track which segment each word originated from.
    _Word = tuple  # (word_str, seg_start, seg_end)
    words: list[_Word] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = seg_start + float(seg["duration"])
        for w in seg["text"].split():
            words.append((w, seg_start, seg_end))

    if not words:
        return []

    chunks: list[TranscriptChunk] = []
    idx = 0
    chunk_index = 0

    while idx < len(words):
        # Accumulate words until we hit the character budget.
        buf: list[str] = []
        buf_len = 0
        start_time = words[idx][1]
        end_time = words[idx][2]
        j = idx

        while j < len(words):
            word = words[j][0]
            added_len = len(word) + (1 if buf else 0)  # space separator
            if buf and buf_len + added_len > max_chars:
                break
            buf.append(word)
            buf_len += added_len
            end_time = words[j][2]
            j += 1

        text = " ".join(buf)
        chunks.append(TranscriptChunk(
            text=text,
            start_time=start_time,
            end_time=end_time,
            chunk_index=chunk_index,
        ))
        chunk_index += 1

        # Advance the pointer, leaving overlap_chars worth of words for next chunk.
        if j >= len(words):
            break

        next_idx = j
        if overlap_chars > 0:
            overlap_len = 0
            k = j - 1
            while k > idx:
                w = words[k][0]
                added = len(w) + (1 if overlap_len > 0 else 0)
                if overlap_len + added > overlap_chars:
                    break
                overlap_len += added
                k -= 1
            next_idx = max(idx + 1, k + 1)

        idx = next_idx

    return chunks
