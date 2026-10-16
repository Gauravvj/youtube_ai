"""YouTube URL validation and transcript extraction service."""

import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
)


# ── URL patterns ──────────────────────────────────────────────────────────────

# Matches an 11-character YouTube video ID (alphanumeric, dash, underscore)
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Supported URL host patterns
_YOUTUBE_HOSTS = {
    "www.youtube.com",
    "youtube.com",
    "m.youtube.com",
    "youtu.be",
}


class YouTubeServiceError(Exception):
    """Base error raised by the YouTube service."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# ── URL parsing ───────────────────────────────────────────────────────────────


def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from a URL string.

    Supported formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID

    Returns:
        The 11-character video ID.

    Raises:
        YouTubeServiceError: If the URL is malformed or no valid ID can be found.
    """
    url = url.strip()

    # If the user pasted a bare video ID, accept it directly.
    if _VIDEO_ID_RE.match(url):
        return url

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise YouTubeServiceError(f"Malformed URL: {exc}") from exc

    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise YouTubeServiceError(
            f"Unsupported host '{host}'. "
            "Please provide a youtube.com or youtu.be link."
        )

    # youtu.be/<VIDEO_ID>
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(video_id):
            return video_id
        raise YouTubeServiceError(
            "Could not extract a valid video ID from the youtu.be link."
        )

    # /watch?v=<VIDEO_ID>
    if parsed.path in ("/watch", "/watch/"):
        qs = parse_qs(parsed.query)
        ids = qs.get("v")
        if ids and _VIDEO_ID_RE.match(ids[0]):
            return ids[0]
        raise YouTubeServiceError(
            "Missing or invalid 'v' parameter in watch URL."
        )

    # /shorts/<VIDEO_ID>, /embed/<VIDEO_ID>, /v/<VIDEO_ID>
    path_prefixes = ("/shorts/", "/embed/", "/v/")
    for prefix in path_prefixes:
        if parsed.path.startswith(prefix):
            video_id = parsed.path[len(prefix):].split("/")[0].split("?")[0]
            if _VIDEO_ID_RE.match(video_id):
                return video_id
            raise YouTubeServiceError(
                f"Invalid video ID in '{prefix.strip('/')}' URL."
            )

    raise YouTubeServiceError(
        "Could not extract a video ID. "
        "Please provide a valid YouTube watch, Shorts, or share link."
    )


# ── Transcript fetching ──────────────────────────────────────────────────────


def fetch_transcript(
    video_id: str,
    languages: tuple[str, ...] = ("en",),
) -> dict:
    """Fetch the transcript for a given video ID.

    Args:
        video_id: An 11-character YouTube video ID.
        languages: Priority-ordered language codes to look for.

    Returns:
        A dict with keys:
            video_id  (str):  The video ID.
            language  (str):  Language code of the transcript used.
            segments  (list): Each segment is a dict with text, start, duration.

    Raises:
        YouTubeServiceError on any retrieval failure.
    """
    if not _VIDEO_ID_RE.match(video_id):
        raise YouTubeServiceError(f"Invalid video ID format: '{video_id}'")

    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.list(video_id)
    except VideoUnavailable:
        raise YouTubeServiceError(
            f"Video '{video_id}' is unavailable or does not exist.",
            status_code=404,
        )
    except VideoUnplayable:
        raise YouTubeServiceError(
            f"Video '{video_id}' is unplayable (may be private or region-locked).",
            status_code=403,
        )
    except AgeRestricted:
        raise YouTubeServiceError(
            f"Video '{video_id}' is age-restricted. "
            "Cannot fetch transcript without authentication.",
            status_code=403,
        )
    except Exception as exc:
        raise YouTubeServiceError(
            f"Failed to retrieve transcript list: {exc}",
            status_code=502,
        ) from exc

    # Attempt to find a transcript in the requested languages.
    try:
        transcript = transcript_list.find_transcript(list(languages))
    except NoTranscriptFound:
        raise YouTubeServiceError(
            f"No transcript found for video '{video_id}' in languages: "
            f"{', '.join(languages)}. "
            "Try specifying a different language.",
            status_code=404,
        )
    except TranscriptsDisabled:
        raise YouTubeServiceError(
            f"Transcripts are disabled for video '{video_id}'.",
            status_code=403,
        )

    # Fetch the actual captions.
    try:
        fetched = transcript.fetch()
    except Exception as exc:
        raise YouTubeServiceError(
            f"Failed to fetch transcript content: {exc}",
            status_code=502,
        ) from exc

    segments = [
        {
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration,
        }
        for snippet in fetched
    ]

    return {
        "video_id": video_id,
        "language": fetched.language_code,
        "segments": segments,
    }
