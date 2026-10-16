"""Tests for YouTube URL validation, transcript extraction, and API endpoints.

Run with:
    python -m pytest backend/tests/test_youtube.py -v
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.youtube_service import (
    YouTubeServiceError,
    extract_video_id,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Unit tests – extract_video_id
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractVideoId:
    """Pure-logic tests for URL parsing – no network calls."""

    # ── Valid URLs ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "url, expected_id",
        [
            # Standard watch URLs
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Watch URL with extra query params
            (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PLx",
                "dQw4w9WgXcQ",
            ),
            # youtu.be share links
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?t=10", "dQw4w9WgXcQ"),
            # Shorts
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Embed
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Legacy /v/ format
            ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Bare video ID
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # IDs with dashes / underscores
            ("https://youtu.be/abc-_123DEf", "abc-_123DEf"),
        ],
    )
    def test_valid_urls(self, url: str, expected_id: str) -> None:
        assert extract_video_id(url) == expected_id

    # ── Invalid URLs ──────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "url",
        [
            "",                                         # empty
            "   ",                                      # whitespace
            "https://www.google.com",                   # wrong host
            "https://vimeo.com/123456789",              # non-YouTube
            "https://www.youtube.com/watch",            # missing v param
            "https://www.youtube.com/watch?v=short",    # ID too short
            "https://youtu.be/",                        # empty path
            "https://www.youtube.com/shorts/",          # empty shorts ID
            "not_a_url_and_too_long_to_be_an_id_xxxxx", # random garbage
        ],
    )
    def test_invalid_urls(self, url: str) -> None:
        with pytest.raises(YouTubeServiceError):
            extract_video_id(url)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. API endpoint tests (FastAPI TestClient)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """GET /health should always return 200."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestVideosEndpoint:
    """POST /videos – validation-level tests (no network)."""

    def test_missing_body(self, client: TestClient) -> None:
        resp = client.post("/videos")
        assert resp.status_code == 422  # Pydantic validation error

    def test_invalid_url_returns_400(self, client: TestClient) -> None:
        resp = client.post("/videos", json={"url": "https://www.google.com"})
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_empty_url_returns_400(self, client: TestClient) -> None:
        resp = client.post("/videos", json={"url": ""})
        assert resp.status_code == 400
