"""Tests for the Phase 3 RAG pipeline.

Covers:
- Transcript chunking with timestamp preservation and overlap.
- Embedding and FAISS CPU vector index search.
- RAG generation with mocked LLM responses (no Ollama required).
- API endpoints: POST /videos (indexing) and POST /chat (Q&A).
- Single active video in-memory store management.

Run with:
    python -m pytest backend/tests/test_rag.py -v
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import get_video_store
from backend.app.main import app
from backend.app.services.chunking_service import TranscriptChunk, chunk_segments
from backend.app.services.rag_service import ask as rag_ask
from backend.app.services.vector_service import VectorIndex, build_index


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def clean_video_store():
    """Ensure in-memory video store is cleared before and after each test."""
    store = get_video_store()
    store.clear()
    yield
    store.clear()


@pytest.fixture()
def client():
    """Create a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture()
def sample_segments():
    """Sample YouTube transcript segments."""
    return [
        {
            "text": "Welcome to this introduction to artificial intelligence.",
            "start": 0.0,
            "duration": 4.5,
        },
        {
            "text": "Machine learning is a subset of AI focused on training models.",
            "start": 4.5,
            "duration": 5.0,
        },
        {
            "text": "Neural networks are inspired by the human brain architecture.",
            "start": 9.5,
            "duration": 4.0,
        },
        {
            "text": "Transformers revolutionized natural language processing in 2017.",
            "start": 13.5,
            "duration": 5.5,
        },
        {
            "text": "Large language models can understand and generate text accurately.",
            "start": 19.0,
            "duration": 6.0,
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chunking Service Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkingService:
    """Tests for chunk_segments."""

    def test_chunking_empty_segments(self) -> None:
        chunks = chunk_segments([])
        assert chunks == []

    def test_chunking_single_segment(self) -> None:
        segments = [{"text": "Hello world", "start": 2.0, "duration": 3.0}]
        chunks = chunk_segments(segments, max_chars=100, overlap_chars=20)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].start_time == 2.0
        assert chunks[0].end_time == 5.0
        assert chunks[0].chunk_index == 0

    def test_chunking_splits_and_preserves_timestamps(
        self, sample_segments: list[dict]
    ) -> None:
        chunks = chunk_segments(sample_segments, max_chars=80, overlap_chars=20)
        assert len(chunks) > 1

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.start_time >= 0.0
            assert chunk.end_time >= chunk.start_time
            assert len(chunk.text) > 0
            assert len(chunk.text) <= 120  # bounded by word boundaries

    def test_chunking_creates_overlap(self, sample_segments: list[dict]) -> None:
        chunks = chunk_segments(sample_segments, max_chars=70, overlap_chars=25)
        assert len(chunks) >= 2

        # Check that consecutive chunks share common words
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert len(first_words.intersection(second_words)) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Vector Index & Search Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVectorService:
    """Tests for VectorIndex and FAISS CPU similarity search."""

    def test_vector_index_manual_embeddings(self) -> None:
        chunks = [
            TranscriptChunk("cats and dogs", 0.0, 5.0, 0),
            TranscriptChunk("quantum physics theory", 5.0, 10.0, 1),
        ]
        # Normalized orthogonal unit vectors in 2D
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = VectorIndex(chunks, embeddings)

        assert index.size == 2

        # Mock embed_texts to return query vector aligned with quantum physics
        with patch("backend.app.services.vector_service.embed_texts") as mock_emb:
            mock_emb.return_value = np.array([[0.0, 1.0]], dtype=np.float32)
            results = index.search("quantum mechanics", top_k=1)

            assert len(results) == 1
            matched_chunk, score = results[0]
            assert matched_chunk.text == "quantum physics theory"
            assert pytest.approx(score, 0.01) == 1.0

    def test_vector_index_empty(self) -> None:
        embeddings = np.empty((0, 4), dtype=np.float32)
        index = VectorIndex([], embeddings)
        assert index.size == 0
        assert index.search("any query") == []

    def test_build_index_with_sentence_transformers(
        self, sample_segments: list[dict]
    ) -> None:
        chunks = chunk_segments(sample_segments, max_chars=120, overlap_chars=30)
        index = build_index(chunks)

        assert index.size == len(chunks)
        results = index.search("What are neural networks?", top_k=2)
        assert len(results) == 2
        # The top result should be the neural networks chunk
        top_chunk, score = results[0]
        assert "neural networks" in top_chunk.text.lower()
        assert score > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RAG Service Tests (Mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRAGService:
    """Tests for RAG Q&A logic using mocked Ollama generation."""

    def test_rag_ask_success_with_mocked_llm(
        self, sample_segments: list[dict]
    ) -> None:
        chunks = chunk_segments(sample_segments, max_chars=120, overlap_chars=30)
        index = build_index(chunks)

        def mock_generate(prompt: str) -> tuple[str, str]:
            assert "=== Transcript Evidence ===" in prompt
            assert "=== Question ===" in prompt
            return (
                "Neural networks are inspired by human brain architecture.",
                "llama3.2:3b",
            )

        answer = rag_ask(
            "What inspired neural networks?",
            index,
            top_k=2,
            _generate_fn=mock_generate,
        )

        assert answer.model == "llama3.2:3b"
        assert "human brain architecture" in answer.answer
        assert len(answer.timestamps) == 2
        assert answer.timestamps[0].start >= 0.0
        assert answer.timestamps[0].end >= answer.timestamps[0].start
        assert len(answer.timestamps[0].text_preview) > 0

    def test_rag_ask_empty_evidence(self) -> None:
        embeddings = np.empty((0, 4), dtype=np.float32)
        index = VectorIndex([], embeddings)

        answer = rag_ask("Any question?", index)
        assert "not contain sufficient information" in answer.answer
        assert answer.timestamps == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. API Integration Tests (POST /videos and POST /chat)
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatEndpoint:
    """Integration tests for POST /chat and end-to-end RAG flow."""

    def test_chat_without_indexed_video_returns_409(
        self, client: TestClient
    ) -> None:
        resp = client.post("/chat", json={"question": "What is this video about?"})
        assert resp.status_code == 409
        assert "No video has been indexed yet" in resp.json()["detail"]

    def test_chat_missing_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_empty_question_returns_422(self, client: TestClient) -> None:
        resp = client.post("/chat", json={"question": ""})
        assert resp.status_code == 422

    @patch("backend.app.api.routes.fetch_transcript")
    @patch("backend.app.services.rag_service._ollama_chat")
    def test_full_rag_flow(
        self,
        mock_ollama,
        mock_fetch,
        client: TestClient,
        sample_segments: list[dict],
    ) -> None:
        # 1. Mock YouTube transcript retrieval
        mock_fetch.return_value = {
            "video_id": "dQw4w9WgXcQ",
            "language": "en",
            "segments": sample_segments,
        }

        # 2. Call POST /videos to index the transcript
        video_resp = client.post(
            "/videos",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert video_resp.status_code == 200
        video_data = video_resp.json()
        assert video_data["video_id"] == "dQw4w9WgXcQ"
        assert video_data["language"] == "en"
        assert video_data["segment_count"] == 5
        assert video_data["chunk_count"] > 0
        assert len(video_data["transcript"]) == 5

        # Verify in-memory store is populated
        store = get_video_store()
        assert store.video_id == "dQw4w9WgXcQ"
        assert store.index is not None
        assert store.chunk_count == video_data["chunk_count"]

        # 3. Mock Ollama response for chat
        mock_ollama.return_value = (
            "Transformers were introduced in 2017 and revolutionized NLP [00:13].",
            "llama3.2:3b",
        )

        # 4. Call POST /chat
        chat_resp = client.post(
            "/chat",
            json={"question": "When were transformers introduced?"},
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert "Transformers were introduced in 2017" in chat_data["answer"]
        assert chat_data["model"] == "llama3.2:3b"
        assert len(chat_data["timestamps"]) > 0

        # Check timestamp format
        first_ts = chat_data["timestamps"][0]
        assert "start" in first_ts
        assert "end" in first_ts
        assert "text_preview" in first_ts
        assert first_ts["end"] >= first_ts["start"]

        # 5. Call POST /chat a second time to ensure memory store is reused
        # fetch_transcript must NOT be called again
        initial_fetch_call_count = mock_fetch.call_count
        client.post(
            "/chat",
            json={"question": "What are neural networks inspired by?"},
        )
        assert mock_fetch.call_count == initial_fetch_call_count
