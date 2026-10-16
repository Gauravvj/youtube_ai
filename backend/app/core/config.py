"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the backend server.

    Values are loaded from a .env file (if present) or environment variables.
    """

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS – comma-separated origins accepted by the frontend dev server
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Ollama (not used in Task 2, kept for later phases)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # Embeddings & RAG (not used in Task 2)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 4

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def cors_origin_list(self) -> list[str]:
        """Return CORS origins as a list."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


# Singleton – import this from anywhere
settings = Settings()
