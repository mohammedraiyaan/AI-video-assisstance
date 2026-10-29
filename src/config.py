"""Central application configuration."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_k: int = 4
    retrieval_score_threshold: float = 0.05
    whisper_model: str = "small"
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cache_dir: Path = Path("cache")
    results_dir: Path = Path("results")
    max_context_chars: int = 24000
    retry_attempts: int = 3
    llm_min_request_interval: float = 1.2
    audio_chunk_minutes: int = 10
    sarvam_piece_seconds: int = 25
    rrf_constant: int = 60

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_k < 1:
            raise ValueError("RETRIEVAL_K must be greater than zero")
        if not 0 <= self.retrieval_score_threshold <= 1:
            raise ValueError("RETRIEVAL_SCORE_THRESHOLD must be between zero and one")
        if self.retry_attempts < 1 or self.audio_chunk_minutes < 1:
            raise ValueError("Retry count and audio chunk duration must be positive")
        if self.sarvam_piece_seconds < 1 or self.rrf_constant < 1:
            raise ValueError("Sarvam piece duration and RRF constant must be positive")
        if self.max_context_chars < 1:
            raise ValueError("MAX_CONTEXT_CHARS must be greater than zero")
        if self.llm_min_request_interval < 0:
            raise ValueError("LLM_MIN_REQUEST_INTERVAL must not be negative")


def get_settings() -> Settings:
    """Return settings, allowing safe numeric overrides via the environment."""
    return Settings(
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
        retrieval_score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.05")),
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        reranker_model=os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        cache_dir=Path(os.getenv("CACHE_DIR", "cache")),
        results_dir=Path(os.getenv("RESULTS_DIR", "results")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "24000")),
        retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
        llm_min_request_interval=float(os.getenv("LLM_MIN_REQUEST_INTERVAL", "1.2")),
        audio_chunk_minutes=int(os.getenv("AUDIO_CHUNK_MINUTES", "10")),
        sarvam_piece_seconds=int(os.getenv("SARVAM_PIECE_SECONDS", "25")),
        rrf_constant=int(os.getenv("RRF_CONSTANT", "60")),
    )
