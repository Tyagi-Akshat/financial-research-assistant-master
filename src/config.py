from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(__file__).parent.parent


class Settings(BaseModel):
    # LLM
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Vector store
    vector_store_type: str = Field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss"))
    vector_store_path: Path = Field(
        default_factory=lambda: ROOT / os.getenv("VECTOR_STORE_PATH", "data/vectorstore")
    )
    embeddings_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    )

    # SEC
    sec_user_agent: str = Field(
        default_factory=lambda: os.getenv("SEC_USER_AGENT", "ResearchBot research@example.com")
    )

    # API
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Retrieval
    retrieval_k: int = 6
    chunk_size: int = 1024
    chunk_overlap: int = 128


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
