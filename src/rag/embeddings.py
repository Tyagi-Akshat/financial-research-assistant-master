"""Shared embeddings instance (sentence-transformers, runs locally)."""
from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    cfg = get_settings()
    return HuggingFaceEmbeddings(
        model_name=cfg.embeddings_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
