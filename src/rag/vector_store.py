"""Build and persist the FAISS / Chroma vector store."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.config import get_settings
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def build_vector_store(chunks: list[Document]) -> VectorStore:
    cfg = get_settings()
    embeddings = get_embeddings()
    cfg.vector_store_path.mkdir(parents=True, exist_ok=True)

    if cfg.vector_store_type == "faiss":
        from langchain_community.vectorstores import FAISS
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(cfg.vector_store_path))
        logger.info("FAISS index saved to %s", cfg.vector_store_path)
        return store

    # chroma
    from langchain_community.vectorstores import Chroma
    store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(cfg.vector_store_path),
    )
    logger.info("Chroma store saved to %s", cfg.vector_store_path)
    return store


def load_vector_store() -> VectorStore:
    cfg = get_settings()
    embeddings = get_embeddings()

    if cfg.vector_store_type == "faiss":
        from langchain_community.vectorstores import FAISS
        if not (cfg.vector_store_path / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index at {cfg.vector_store_path}. Run the ingestion script first."
            )
        return FAISS.load_local(
            str(cfg.vector_store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    from langchain_community.vectorstores import Chroma
    return Chroma(
        persist_directory=str(cfg.vector_store_path),
        embedding_function=embeddings,
    )
