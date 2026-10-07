"""Retriever with MMR and optional metadata filtering."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.config import get_settings


def retrieve(
    query: str,
    store: VectorStore,
    k: int | None = None,
    filter_metadata: dict | None = None,
) -> list[Document]:
    cfg = get_settings()
    k = k or cfg.retrieval_k

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            **({"filter": filter_metadata} if filter_metadata else {}),
        },
    )
    return retriever.invoke(query)


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
