"""Retriever agent — runs RAG for each sub-query and deduplicates results."""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from src.agents.state import AgentState
from src.rag.retriever import retrieve
from src.rag.vector_store import load_vector_store

logger = logging.getLogger(__name__)

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = load_vector_store()
    return _store


def retriever_node(state: AgentState) -> AgentState:
    store = _get_store()
    sub_queries = state.get("sub_queries") or [state["query"]]

    seen_ids: set[str] = set()
    all_docs: list[Document] = []

    for sq in sub_queries:
        docs = retrieve(sq, store)
        for doc in docs:
            uid = doc.page_content[:120]
            if uid not in seen_ids:
                seen_ids.add(uid)
                all_docs.append(doc)

    logger.info("Retriever gathered %d unique chunks across %d sub-queries", len(all_docs), len(sub_queries))
    return {**state, "retrieved_docs": all_docs}
