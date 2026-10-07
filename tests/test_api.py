"""Integration tests for the FastAPI endpoints (mocked pipeline)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api.main import app

client = TestClient(app)


def _mock_state():
    return {
        "query": "Test question?",
        "sub_queries": ["Sub Q1"],
        "retrieved_docs": [Document(page_content="Context", metadata={"source": "test.txt"})],
        "draft_answer": "Draft answer.",
        "verification_result": "All claims verified.",
        "is_grounded": True,
        "unsupported_claims": [],
        "final_answer": "Final answer. --- *Verification: PASS*",
        "messages": [],
        "market_data": {},
    }


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@patch("src.api.main.run_query", return_value=_mock_state())
def test_query_endpoint(mock_run):
    resp = client.post("/query", json={"query": "What was Apple revenue in 2023?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_grounded"] is True
    assert "Final answer" in data["final_answer"]
    assert len(data["sources"]) == 1


@patch("src.api.main.run_query", side_effect=FileNotFoundError("No vector store"))
def test_query_no_vectorstore(mock_run):
    resp = client.post("/query", json={"query": "Something?"})
    assert resp.status_code == 503
