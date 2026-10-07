"""Unit tests for individual agent nodes (no real LLM calls — mocked)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.agents.state import AgentState


def _base_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "query": "What was Apple revenue in 2023?",
        "sub_queries": [],
        "retrieved_docs": [],
        "draft_answer": "",
        "verification_result": "",
        "is_grounded": False,
        "unsupported_claims": [],
        "final_answer": "",
        "messages": [],
        "market_data": {},
    }
    defaults.update(kwargs)
    return defaults


# ── Planner ────────────────────────────────────────────────────────────────────

@patch("src.agents.planner.get_llm")
def test_planner_parses_json(mock_get_llm):
    sub_qs = ["What was Apple revenue?", "What were Apple risk factors?"]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(sub_qs))
    mock_get_llm.return_value = mock_llm

    from src.agents.planner import planner_node
    result = planner_node(_base_state())
    assert result["sub_queries"] == sub_qs


@patch("src.agents.planner.get_llm")
def test_planner_fallback_on_bad_json(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="not json at all")
    mock_get_llm.return_value = mock_llm

    from src.agents.planner import planner_node
    state = _base_state(query="Some query")
    result = planner_node(state)
    assert result["sub_queries"] == ["Some query"]


# ── Verifier ───────────────────────────────────────────────────────────────────

@patch("src.agents.verifier.get_llm")
def test_verifier_pass(mock_get_llm):
    payload = {
        "is_grounded": True,
        "grounded_claims": ["Revenue was $383B"],
        "unsupported_claims": [],
        "verdict": "PASS",
        "explanation": "All claims verified.",
    }
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(payload))
    mock_get_llm.return_value = mock_llm

    from src.agents.verifier import verifier_node
    doc = Document(page_content="Revenue was $383B", metadata={"source": "test.txt"})
    state = _base_state(draft_answer="Revenue was $383B", retrieved_docs=[doc])
    result = verifier_node(state)
    assert result["is_grounded"] is True
    assert result["unsupported_claims"] == []


@patch("src.agents.verifier.get_llm")
def test_verifier_fail(mock_get_llm):
    payload = {
        "is_grounded": False,
        "grounded_claims": [],
        "unsupported_claims": ["Revenue was $500B"],
        "verdict": "FAIL",
        "explanation": "Claim not in sources.",
    }
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=json.dumps(payload))
    mock_get_llm.return_value = mock_llm

    from src.agents.verifier import verifier_node
    state = _base_state(draft_answer="Revenue was $500B", retrieved_docs=[])
    result = verifier_node(state)
    assert result["is_grounded"] is False
    assert len(result["unsupported_claims"]) == 1


# ── Finaliser ──────────────────────────────────────────────────────────────────

def test_finaliser_pass_appends_pass_note():
    from src.agents.verifier import finaliser_node
    state = _base_state(draft_answer="Good answer.", is_grounded=True, unsupported_claims=[])
    result = finaliser_node(state)
    assert "PASS" in result["final_answer"]


def test_finaliser_fail_appends_warning():
    from src.agents.verifier import finaliser_node
    state = _base_state(
        draft_answer="Bad answer.",
        is_grounded=False,
        unsupported_claims=["Fabricated claim"],
    )
    result = finaliser_node(state)
    assert "FAIL" in result["final_answer"]
    assert "Fabricated claim" in result["final_answer"]
