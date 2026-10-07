"""LangGraph workflow: Planner → Retriever → Analyst → Verifier → Finaliser."""
from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agents.analyst import analyst_node
from src.agents.planner import planner_node
from src.agents.retriever_agent import retriever_node
from src.agents.state import AgentState
from src.agents.verifier import finaliser_node, verifier_node

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # how many times the analyst may re-draft on a FAIL verdict


def _should_retry(state: AgentState) -> str:
    """Edge condition: route back to analyst once on failure, then finalise anyway."""
    retries = state.get("_retries", 0)  # type: ignore[typeddict-item]
    if not state.get("is_grounded") and retries < MAX_RETRIES:
        logger.info("Verifier FAIL — routing back to analyst (retry %d)", retries + 1)
        return "retry"
    return "done"


def _increment_retry(state: AgentState) -> AgentState:
    retries = state.get("_retries", 0)  # type: ignore[typeddict-item]
    return {**state, "_retries": retries + 1}  # type: ignore[typeddict-item]


@lru_cache(maxsize=1)
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("finaliser", finaliser_node)
    builder.add_node("increment_retry", _increment_retry)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "analyst")
    builder.add_edge("analyst", "verifier")

    builder.add_conditional_edges(
        "verifier",
        _should_retry,
        {"retry": "increment_retry", "done": "finaliser"},
    )
    builder.add_edge("increment_retry", "analyst")
    builder.add_edge("finaliser", END)

    return builder.compile()


def run_query(query: str) -> AgentState:
    graph = build_graph()
    initial_state: AgentState = {
        "query": query,
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
    result = graph.invoke(initial_state)
    return result
