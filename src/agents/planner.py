"""Planner agent — decomposes a complex query into focused sub-queries."""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.llm_factory import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial research planner. Your job is to break a complex
financial question into 2-4 focused sub-questions that, when answered individually,
will together fully answer the original question.

Return ONLY a valid JSON array of strings. No markdown, no explanation.
Example output: ["What was Apple's revenue in 2023?", "What were Apple's main risk factors in 2023?"]
"""


def planner_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0)
    query = state["query"]

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Decompose this query into sub-questions:\n{query}"),
    ])

    try:
        sub_queries = json.loads(response.content)
        if not isinstance(sub_queries, list):
            raise ValueError("Expected a JSON list")
    except Exception:
        logger.warning("Planner JSON parse failed; falling back to single query")
        sub_queries = [query]

    logger.info("Planner produced %d sub-queries", len(sub_queries))
    return {**state, "sub_queries": sub_queries}
