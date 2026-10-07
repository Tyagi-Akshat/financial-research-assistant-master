"""Shared LangGraph state schema passed between all agents."""
from __future__ import annotations

from typing import Annotated

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # User's original question
    query: str

    # Planner output: list of sub-questions
    sub_queries: list[str]

    # Retrieved context chunks
    retrieved_docs: list[Document]

    # Analyst's draft answer
    draft_answer: str

    # Verifier output
    verification_result: str
    is_grounded: bool
    unsupported_claims: list[str]

    # Final answer delivered to user
    final_answer: str

    # LangGraph message history (for tool-calling nodes)
    messages: Annotated[list, add_messages]

    # Optional: live market data fetched by analyst
    market_data: dict
