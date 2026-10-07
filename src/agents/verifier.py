"""Verifier / Critic agent — checks every claim in the draft against source documents.

This is the reliability star of the system. It detects hallucinations and returns a
structured verdict so the graph can decide to pass or re-route.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.llm_factory import get_llm
from src.rag.retriever import format_docs

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a rigorous fact-checker for financial research reports.

Given:
1. A DRAFT ANSWER written by an analyst.
2. The SOURCE DOCUMENTS the analyst had access to.

Your job:
- Identify every factual claim in the draft.
- For each claim, check whether it is directly supported by the source documents.
- Return a JSON object with this exact schema:

{
  "is_grounded": true | false,
  "grounded_claims": ["claim 1 ...", ...],
  "unsupported_claims": ["claim A ...", ...],
  "verdict": "PASS | FAIL",
  "explanation": "one-paragraph summary of findings"
}

is_grounded = true if ALL major factual claims are supported.
verdict = "PASS" if is_grounded, else "FAIL".

Return ONLY the JSON object. No markdown fences, no extra text.
"""


def verifier_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0)
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_docs", [])
    context = format_docs(docs)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"DRAFT ANSWER:\n{draft}\n\n"
            f"SOURCE DOCUMENTS:\n{context}"
        )),
    ])

    try:
        result = json.loads(response.content)
        is_grounded = bool(result.get("is_grounded", False))
        unsupported = result.get("unsupported_claims", [])
        explanation = result.get("explanation", "")
    except Exception:
        logger.warning("Verifier JSON parse failed; treating as ungrounded")
        is_grounded = False
        unsupported = ["(could not parse verifier output)"]
        explanation = response.content

    logger.info("Verifier verdict: %s | unsupported claims: %d", "PASS" if is_grounded else "FAIL", len(unsupported))
    return {
        **state,
        "verification_result": explanation,
        "is_grounded": is_grounded,
        "unsupported_claims": unsupported,
    }


def finaliser_node(state: AgentState) -> AgentState:
    """Attach verification metadata to draft to produce the final answer."""
    draft = state.get("draft_answer", "")
    is_grounded = state.get("is_grounded", False)
    unsupported = state.get("unsupported_claims", [])
    explanation = state.get("verification_result", "")

    if is_grounded:
        final = draft + "\n\n---\n*Verification: PASS — all major claims are grounded in source documents.*"
    else:
        caveat_lines = "\n".join(f"- {c}" for c in unsupported) if unsupported else explanation
        final = (
            draft
            + f"\n\n---\n⚠️ **Verification: FAIL** — the following claims could not be verified "
            f"against source documents and may be hallucinated:\n{caveat_lines}"
        )

    return {**state, "final_answer": final}
