"""LLM-as-judge fallback evaluator (no RAGAS dependency).

Scores each answer on: faithfulness (0-1), completeness (0-1), hallucination (bool).
Useful as a secondary signal or when RAGAS metrics are unavailable.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_factory import get_llm

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are an expert evaluator of financial research answers.

Given a QUESTION, a REFERENCE ANSWER (ground truth), a GENERATED ANSWER, and the
SOURCE DOCUMENTS used, score the generated answer.

Return ONLY a JSON object:
{
  "faithfulness": <float 0.0-1.0>,       // is every claim grounded in sources?
  "completeness": <float 0.0-1.0>,       // does it cover all key points from ground truth?
  "hallucination_detected": <bool>,       // does it contain claims absent from sources?
  "reasoning": "<one sentence>"
}
"""


def judge_answer(
    question: str,
    generated_answer: str,
    ground_truth: str,
    source_contexts: list[str],
) -> dict:
    llm = get_llm(temperature=0.0)
    context_text = "\n---\n".join(source_contexts[:4])

    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=(
            f"QUESTION:\n{question}\n\n"
            f"GROUND TRUTH:\n{ground_truth}\n\n"
            f"GENERATED ANSWER:\n{generated_answer}\n\n"
            f"SOURCE DOCUMENTS:\n{context_text}"
        )),
    ])

    try:
        return json.loads(response.content)
    except Exception:
        logger.warning("LLM judge parse failed")
        return {
            "faithfulness": 0.0,
            "completeness": 0.0,
            "hallucination_detected": True,
            "reasoning": response.content,
        }
