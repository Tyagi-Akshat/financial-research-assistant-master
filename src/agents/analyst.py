"""Analyst agent — synthesises context + optional live market data into a draft answer."""
from __future__ import annotations

import logging

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.llm_factory import get_llm
from src.rag.retriever import format_docs

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior financial analyst. Using ONLY the provided document
excerpts (and any live market data given), write a thorough, well-structured answer to the
user's question.

Rules:
- Cite every factual claim with [Source N] notation matching the context.
- If a claim cannot be supported by the provided sources, clearly say "I could not find
  information about this in the available documents."
- Do not speculate or use general knowledge not present in the context.
- Use bullet points and clear section headers where appropriate.
"""


def _fetch_market_data(query: str) -> dict:
    """Best-effort: extract tickers from query and pull yfinance summary."""
    import re
    tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
    # filter obvious non-tickers
    tickers = [t for t in tickers if t not in {"SEC", "10K", "Q1", "Q2", "Q3", "Q4", "CEO", "CFO", "EPS", "PE"}]
    data = {}
    for ticker in tickers[:3]:
        try:
            info = yf.Ticker(ticker).fast_info
            data[ticker] = {
                "last_price": getattr(info, "last_price", None),
                "market_cap": getattr(info, "market_cap", None),
                "52w_high": getattr(info, "year_high", None),
                "52w_low": getattr(info, "year_low", None),
            }
        except Exception:
            pass
    return data


def analyst_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.1)
    docs = state.get("retrieved_docs", [])
    context = format_docs(docs)

    market_data = _fetch_market_data(state["query"])

    market_section = ""
    if market_data:
        lines = [f"  {t}: last=${v.get('last_price')}, mkt_cap={v.get('market_cap')}" for t, v in market_data.items()]
        market_section = "\n\nLive Market Data (yfinance):\n" + "\n".join(lines)

    user_msg = (
        f"Question: {state['query']}\n\n"
        f"Document Context:\n{context}"
        f"{market_section}"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    logger.info("Analyst produced draft answer (%d chars)", len(response.content))
    return {**state, "draft_answer": response.content, "market_data": market_data}
