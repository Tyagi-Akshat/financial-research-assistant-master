"""Central factory that returns a LangChain chat model based on config."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.config import get_settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    cfg = get_settings()

    if cfg.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=cfg.ollama_model,
            base_url=cfg.ollama_base_url,
            temperature=temperature,
        )

    if cfg.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=cfg.openai_api_key,
            temperature=temperature,
        )

    if cfg.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=cfg.anthropic_api_key,
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM provider: {cfg.llm_provider}")
