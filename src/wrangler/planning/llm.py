"""Chat-model factory. Provider packages are optional extras.

    WRANGLER_LLM=anthropic:claude-sonnet-5-5
    WRANGLER_LLM=openai:gpt-4.1-mini
    WRANGLER_LLM=ollama:llama3.1          # fully local
    WRANGLER_LLM=heuristic                # no LLM (default)
"""

from __future__ import annotations

import os

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4.1-mini",
    "ollama": "llama3.1",
}


def make_chat_model(spec: str):
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    model = model or DEFAULT_MODELS.get(provider, "")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0, max_tokens=4096)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0)
    raise ValueError(f"unknown LLM provider {provider!r}")


def make_planner(spec: str | None = None):
    from .planners import HeuristicPlanner, LLMPlanner

    spec = spec or os.environ.get("WRANGLER_LLM", "heuristic")
    if spec in ("", "heuristic", "none", "offline"):
        return HeuristicPlanner()
    return LLMPlanner(make_chat_model(spec))
