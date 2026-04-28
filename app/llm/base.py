"""
LLM 어댑터 — provider-agnostic 인터페이스.

각 provider는 자신의 native structured output 기능을 활용해 Pydantic 모델 객체를 반환합니다.
- Gemini:    response_schema= (Pydantic 직접 전달)
- OpenAI:    response_format={"type": "json_schema", ...}
- Anthropic: tool_use 강제 (input_schema에 model_json_schema 전달)
"""
from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """모든 provider가 구현하는 최소 인터페이스."""

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
        temperature: float = 0.9,
    ) -> T: ...


def get_client(provider: str | None = None) -> LLMClient:
    """env LLM_PROVIDER 또는 인자로 provider 선택."""
    p = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    if p == "gemini":
        from app.llm.gemini import GeminiClient
        return GeminiClient()
    if p == "openai":
        from app.llm.openai import OpenAIClient
        return OpenAIClient()
    if p == "anthropic":
        from app.llm.anthropic import AnthropicClient
        return AnthropicClient()
    raise ValueError(f"Unknown LLM provider: {p!r}. Use one of: gemini, openai, anthropic")
