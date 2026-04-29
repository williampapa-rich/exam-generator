"""
LLM 어댑터 — provider-agnostic 인터페이스.

각 provider는 자신의 native structured output 기능을 활용해 Pydantic 모델 객체를 반환합니다.
- Gemini:    response_schema= (Pydantic 직접 전달)
- OpenAI:    response_format={"type": "json_schema", ...}
- Anthropic: tool_use 강제 (input_schema에 model_json_schema 전달)

반환값은 (parsed, UsageInfo) 튜플 — 호출자가 비용 누적 로깅을 위해 토큰 사용량 활용.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class UsageInfo:
    """LLM 호출 1회 분 토큰 사용량 + 메타. cost 환산은 pricing.compute_cost 로 후처리."""
    model:             str
    prompt_tokens:     int
    completion_tokens: int


class LLMClient(Protocol):
    """모든 provider가 구현하는 최소 인터페이스."""

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 8000,
        temperature: float = 0.9,
    ) -> tuple[T, UsageInfo]: ...


def get_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """env LLM_PROVIDER 또는 인자로 provider 선택. model 도 명시 가능."""
    p = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    if p == "gemini":
        from app.llm.gemini import GeminiClient
        return GeminiClient(model=model)
    if p == "openai":
        from app.llm.openai import OpenAIClient
        return OpenAIClient(model=model)
    if p == "anthropic":
        from app.llm.anthropic import AnthropicClient
        return AnthropicClient(model=model)
    raise ValueError(f"Unknown LLM provider: {p!r}. Use one of: gemini, openai, anthropic")
