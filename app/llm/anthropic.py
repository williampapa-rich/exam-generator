"""Anthropic provider — anthropic SDK 기반 (tool_use 강제 패턴)."""
from __future__ import annotations

import os
from typing import TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
TOOL_NAME = "emit_question"


class AnthropicClient:
    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._client: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY 가 설정되지 않았습니다.")
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 2000,
        temperature: float = 0.9,
    ) -> T:
        # tool_use 강제: 모델이 무조건 emit_question 도구를 호출하게 함
        # input_schema에 Pydantic JSON Schema 그대로 전달
        msg = await self._get_client().messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            tools=[{
                "name": TOOL_NAME,
                "description": "Emit a question matching the schema.",
                "input_schema": schema.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                return schema.model_validate(block.input)
        raise RuntimeError(f"Anthropic 응답에 {TOOL_NAME} tool_use 블록 없음: {msg.content}")
