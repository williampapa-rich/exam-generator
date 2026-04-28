"""OpenAI provider — openai SDK 기반 (Structured Outputs)."""
from __future__ import annotations

import os
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient:
    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")
            self._client = AsyncOpenAI(api_key=self._api_key)
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
        # Structured Outputs: schema를 그대로 모델에 강제
        completion = await self._get_client().beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(f"OpenAI 응답 parsing 실패: {completion.choices[0].message}")
        return parsed
