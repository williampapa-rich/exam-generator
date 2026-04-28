"""Gemini provider — google-genai 기반 (async-native client.aio.*)."""
from __future__ import annotations

import os
from typing import TypeVar

from google import genai
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.5-flash-lite"


class GeminiClient:
    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("GEMINI_API_KEY 가 설정되지 않았습니다.")
            self._client = genai.Client(api_key=self._api_key)
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
        # async-native: client.aio.* 사용 → run_in_executor 불필요 + httpx 동시성 안전
        resp = await self._get_client().aio.models.generate_content(
            model=self._model,
            contents=user,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        if hasattr(resp, "parsed") and isinstance(resp.parsed, schema):
            return resp.parsed
        return schema.model_validate_json(resp.text)
