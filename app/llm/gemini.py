"""Gemini provider — google-genai 기반 (async-native client.aio.*)."""
from __future__ import annotations

import logging
import os
from typing import TypeVar

from google import genai
from pydantic import BaseModel

from app.llm.base import UsageInfo

T = TypeVar("T", bound=BaseModel)
_log = logging.getLogger("llm.gemini")

DEFAULT_MODEL = "gemini-2.5-flash"  # flash-lite는 JSON truncation 빈발 → flash로 상향


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
        max_tokens: int = 8000,
        temperature: float = 0.9,
    ) -> tuple[T, UsageInfo]:
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
        # finish_reason / token usage 로깅 — JSON truncation 추적용
        finish_reason = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            cands = getattr(resp, "candidates", None) or []
            if cands:
                finish_reason = str(getattr(cands[0], "finish_reason", "") or "")
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                _log.info(
                    "gemini call: model=%s finish=%s prompt_tokens=%s completion_tokens=%s",
                    self._model, finish_reason, prompt_tokens, completion_tokens,
                )
            if finish_reason.upper().endswith("MAX_TOKENS"):
                raise RuntimeError(
                    f"Gemini 응답이 max_output_tokens={max_tokens} 에 걸려 잘림. "
                    f"프롬프트 축소 또는 max_tokens 상향 필요."
                )
        except RuntimeError:
            raise
        except Exception:
            pass

        usage_info = UsageInfo(
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if hasattr(resp, "parsed") and isinstance(resp.parsed, schema):
            return resp.parsed, usage_info
        return schema.model_validate_json(resp.text), usage_info
