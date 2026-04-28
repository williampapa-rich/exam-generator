"""
PDF 텍스트 추출 — 현재 Gemini Vision만 지원.

다른 provider는 향후 추가 (OpenAI Files API + vision, Anthropic PDF beta).
"""
from __future__ import annotations

import os

from google import genai

from app.llm.gemini import DEFAULT_MODEL


_PDF_PROMPT = (
    "이 PDF는 영어 시험지 또는 교재입니다. "
    "주요 지문 텍스트, 주제, 핵심 어휘를 추출해주세요. "
    "원문 영어 텍스트를 최대한 보존해서 정리해주세요."
)


async def extract_reference_text(pdf_bytes: bytes) -> str:
    """PDF → 텍스트 추출 (Gemini Vision, async-native)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("PDF 추출 실패: GEMINI_API_KEY 미설정")
        return ""

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=api_key)

    try:
        resp = await client.aio.models.generate_content(
            model=model_name,
            contents=[
                genai.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                _PDF_PROMPT,
            ],
            config=genai.types.GenerateContentConfig(max_output_tokens=3000),
        )
        return resp.text or ""
    except Exception as e:
        print(f"PDF 추출 오류: {e}")
        return ""
