"""LLM 어댑터 테스트 — provider별 인스턴스화 + factory + Mock으로 generator 흐름."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.llm.base import LLMClient, get_client
from app.llm.gemini import GeminiClient, DEFAULT_MODEL as GEMINI_DEFAULT
from app.llm.openai import OpenAIClient, DEFAULT_MODEL as OPENAI_DEFAULT
from app.llm.anthropic import AnthropicClient, DEFAULT_MODEL as ANTHROPIC_DEFAULT
from app.llm.generator import generate_one, generate_all, _fallback
from shared.schemas.question import Question, QuestionConfig


# ─── factory ─────────────────────────────────────────────────────────────────

def test_factory_default_is_gemini():
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=False):
        assert isinstance(get_client(), GeminiClient)


def test_factory_explicit_provider():
    assert isinstance(get_client("openai"), OpenAIClient)
    assert isinstance(get_client("anthropic"), AnthropicClient)
    assert isinstance(get_client("Gemini"), GeminiClient)  # case-insensitive


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_client("groq")


# ─── 각 provider 인스턴스화 (실제 호출 X) ────────────────────────────────────

def test_gemini_instantiable_with_dummy_key():
    c = GeminiClient(api_key="dummy", model="custom-model")
    assert c._model == "custom-model"
    assert c._api_key == "dummy"


def test_openai_uses_default_model():
    c = OpenAIClient(api_key="dummy")
    assert c._model == OPENAI_DEFAULT


def test_anthropic_uses_default_model():
    c = AnthropicClient(api_key="dummy")
    assert c._model == ANTHROPIC_DEFAULT


def test_provider_missing_key_raises_on_use():
    with patch.dict(os.environ, {}, clear=True):
        c = GeminiClient()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            c._get_client()


# ─── generator 흐름 (Mock client) ─────────────────────────────────────────────

class MockClient:
    """generate_json 호출 시 미리 정한 Question의 *복사본*을 반환 (실제 provider 동작 모사)."""
    def __init__(self, response: Question | Exception):
        self.response = response
        self.calls: list[dict] = []

    async def generate_json(self, *, system, user, schema, max_tokens=2000, temperature=0.9):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response.model_copy(deep=True)


@pytest.mark.asyncio
async def test_generate_one_success():
    fake = Question(number=22, type="요지(22)", points=2.0, answer=3,
                    passage=["a", "b"], question_text="다음 글의 요지...")
    cli = MockClient(fake)
    cfg = QuestionConfig(number=22, type="요지(22)", points=2.0)
    q = await generate_one(cfg, "고3 (수능)", client=cli)
    assert q.number == 22
    assert q.type == "요지(22)"
    assert q.answer_symbol == "③"
    assert len(cli.calls) == 1
    assert cli.calls[0]["schema"] is Question


@pytest.mark.asyncio
async def test_generate_one_overrides_meta_from_config():
    """LLM이 number/type/points를 채우지 않거나 다른 값 줘도 cfg의 값으로 덮어씀."""
    fake = Question(answer=2)  # number/type/points 모두 None
    cli = MockClient(fake)
    cfg = QuestionConfig(number=33, type="빈칸-절(33)", points=2.5)
    q = await generate_one(cfg, "고3 (수능)", client=cli)
    assert q.number == 33
    assert q.type == "빈칸-절(33)"
    assert q.points == 2.5


@pytest.mark.asyncio
async def test_generate_one_falls_back_on_exception():
    cli = MockClient(RuntimeError("API down"))
    cfg = QuestionConfig(number=22, type="요지(22)", points=2.0)
    q = await generate_one(cfg, "고3 (수능)", client=cli)
    assert q.number == 22
    assert "수동 입력" in q.passage[0]
    assert q.explanation == "생성 실패"


@pytest.mark.asyncio
async def test_generate_all_parallel():
    fake = Question(answer=1)
    cli = MockClient(fake)
    cfgs = [
        QuestionConfig(number=22, type="요지(22)", points=2.0),
        QuestionConfig(number=23, type="주제(23)", points=2.0),
        QuestionConfig(number=24, type="제목(24)", points=2.0),
    ]
    out = await generate_all(cfgs, "고3 (수능)", client=cli)
    assert len(out) == 3
    assert [q.number for q in out] == [22, 23, 24]
    assert len(cli.calls) == 3


def test_fallback_question_is_valid():
    fb = _fallback("요지(22)", 22)
    assert isinstance(fb, Question)
    assert fb.number == 22
    assert fb.answer_symbol == "①"
