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
        from app.llm.base import UsageInfo
        self.calls.append({"system": system, "user": user, "schema": schema})
        if isinstance(self.response, Exception):
            raise self.response
        usage = UsageInfo(model="mock-model", prompt_tokens=100, completion_tokens=50)
        return self.response.model_copy(deep=True), usage


@pytest.mark.asyncio
async def test_generate_one_success():
    # 본문 길이 검증(평가원 평균 ±100자)을 통과하도록 평균(~997자) 근방으로 더미 채움
    fake = Question(number=22, type="요지(22)", points=2.0, answer=3,
                    passage=["lorem ipsum " * 80], question_text="다음 글의 요지...")
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
    # 새 검증 시스템(question_text 비어있음/길이/마커 등) 통과를 위해 최소 fixture 보강
    fake = Question(
        answer=2,
        passage=["lorem ______ ipsum " * 50],   # 빈칸-절 33 평균(~927자) 근처
        question_text="다음 빈칸에 들어갈 말로 가장 적절한 것은?",
        choices=["a", "b", "c", "d", "e"],
    )
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
    # generator.py 의 fallback 메시지 (자동 생성 실패 안내)
    assert "자동 생성에 실패" in q.passage[0]
    assert q.explanation.startswith("[fallback]")


@pytest.mark.asyncio
async def test_generate_all_parallel():
    # 검증 통과를 위해 본문 길이 / 질문문 / 선택지 충족
    fake = Question(
        answer=1,
        passage=["lorem ipsum " * 80],   # ~960자 (요지/주제/제목 평균 ~1000자 근처)
        question_text="다음 글의 요지로 가장 적절한 것은?",
        choices=["a", "b", "c", "d", "e"],
    )
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


# ─── 본문 길이 검증 (validators._check_passage_length) ────────────────────────
# 평가원 5년치 통계(±100자 hard limit) 가 prompts.py / validators.py 양쪽에
# 일관되게 반영되어 있는지 확인.

def test_passage_length_within_tolerance_passes():
    from app.llm.validators import _check_passage_length
    q = Question(
        number=18, type="목적(18)", points=2.0, answer=1,
        passage=["x" * 641],  # 평가원 평균 정확히
        question_text="다음 글의 목적으로 가장 적절한 것은?",
        choices=["a", "b", "c", "d", "e"],
    )
    assert _check_passage_length(q) is None


def test_passage_length_too_short_fails():
    from app.llm.validators import _check_passage_length, LENGTH_TOLERANCE
    # 목적(18) 평균 641 — tolerance 보다 더 짧게 만들어 확실히 미달 처리되도록
    too_short = 641 - LENGTH_TOLERANCE - 50
    q = Question(
        number=18, type="목적(18)", points=2.0, answer=1,
        passage=["x" * too_short],
        question_text="다음 글의 목적으로 가장 적절한 것은?",
        choices=["a", "b", "c", "d", "e"],
    )
    err = _check_passage_length(q)
    assert err is not None
    assert "미달" in err
    assert "목적(18)" in err


def test_passage_length_too_long_fails():
    from app.llm.validators import _check_passage_length, LENGTH_TOLERANCE
    too_long = 641 + LENGTH_TOLERANCE + 50
    q = Question(
        number=18, type="목적(18)", points=2.0, answer=1,
        passage=["x" * too_long],
        question_text="다음 글의 목적으로 가장 적절한 것은?",
        choices=["a", "b", "c", "d", "e"],
    )
    err = _check_passage_length(q)
    assert err is not None
    assert "초과" in err


def test_long_set_uses_group_passage_len():
    """장문독해(43-45) 는 passage + sub_passages 합산 길이로 검증해야 함."""
    from app.llm.validators import _check_passage_length, _measured_length
    q = Question(
        number=43, type="장문독해(43-45)", points=2.0, answer=1,
        passage=["x" * 500],
        sub_passages=[["y" * 500], ["z" * 500], ["w" * 500]],  # 합 2000자, 평균 1999
        question_text="주어진 글 (A) 다음에 이어질 글의 순서로 가장 적절한 것은?",
        choices=["(B)-(D)-(C)","(C)-(B)-(D)","(C)-(D)-(B)","(D)-(B)-(C)","(D)-(C)-(B)"],
    )
    assert _measured_length(q) == 2000
    assert _check_passage_length(q) is None  # 1999 ± 100 안


def test_unknown_type_skips_length_check():
    """type_profiles 에 없는 유형은 길이 검증을 건너뛴다."""
    from app.llm.validators import _check_passage_length
    q = Question(
        number=18, type="존재하지않는유형", points=2.0, answer=1,
        passage=["x" * 10],
        question_text="?",
        choices=["a", "b", "c", "d", "e"],
    )
    assert _check_passage_length(q) is None
