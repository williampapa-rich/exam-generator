"""SSOT (shared.schemas.question) 검증 테스트."""
import json

import pytest
from pydantic import ValidationError

from shared.schemas.question import (
    CIRCLED, ExamMeta, GenerateRequest, Question, QuestionConfig, SubQuestion,
)


def test_circled_constant():
    assert CIRCLED == ("①", "②", "③", "④", "⑤")


def test_question_defaults():
    q = Question()
    assert q.number is None
    assert q.passage == []
    assert q.choices == list(CIRCLED)
    assert q.answer == 1
    assert q.answer_symbol == "①"


def test_question_answer_symbol():
    q = Question(number=22, type="요지(22)", answer=3)
    assert q.answer_symbol == "③"


def test_question_passage_text():
    q = Question(number=22, type="요지(22)", passage=["a", "b", "c"])
    assert q.passage_text == "a b c"


def test_question_answer_range():
    with pytest.raises(ValidationError):
        Question(number=22, type="요지(22)", answer=0)
    with pytest.raises(ValidationError):
        Question(number=22, type="요지(22)", answer=6)


def test_question_round_trip():
    """LLM 출력 dict → Question → dict round-trip"""
    raw = {
        "number": 22, "type": "요지(22)", "points": 2.0,
        "passage": ["a", "b"], "question_text": "다음 글의...",
        "choices": ["c1", "c2", "c3", "c4", "c5"],
        "answer": 3, "explanation": "정답 근거",
    }
    q = Question.model_validate(raw)
    assert q.number == 22
    assert q.answer_symbol == "③"
    out = q.model_dump(exclude_none=True)
    assert out["answer"] == 3
    assert "given_sentence" not in out  # None은 제외


def test_question_with_sub_questions():
    """장문 세트 (43-45)"""
    q = Question(
        number=43, type="장문독해(43-45)",
        sub_questions=[
            SubQuestion(question_text="Q1", choices=["a"]*5, answer=1),
            SubQuestion(question_text="Q2", choices=["b"]*5, answer=4),
        ],
    )
    assert q.sub_questions is not None
    assert len(q.sub_questions) == 2
    assert q.sub_questions[1].answer == 4


def test_examMeta_layout_literal():
    ExamMeta(layout="2단 (수능형)")
    ExamMeta(layout="1단")
    with pytest.raises(ValidationError):
        ExamMeta(layout="3단")


def test_generate_request_round_trip():
    req = GenerateRequest(
        meta=ExamMeta(title="t", date="2026-04-28"),
        questions=[QuestionConfig(number=22, type="요지(22)", points=2.0)],
    )
    payload = req.model_dump_json()
    parsed = GenerateRequest.model_validate_json(payload)
    assert parsed.meta.title == "t"
    assert parsed.questions[0].number == 22


def test_question_json_schema_is_serializable():
    schema = Question.model_json_schema()
    # JSON 직렬화 가능해야 templates/schema.json 으로 떨어진다
    json.dumps(schema)
    assert "properties" in schema
    assert "number" in schema["properties"]
