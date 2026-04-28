"""
Provider-agnostic 문제 생성기.

LLM provider(Gemini/OpenAI/Anthropic) 선택은 base.get_client()가 담당.
각 provider는 native structured output으로 Question 모델을 직접 반환합니다.

장문 세트 (41-42, 43-45) 처리:
  사용자 입력은 1 cfg지만 결과는 sub_questions 갯수만큼 Question이 펼쳐져서 반환.
  number는 cfg.number 부터 +1, +2... 자동 부여.
  passage/sub_passages 는 첫 번째 펼친 Question 에만 보존, 나머지는 빈 본문.
"""
from __future__ import annotations

import asyncio

from app.llm.base import LLMClient, get_client
from app.llm.prompts import prompt_for_type
from shared.schemas.question import Question, QuestionConfig, TYPE_SLOT_COUNT


def _fallback(q_type: str, q_num: int) -> Question:
    return Question(
        number=q_num,
        type=q_type,
        passage=[f"[{q_type} 지문 생성 실패 — 수동 입력 필요]"],
        question_text=f"[{q_type}] 문제를 생성하지 못했습니다.",
        answer=1,
        explanation="생성 실패",
    )


async def generate_one(
    cfg: QuestionConfig,
    grade: str,
    reference_text: str = "",
    *,
    client: LLMClient | None = None,
) -> Question:
    system, user = prompt_for_type(cfg.type, grade, cfg.points, reference_text)
    cli = client or get_client()

    try:
        q = await cli.generate_json(
            system=system,
            user=user,
            schema=Question,
        )
    except Exception as e:
        print(f"[Q{cfg.number}] LLM 호출 실패: {type(e).__name__}: {e}")
        return _fallback(cfg.type, cfg.number)

    # provider가 number/type/points를 못 채울 수 있으니 명시적으로 강제
    q.number = cfg.number
    q.type   = cfg.type
    q.points = cfg.points
    return q


def _expand_sub_questions(q: Question, cfg: QuestionConfig) -> list[Question]:
    """장문 세트(P10/P11): 메인 질문(q.question_text) + sub_questions 를 N개 Question으로 펼침.

    prompt 규약:
      - q.question_text/choices/answer 에는 첫번째 문제 (예: 41번)
      - q.sub_questions 에는 두번째부터 (예: 42번 1개 / 43-45는 44번,45번 2개)
    펼친 결과:
      - 첫 슬롯: 본문/group_label/sub_passages 모두 보존, 첫 질문/보기 그대로
      - 후속 슬롯: 본문 없이 질문/보기만 (앞 슬롯의 본문 박스를 공유한다고 간주)
    """
    expected = TYPE_SLOT_COUNT.get(cfg.type, 1)
    if expected <= 1:
        return [q]

    from shared.schemas.question import SubQuestion, CIRCLED

    # 후속 슬롯 갯수 = expected - 1
    subs = list(q.sub_questions or [])[:expected - 1]
    while len(subs) < expected - 1:
        subs.append(SubQuestion(
            question_text=f"[{cfg.type} {cfg.number + 1 + len(subs)}번 생성 실패 — 수동 입력 필요]",
            choices=list(CIRCLED), answer=1,
        ))

    # 첫 슬롯: 메인 질문 그대로, sub_questions 만 비움
    first = q.model_copy(update={
        "number":        cfg.number,
        "sub_questions": None,
    })
    out: list[Question] = [first]

    # 후속 슬롯들
    for i, sub in enumerate(subs):
        out.append(Question(
            number=cfg.number + 1 + i,
            type=cfg.type,
            points=cfg.points,
            passage=[],
            question_text=sub.question_text,
            choices=sub.choices,
            answer=sub.answer,
            group_label=None,
        ))
    return out


async def generate_all(
    q_configs: list[QuestionConfig],
    grade: str,
    reference_text: str = "",
    *,
    client: LLMClient | None = None,
) -> list[Question]:
    cli = client or get_client()
    tasks = [generate_one(cfg, grade, reference_text, client=cli) for cfg in q_configs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: list[Question] = []
    for i, r in enumerate(results):
        cfg = q_configs[i]
        if isinstance(r, BaseException):
            print(f"[Q{cfg.number}] 생성 예외: {r}")
            out.append(_fallback(cfg.type, cfg.number))
            continue
        # 장문 세트는 sub_questions를 펼쳐 N개 슬롯으로 확장
        if cfg.type in TYPE_SLOT_COUNT:
            out.extend(_expand_sub_questions(r, cfg))
        else:
            out.append(r)
    return out
