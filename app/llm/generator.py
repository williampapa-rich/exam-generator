"""
Provider-agnostic 문제 생성기.

LLM provider(Gemini/OpenAI/Anthropic) 선택은 base.get_client()가 담당.
각 provider는 native structured output으로 Question 모델을 직접 반환합니다.

장문 세트 (41-42, 43-45) 처리:
  사용자 입력은 1 cfg지만 결과는 sub_questions 갯수만큼 Question이 펼쳐져서 반환.
  number는 cfg.number 부터 +1, +2... 자동 부여.
  passage/sub_passages 는 첫 번째 펼친 Question 에만 보존, 나머지는 빈 본문.

마커 검증 (validators.py):
  유형별 필수 마커가 본문에 들어왔는지 검증. 누락 시 최대 3회 재시도 후 fallback.

그룹 라벨 (group_label):
  prompt 단에서 LLM이 채우지 않고 항상 null. generate_all 끝에서
  같은 대분류가 연속될 때만 자동으로 [N~M] 라벨을 붙여준다.
"""
from __future__ import annotations

import asyncio

import os
import time

from app.llm.base import LLMClient, UsageInfo, get_client
from app.llm.pricing import compute_cost
from app.llm.prompts import prompt_for_type
from app.llm.usage_log import log_call
from app.llm.validators import validate_question
from shared.schemas.question import (
    GROUP_LABEL_TEMPLATES, Question, QuestionConfig, TYPE_SLOT_COUNT,
    type_category,
)

MAX_RETRIES = 3  # 마커 검증 실패 시 최대 재생성 횟수

# 시도별 temperature 스케줄 (Phase 1).
# 길이/마커 위반은 다양성 부족이 아니라 instruction following 부족이므로
# 재시도일수록 temperature 를 낮춰 규약 준수 쪽으로 모델을 끌고 간다.
# 1차 시도는 다양성을 위해 0.7 유지, 2~3차는 점진적으로 낮춤.
RETRY_TEMPS: tuple[float, ...] = (0.7, 0.5, 0.3)


def _fallback(q_type: str, q_num: int, reason: str = "생성 실패") -> Question:
    """문항 생성 실패 시 사용자에게 안내 placeholder 를 반환.

    raw 에러 메시지(ValidationError, traceback 등)는 절대 본문에 박지 않는다.
    실제 사유는 explanation 에만 보존하여 디버깅용으로 사용.
    """
    return Question(
        number=q_num,
        type=q_type,
        passage=["※ 본 문항은 자동 생성에 실패했습니다. 다시 생성해 주세요."],
        question_text=f"[{q_num}번] 문항 생성 실패 — 재생성이 필요합니다.",
        choices=["", "", "", "", ""],
        answer=1,
        explanation=f"[fallback] {reason}",  # 디버그용, 화면에는 미노출
    )


async def generate_one(
    cfg: QuestionConfig,
    grade: str,
    reference_text: str = "",
    *,
    client: LLMClient | None = None,
) -> Question:
    """단일 유형 생성. 마커 검증 실패 시 최대 MAX_RETRIES 회 재시도.

    재시도 시 temperature 를 약간 올려 LLM의 다양성을 강제 — 동일 응답 반복 방지.
    """
    system, user = prompt_for_type(cfg.type, grade, cfg.points, reference_text)
    cli = client or get_client()

    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    last_error: str = "알 수 없는 오류"
    for attempt in range(1, MAX_RETRIES + 1):
        # 재시도일 때는 user 메시지 끝에 직전 실패 사유를 명시해 LLM이 보정하도록 유도
        retry_user = user
        if attempt > 1:
            extra_hint = ""
            # 길이 편차 검증 실패는 모델이 무시하기 쉬운 패턴이라 강한 명령형 추가
            if "본문 길이" in last_error and ("초과" in last_error or "미달" in last_error):
                extra_hint = (
                    "\n\n**중요**: 본문 길이를 무조건 hard limit 범위 안에 맞출 것. "
                    "내용을 압축하거나 주제 범위를 좁혀서라도 길이를 먼저 맞추고, 그 안에서 자연스러운 글을 쓸 것. "
                    "직전 시도가 길었으면 단락 수를 줄이거나 부수 설명을 잘라낼 것. "
                    "직전 시도가 짧았으면 핵심 논리를 더 풀어쓸 것."
                )
            retry_user = (
                f"{user}\n\n[직전 시도 실패 사유: {last_error}]\n"
                f"위 사유를 반드시 해소하여 재생성해주세요.{extra_hint}"
            )
        # 시도별 temperature: 0.7 → 0.5 → 0.3 (instruction following 강화 방향).
        # 4번째 이상 시도가 발생해도 안전하게 마지막 값 유지.
        temp = RETRY_TEMPS[min(attempt - 1, len(RETRY_TEMPS) - 1)]

        t0 = time.monotonic()
        usage: UsageInfo | None = None
        success = False
        q: Question | None = None
        try:
            q, usage = await cli.generate_json(
                system=system,
                user=retry_user,
                schema=Question,
                temperature=temp,
            )
        except Exception as e:
            last_error = f"LLM 호출 실패: {type(e).__name__}: {e}"
            print(f"[Q{cfg.number}] 시도 {attempt}/{MAX_RETRIES}: {last_error}")
            _log_usage(
                cfg.type, provider, usage, attempt, success, t0,
                fail_reason=last_error, question=None, temperature=temp,
            )
            continue

        # provider가 number/type/points를 못 채울 수 있으니 명시적으로 강제
        q.number = cfg.number
        q.type   = cfg.type
        q.points = cfg.points
        # group_label 은 후처리에서만 부착 — LLM이 채워도 무시
        q.group_label = None

        # 마커 검증
        err = validate_question(q)
        if err is None:
            success = True
            _log_usage(
                cfg.type, provider, usage, attempt, success, t0,
                question=q, temperature=temp,
            )
            return q
        last_error = err
        print(f"[Q{cfg.number}] 시도 {attempt}/{MAX_RETRIES} 마커 검증 실패: {err}")
        _log_usage(
            cfg.type, provider, usage, attempt, success, t0,
            fail_reason=err, question=q, temperature=temp,
        )

    print(f"[Q{cfg.number}] {MAX_RETRIES}회 시도 모두 실패 → fallback. 최종 사유: {last_error}")
    return _fallback(cfg.type, cfg.number, last_error)


def _build_question_extra(q: Question | None) -> dict:
    """Phase 1 로깅용: Question 에서 plan/naturalness/길이 정보 추출.

    사후 분석 목적:
      - scope 가 좁을수록 길이가 잘 맞나? (plan_topic_scope ↔ actual_words 상관)
      - naturalness_check 'OK' 와 실제 자연스러움이 일치하나?
      - 모델 자기보고 target_word_count vs 실제 단어수 분포 (LLM 카운트 정확도 측정)
    """
    if q is None:
        return {}
    extra: dict = {}
    if q.plan is not None:
        extra["plan_topic_scope"]       = q.plan.topic_scope
        extra["plan_thesis"]            = q.plan.thesis_sentence
        extra["plan_structure"]         = q.plan.structure_plan
        extra["plan_target_word_count"] = q.plan.target_word_count
    if q.naturalness_check is not None:
        extra["naturalness_check"] = q.naturalness_check
    # 길이 측정 — 글자/단어 모두 (검증 단위와 비교 가능하도록 _measured_length 사용)
    from app.llm.validators import _measured_length
    actual_chars = _measured_length(q)
    passage_text = " ".join(q.passage or [])
    if q.sub_passages:
        passage_text += " " + " ".join(" ".join(sp) for sp in q.sub_passages)
    actual_words = len(passage_text.split())
    extra["actual_chars"] = actual_chars
    extra["actual_words"] = actual_words
    if q.plan is not None:
        extra["word_count_diff"] = actual_words - q.plan.target_word_count
    return extra


def _log_usage(
    q_type:      str,
    provider:    str,
    usage:       UsageInfo | None,
    attempt:     int,
    success:     bool,
    t0:          float,
    fail_reason: str | None = None,
    question:    Question | None = None,
    temperature: float | None = None,
) -> None:
    """LLM 호출 1회 분 사용량을 jsonl 로 누적. usage 미수집(예외 등) 시 0 으로 기록.

    fail_reason: 검증 실패 또는 호출 예외 사유 — 디버깅용으로 jsonl 의 extra 필드에 저장.
    question:    LLM 응답이 Pydantic 파싱까지 성공한 경우 plan/naturalness/길이를 추가 로깅.
    temperature: 시도에 사용된 temperature (Phase 1 RETRY_TEMPS 효과 분석용).
    """
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    extra: dict = {}
    if fail_reason:
        extra["fail_reason"] = fail_reason
    if temperature is not None:
        extra["temperature"] = temperature
    extra.update(_build_question_extra(question))
    if usage is None:
        log_call(
            type=q_type, provider=provider, model="?",
            in_tokens=0, out_tokens=0, cost_krw=0.0,
            attempt=attempt, success=success, elapsed_ms=elapsed_ms,
            extra=extra or None,
        )
        return
    cost = compute_cost(usage.model, usage.prompt_tokens, usage.completion_tokens)
    log_call(
        type=q_type, provider=provider, model=usage.model,
        in_tokens=usage.prompt_tokens, out_tokens=usage.completion_tokens,
        cost_krw=cost, attempt=attempt, success=success, elapsed_ms=elapsed_ms,
        extra=extra or None,
    )


def _expand_sub_questions(q: Question, cfg: QuestionConfig) -> list[Question]:
    """장문 세트(41-42, 43-45): 메인 질문(q.question_text) + sub_questions 를 N개 Question으로 펼침.

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

    subs = list(q.sub_questions or [])[:expected - 1]
    while len(subs) < expected - 1:
        subs.append(SubQuestion(
            question_text=f"[{cfg.type} {cfg.number + 1 + len(subs)}번 생성 실패 — 수동 입력 필요]",
            choices=list(CIRCLED), answer=1,
        ))

    first = q.model_copy(update={
        "number":        cfg.number,
        "sub_questions": None,
    })
    out: list[Question] = [first]

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


def _attach_group_labels(questions: list[Question]) -> None:
    """동일 대분류가 연속될 때 [N~M] 그룹 라벨을 자동 부착.

    규칙:
      - 장문(41-42), 장문독해(43-45) 등 TYPE_SLOT_COUNT > 1 인 세트는 항상 라벨.
      - 그 외(순서배열/문장삽입 등)는 같은 대분류가 2개 이상 연속될 때만 라벨.
      - 라벨은 묶음의 **첫 슬롯**에만 부착. 나머지는 None.
      - 번호 범위는 묶음 첫 question.number ~ 마지막 question.number.
    """
    n = len(questions)
    i = 0
    while i < n:
        cat = type_category(questions[i].type)
        if not cat:
            i += 1
            continue

        # 같은 대분류 연속 묶음 찾기
        j = i
        while j + 1 < n and type_category(questions[j + 1].type) == cat:
            j += 1

        group_size = j - i + 1
        is_long_set = bool(questions[i].type in TYPE_SLOT_COUNT)
        # 라벨이 정의돼 있고 (a) 장문 세트이거나 (b) 같은 대분류 2개 이상이면 부착
        tmpl = GROUP_LABEL_TEMPLATES.get(cat)
        if tmpl and (is_long_set or group_size >= 2):
            first_num = questions[i].number or (i + 1)
            last_num  = questions[j].number or (j + 1)
            label_range = f"{first_num}~{last_num}" if first_num != last_num else f"{first_num}"
            questions[i].group_label = tmpl.format(range=label_range)
            # 묶음 내 나머지는 라벨 비움
            for k in range(i + 1, j + 1):
                questions[k].group_label = None
        else:
            # 단독 출제: 라벨 없음
            questions[i].group_label = None

        i = j + 1


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
            out.append(_fallback(cfg.type, cfg.number, str(r)))
            continue
        if cfg.type in TYPE_SLOT_COUNT:
            out.extend(_expand_sub_questions(r, cfg))
        else:
            out.append(r)

    # 그룹 라벨 자동 부착 (동일 대분류 연속 묶음)
    _attach_group_labels(out)
    return out
