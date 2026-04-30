"""
SSOT: parser/LLM/renderer/FastAPI 모두 이 모델을 공유.

owner: tools/parser/ (스키마 변경은 parser 작업의 일부)
readers: app/llm, app/renderer, app/main, tools/parser
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

CIRCLED: tuple[str, ...] = ("①", "②", "③", "④", "⑤")

# ─── 인라인 마커 (renderer가 별도 hp:run으로 분리) ────────────────────────────
# LLM 출력 / parser 추출에서 본문 안에 들어오는 강조 마커.
# UNDERLINE_MARK 사이의 텍스트는 underline=BOTTOM charPr로,
# BLANK_PLACEHOLDER 는 빈칸용 underline=BOTTOM run으로 렌더된다.
UNDERLINE_MARK     = "_"        # ex) "is _hunting the shadow_ here" → 가운데가 밑줄
BLANK_PLACEHOLDER  = "______"   # 언더스코어 6개 — 빈칸 (밑줄로만 표시)

# ─── 유형별 본문 레이아웃 패턴 ──────────────────────────────────────────────
# 평가원 png 분석 결과 다음 6가지로 분류된다:
#   reasoning            : 질문(좌정렬) → 지문(첫줄 들여쓰기) → 보기(좌정렬)
#                          예: 요지(22), 주제(23), 제목(24), 주장(20), 심경(19), 밑줄함의(21)
#   letter_box           : 질문 → 편지/안내문 본문(들여쓰기 없음, 줄바꿈 보존) → 보기
#                          예: 목적(18), 안내문(27), 내용불일치(26)
#   blank_inline         : 질문 → 지문(______ 빈칸 inline) → 보기
#                          예: 빈칸-단어(30), 빈칸-구(31), 빈칸-절(32~34)
#   marker_inline        : 질문 → 지문(① ② ③ ④ ⑤ inline 또는 (a)~(e)) → 선택지=['①',...]
#                          예: 어법(28), 어휘(29), 무관문장(35), 문장삽입(38,39)
#   passage_segments     : 질문 → 주어진 글 + (A)/(B)/(C) 분할 단락 → 보기
#                          예: 순서배열(36,37), 요약문(40)
#   long_set             : 질문 1 + 본문 (A)~(D) 분할 + 부속 문항 (sub_questions)
#                          예: 장문어법(41-42), 장문독해(43-45)

LAYOUT_PATTERN: dict[str, str] = {
    "목적(18)":         "letter_box",
    "심경(19)":         "reasoning",
    "주장(20)":         "reasoning",
    "밑줄함의(21)":     "reasoning",
    "요지(22)":         "reasoning",
    "주제(23)":         "reasoning",
    "제목(24)":         "reasoning",
    "인물일치(26)":     "reasoning",      # 인물 약력 — 박스 없음, 추론형 본문
    "안내문(27)":       "letter_box",     # Adventure City Pass 류 박스
    "안내문(28)":       "letter_box",     # Luckwood Snow Festival 류 박스
    "어법(29)":         "marker_inline",  # 본문 ①②③④⑤ + 밑줄
    "어휘(30)":         "marker_inline",  # 본문 단어 ①②③④⑤ + 밑줄
    "빈칸-구(31)":      "blank_inline",
    "빈칸-절(32)":      "blank_inline",
    "빈칸-절(33)":      "blank_inline",
    "빈칸-절(34)":      "blank_inline",
    "무관문장(35)":     "marker_inline",
    "순서배열(36)":     "passage_segments",
    "순서배열(37)":     "passage_segments",
    "문장삽입(38)":     "marker_inline",
    "문장삽입(39)":     "marker_inline",
    "요약문(40)":       "passage_segments",
    "장문(41-42)":      "long_set",       # 41=제목 + 42=어휘
    "장문독해(43-45)":  "long_set",       # 43=순서 + 44=지칭 + 45=일치
}


# ─── 지원 유형 (18~45번, 듣기 1~17 제외, 도표 25 비활성) ────────────────────────
# 활성 유형: GUI 셀렉트박스에 노출되는 유형
# 비활성 유형: prompts에는 정의 있으나 LLM 호출 거부 (도표 25)
ACTIVE_TYPES: tuple[str, ...] = (
    "목적(18)", "심경(19)", "주장(20)", "밑줄함의(21)",
    "요지(22)", "주제(23)", "제목(24)",
    "인물일치(26)", "안내문(27)", "안내문(28)",
    "어법(29)", "어휘(30)",
    "빈칸-구(31)", "빈칸-절(32)", "빈칸-절(33)", "빈칸-절(34)",
    "무관문장(35)",
    "순서배열(36)", "순서배열(37)",
    "문장삽입(38)", "문장삽입(39)",
    "요약문(40)",
    "장문(41-42)", "장문독해(43-45)",
)
DISABLED_TYPES: tuple[str, ...] = ("도표(25)",)

# 유형 → 차지하는 슬롯 수 (장문 세트는 자동 확장)
TYPE_SLOT_COUNT: dict[str, int] = {
    "장문(41-42)":     2,
    "장문독해(43-45)": 3,
}


def type_category(q_type: str | None) -> str:
    """유형명에서 괄호 앞 대분류만 추출 ('순서배열(36)' → '순서배열').

    그룹 라벨 자동 부착 시 같은 대분류가 연속된 경우만 묶음 라벨을 표시한다.
    """
    if not q_type:
        return ""
    idx = q_type.find("(")
    return q_type[:idx] if idx > 0 else q_type


# 박스 단락이 필요한 유형 (letter_box / 주어진 글 / 요약문 / 장문 본문)
TYPES_WITH_BOX_PASSAGE: frozenset[str] = frozenset({
    "목적(18)", "안내문(27)", "안내문(28)",
})
TYPES_WITH_GIVEN_BOX: frozenset[str] = frozenset({
    "순서배열(36)", "순서배열(37)",   # 주어진 글 박스
    "문장삽입(38)", "문장삽입(39)",   # 주어진 문장 박스
})
TYPES_WITH_SUMMARY_BOX: frozenset[str] = frozenset({
    "요약문(40)",
})
TYPES_WITH_LONGSET_BOX: frozenset[str] = frozenset({
    "장문(41-42)", "장문독해(43-45)",
})


# 동일 대분류가 연속될 때 자동 부착되는 그룹 라벨 템플릿.
# {range} 자리에 '36~37' 같은 번호 범위가 들어간다.
# 장문 세트는 자기 자신만으로 1세트라 항상 라벨 부착.
GROUP_LABEL_TEMPLATES: dict[str, str] = {
    "순서배열": "[{range}] 주어진 글 다음에 이어질 글의 순서로 가장 적절한 것을 고르시오.",
    "문장삽입": "[{range}] 글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳을 고르시오.",
    "장문":     "[{range}] 다음 글을 읽고, 물음에 답하시오.",
    "장문독해": "[{range}] 다음 글을 읽고, 물음에 답하시오.",
}


class SubQuestion(BaseModel):
    """장문 세트(41-42, 43-45) 안의 개별 문항"""
    question_text: str
    choices:       list[str]
    answer:        int = Field(ge=1, le=5)


class QuestionPlan(BaseModel):
    """passage 작성 전 LLM이 강제로 채우는 계획 필드 (Phase 1).

    structured output 토큰 생성 순서상 Question.plan 이 passage 보다 먼저 출력되므로,
    passage 작성 시점에 모델 자기 입으로 출력한 scope/thesis 가 context 에 남아 있다.
    이게 길이 통제의 핵심 — 길이를 직접 강제하지 않고 scope 를 좁혀 길이가 부산물이 되게 한다.
    """
    topic_scope: str = Field(
        description=(
            "이 글이 다룰 주제 한 문장. 다음 조건을 모두 충족해야 함: "
            "(1) 단 하나의 주장/현상/관찰만 다룬다. "
            "(2) 예시는 0개 또는 1개. "
            "(3) 도입-전개-결론을 짧게 끝낼 수 있어야 한다. "
            "(4) 추상적 큰 개념('the importance of education', 'environmental protection') 금지. "
            "한 문장으로 좁고 구체적으로 진술. 영어로 작성."
        )
    )
    thesis_sentence: str = Field(
        description=(
            "이 글이 도달할 결론 한 문장. 영어로. "
            "passage 본문은 이 문장을 향해 수렴해야 함. "
            "이 문장 자체를 본문에 그대로 쓸 필요는 없음."
        )
    )
    structure_plan: str = Field(
        description=(
            "도입-전개-결론 구조 계획. 'Hook: ... / Develop: ... / Close: ...' 형식. "
            "각 부분 1~2개 영어 단어로만. 이 계획에서 벗어난 곁가지 금지."
        )
    )
    target_word_count: int = Field(
        ge=50, le=400,
        description=(
            "목표 영어 단어 수. 유형별 가이드라인을 참고하여 정수로 작성. "
            "이 값은 자기 인식용이며, 실제 작성 시 ±15% 편차는 허용됨. "
            "정확한 카운트보다 scope 에 맞는 자연스러운 길이가 우선."
        ),
    )


# naturalness_check 가 가질 수 있는 값 (validators 가 import 해서 reject 판정)
NATURALNESS_OK = "OK"
NATURALNESS_REWRITE_VALUES: tuple[str, ...] = (
    "REWRITE_SCOPE_TOO_BROAD",
    "REWRITE_ABRUPT_ENDING",
    "REWRITE_REPETITIVE",
    "REWRITE_FORCED_BREVITY",
)


class Question(BaseModel):
    """LLM 출력 + renderer 입력 + parser 출력의 공통 모델."""

    # parser는 추출 실패 시 None일 수 있음. LLM/renderer 단에서는 항상 채워져 있어야 함.
    number: Optional[int] = Field(default=None, ge=1, le=45)
    type:   Optional[str] = None
    points: Optional[float] = None

    # 1단계: 계획 (LLM 한정 — passage 보다 먼저 출력되도록 위에 배치).
    # parser/renderer 는 이 필드를 무시. 따라서 default None.
    plan: Optional[QuestionPlan] = None

    passage:       list[str] = Field(default_factory=list)
    question_text: str = ""
    choices:       list[str] = Field(default_factory=lambda: list(CIRCLED))
    answer:        int = Field(default=1, ge=1, le=5)
    explanation:   str = ""

    # 유형별 선택 필드
    given_sentence: Optional[str]                = None  # 38, 39
    sub_passages:   Optional[list[list[str]]]    = None  # 36, 37
    summary:        Optional[str]                = None  # 40
    sub_questions:  Optional[list[SubQuestion]]  = None  # 41-42, 43-45

    # 3단계: 자가검증 (LLM 한정 — passage/문항이 모두 채워진 뒤 자기 글을 평가).
    # parser/renderer 는 이 필드를 무시. 따라서 default None.
    # validators 가 None 또는 'OK' 외의 값을 보면 reject 하여 재시도.
    naturalness_check: Optional[Literal[
        "OK",
        "REWRITE_SCOPE_TOO_BROAD",
        "REWRITE_ABRUPT_ENDING",
        "REWRITE_REPETITIVE",
        "REWRITE_FORCED_BREVITY",
    ]] = Field(
        default=None,
        description=(
            "passage 를 다시 읽고 평가: "
            "'OK' = 자연스러움. "
            "'REWRITE_SCOPE_TOO_BROAD' = 주제가 너무 넓어 글이 늘어짐. "
            "'REWRITE_ABRUPT_ENDING' = 결론이 갑자기 튀어나옴 / 비약. "
            "'REWRITE_REPETITIVE' = 같은 주장 반복으로 분량 채움. "
            "'REWRITE_FORCED_BREVITY' = 분량 줄이느라 논리 끊김."
        ),
    )

    # 장문독해(43-45) 전용 — 본문 (a)~(e) 5개 라벨이 실제로 가리키는 인물명을 인덱스 순서대로
    # 명시. 이게 4:1 분포의 단일 진실 원천 — validator 가 Counter 로 4:1 / answer 일치 자동 검증.
    # plan 단계의 사전 분포 commit (referent_distribution) + 자가검증 (referent_check) 은
    # 모두 제거 — 모델 인지 부담만 폭증시키고 honesty 거짓말로 우회됨. 단일 필드면 거짓말 불가.
    referent_assignments: Optional[list[str]] = Field(
        default=None,
        description=(
            "장문독해(43-45) 전용. (a)~(e) 5개 라벨이 가리키는 인물을 5개 문자열로 적기 "
            "(인덱스 0=a, 1=b, 2=c, 3=d, 4=e). "
            "예: ['Leo', 'Mr. Harrison', 'Leo', 'Leo', 'Leo']. "
            "**필수**: 정확히 4:1 분포 (한 인물 4번, 다른 인물 1번). "
            "**필수**: '1번 등장 인물' 의 인덱스 + 1 이 sub_questions[0].answer 와 일치. "
            "주의: 'X advised (b) him' 에서 him 은 청자(주인공) 이지 X 가 아님."
        ),
    )

    group_label:    Optional[str]                = None

    # parser 메타 (LLM/renderer는 무시 가능)
    has_passage_box:    bool = False
    has_inline_markers: bool = False
    has_blanks:         bool = False
    raw_paragraphs:     list[str] = Field(default_factory=list)
    paragraph_indices:  list[int] = Field(default_factory=list)

    @property
    def passage_text(self) -> str:
        return " ".join(self.passage)

    @property
    def answer_symbol(self) -> str:
        return CIRCLED[self.answer - 1]

    @property
    def layout_pattern(self) -> str:
        """렌더러가 단락 시퀀스를 결정하는 데 쓰는 유형 분류.

        type 필드가 LAYOUT_PATTERN 매핑에 없으면 'reasoning' 으로 폴백 (가장 일반적).
        """
        return LAYOUT_PATTERN.get(self.type or "", "reasoning")


class QuestionConfig(BaseModel):
    """클라이언트가 /api/generate에 보내는 각 문항 설정"""
    number: int = Field(ge=1, le=45)
    type:   str
    points: float = 3.0


class ExamMeta(BaseModel):
    title:           str = "영어 모의고사"          # → {{EXAM_TITLE}}
    subject_name:    str = "영어"                    # → {{SUBJECT_NAME}}
    top_left_label:  str = ""                        # → {{TOP_LEFT_LABEL}} (머릿글 좌상단)
    top_right_label: str = ""                        # → {{TOP_RIGHT_LABEL}} (머릿글 우상단)
    school:          str = ""
    grade:           str = "고3 (수능)"
    date:            str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    time_limit:      str = "45분"
    layout:          Literal["1단", "2단", "2단 (수능형)"] = "2단 (수능형)"


MAX_QUESTIONS = 50


class GenerateRequest(BaseModel):
    meta:           ExamMeta
    questions:      list[QuestionConfig] = Field(..., min_length=1, max_length=MAX_QUESTIONS)
    reference_text: str = ""
    export_format:  Literal["hwpx"] = "hwpx"
    provider:       Optional[str] = None  # "gemini"/"openai"/"anthropic" — None이면 env LLM_PROVIDER
    model:          Optional[str] = None  # provider별 모델 ID — None이면 env / DEFAULT_MODEL
