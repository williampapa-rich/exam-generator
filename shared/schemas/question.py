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
    "내용불일치(26)":   "letter_box",
    "안내문(27)":       "letter_box",
    "어법(28)":         "marker_inline",
    "어휘(29)":         "marker_inline",
    "빈칸-단어(30)":    "blank_inline",
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
    "장문어법(41-42)":  "long_set",
    "장문독해(43-45)":  "long_set",
}


# ─── 지원 유형 (18~45번, 듣기 1~17 제외, 도표 25 비활성) ────────────────────────
# 활성 유형: GUI 셀렉트박스에 노출되는 유형
# 비활성 유형: prompts에는 정의 있으나 LLM 호출 거부 (도표 25)
ACTIVE_TYPES: tuple[str, ...] = (
    "목적(18)", "심경(19)", "주장(20)", "밑줄함의(21)",
    "요지(22)", "주제(23)", "제목(24)",
    "내용불일치(26)", "안내문(27)",
    "어법(28)", "어휘(29)",
    "빈칸-단어(30)", "빈칸-구(31)", "빈칸-절(32)", "빈칸-절(33)", "빈칸-절(34)",
    "무관문장(35)",
    "순서배열(36)", "순서배열(37)",
    "문장삽입(38)", "문장삽입(39)",
    "요약문(40)",
    "장문어법(41-42)", "장문독해(43-45)",
)
DISABLED_TYPES: tuple[str, ...] = ("도표(25)",)

# 유형 → 차지하는 슬롯 수 (장문 세트는 자동 확장)
TYPE_SLOT_COUNT: dict[str, int] = {
    "장문어법(41-42)":  2,
    "장문독해(43-45)":  3,
}


class SubQuestion(BaseModel):
    """장문 세트(41-42, 43-45) 안의 개별 문항"""
    question_text: str
    choices:       list[str]
    answer:        int = Field(ge=1, le=5)


class Question(BaseModel):
    """LLM 출력 + renderer 입력 + parser 출력의 공통 모델."""

    # parser는 추출 실패 시 None일 수 있음. LLM/renderer 단에서는 항상 채워져 있어야 함.
    number: Optional[int] = Field(default=None, ge=1, le=45)
    type:   Optional[str] = None
    points: Optional[float] = None

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
