"""
정규식 기반 문제 단위 감지기.

작동 방식:
  1. section0.xml 의 모든 hp:p 단락을 순회
  2. 각 단락의 (텍스트 + 표 보유 여부 + 단락속성 ID) 를 시퀀스로 만듦
  3. 패턴: "번호+질문" 단락 발견 → 그 앞뒤 단락을 묶어 하나의 Question 로 구성
     - 앞 N개 중 표 보유 단락 = 지문 박스
     - 다음 1~5개 = 선택지

핵심 정규식 라이브러리는 REGEX_LIB에 정의되어 있어 외부에서 사용 가능.
"""
from __future__ import annotations

import re
from typing import Optional

from tools.parser.unpack import (
    HwpxFile, get_paragraph_text, get_paragraph_props, has_table,
    get_table_inner_paragraphs,
)
from shared.schemas.question import Question


# ─── 정규식 라이브러리 ───────────────────────────────────────────────────────
# 각 패턴은 외부에서 재사용 가능하도록 raw string으로 보관
REGEX_LIB = {
    # 문제 시작: "1." 또는 "1)" 또는 "[1]" 패턴
    "question_number":   r"^\s*(?P<num>\d{1,2})[.\)]\s*",

    # 질문 + 배점: "다음 글의 ... 가장 적절한 것은? [3점]"
    "question_with_pts":  r"(?P<question>.+?(?:적절한|올바른|일치하지|들어갈|순서로|뜻하는|관계 없는|언급되지 않은|언급된)\s*(?:것은|곳은|문장은|것을\s*고르시오|것만을\s*고르시오)[\?.])\s*\[(?P<pts>\d+\.?\d*)점\]",
    # 질문만 (배점 없음). 마지막 [3점] 누락 케이스
    "question_only":      r"(?P<question>.+?(?:적절한|올바른|일치하지|들어갈|순서로|뜻하는|관계\s*없는|언급되지\s*않은|언급된)\s*(?:것은|곳은|문장은|것을\s*고르시오|것만을\s*고르시오)[\?.])",

    # 배점 단독 검색
    "points":            r"\[(?P<pts>\d+\.?\d*)점\]",

    # 보기 기호 (원문자 ①②③④⑤)
    "choice_symbol":      r"[①②③④⑤]",
    "choice_marker":      r"(?P<sym>[①②③④⑤])\s*(?P<text>[^①②③④⑤\n]*)",

    # 그룹 레이블 (예: "[1~3] 다음 글을 읽고 물음에 답하시오.")
    "group_label":        r"^\s*\[(?P<range>\d+\s*[~∼-]\s*\d+)\]\s*(?P<instruction>.+)$",

    # 빈칸 표시 (다양한 형태)
    "blank":             r"_{3,}|\(\s*[A-D]\s*\)\s*_+|_+\s*\([A-D]\)",

    # 어법/어휘 인라인 마커
    "underline_marker":   r"[①②③④⑤](?:\s*\[[^/\]]+/[^\]]+\])?",
    "abc_choice_inline":   r"\((?P<letter>[A-D])\)\s*\[(?P<opt1>[^/\]]+)/(?P<opt2>[^/\]]+)\]",

    # (A)-(B)-(C) 조합 선택지
    "abc_combination":     r"\(A\)\s*[\w\-]+\s*[-–·]\s*\(B\)\s*[\w\-]+\s*[-–·]\s*\(C\)\s*[\w\-]+",

    # 영문 문장 (지문 판별용 휴리스틱)
    "english_sentence":    r"[A-Za-z][A-Za-z\s,.;:'\"\-()]+[.!?]",
}

# 컴파일된 패턴 (자주 사용)
_PAT_QNUM = re.compile(REGEX_LIB["question_number"])
_PAT_QWITH = re.compile(REGEX_LIB["question_with_pts"])
_PAT_QONLY = re.compile(REGEX_LIB["question_only"])
_PAT_PTS = re.compile(REGEX_LIB["points"])
_PAT_CSYM = re.compile(REGEX_LIB["choice_symbol"])
_PAT_CMARK = re.compile(REGEX_LIB["choice_marker"])
_PAT_GROUP = re.compile(REGEX_LIB["group_label"])
_PAT_BLANK = re.compile(REGEX_LIB["blank"])


# Question 모델은 shared/schemas/question.py에서 import (SSOT).
# parser는 detection 결과를 그 모델에 채워서 반환합니다.


def _fmt_question(q: Question) -> str:
    """디버깅용 한 줄 요약 (구버전 __repr__ 대체)"""
    return (
        f"Q{q.number} ({q.points or '?'}점) "
        f"[지문{len(q.passage)}줄, 보기{len(q.choices)}개] "
        f"{q.question_text[:30]}..."
    )


# ─── 단락 시퀀스 → 문제 리스트 ────────────────────────────────────────────────

def detect_questions(hwpx: HwpxFile) -> list[Question]:
    """전체 섹션에서 문제 목록 감지"""
    paragraphs = hwpx.all_section_paragraphs()
    if not paragraphs:
        return []

    # 각 단락의 메타 정보 미리 계산
    # ⚠️ 평가원 양식은 페이지 전체를 하나의 큰 표로 감싸는 경우가 있음.
    #    그 표 안에는 본문 hp:p 단락들이 들어 있는데, all_section_paragraphs()는
    #    이미 표 안의 단락도 평탄화해서 줍니다. 여기서는 "그 단락 자체가 표를
    #    포함하는가?"만 체크합니다 — 즉 본문 안에 별도로 박힌 작은 표만 박스로 봄.
    para_info = []
    for i, p in enumerate(paragraphs):
        text = get_paragraph_text(p).strip()
        is_table = has_table(p)
        # 표 텍스트가 너무 길면(>500자) 페이지 컨테이너일 가능성 → 박스 아님
        is_passage_box = is_table and len(text) < 500 and len(text) > 30

        para_info.append({
            "index":       i,
            "text":        text,
            "has_table":   is_table,
            "is_passage_box": is_passage_box,
            "props":       get_paragraph_props(p),
        })

    # 문제 시작 단락 찾기
    question_starts: list[int] = []
    for i, info in enumerate(para_info):
        if _looks_like_question_start(info["text"]):
            question_starts.append(i)

    if not question_starts:
        return []

    # 각 시작 인덱스 → 다음 시작 직전까지를 한 문제로 묶기
    questions: list[Question] = []
    for q_idx, start in enumerate(question_starts):
        end = question_starts[q_idx + 1] if q_idx + 1 < len(question_starts) else len(para_info)

        # 그룹 레이블 (시작 직전 5개 단락 안에서 검색)
        current_group: Optional[str] = None
        for j in range(max(0, start - 5), start):
            m = _PAT_GROUP.match(para_info[j]["text"])
            if m:
                current_group = para_info[j]["text"]
                break

        # 시작 직전의 지문 박스 찾기 (진짜 박스만 — 페이지 컨테이너 제외)
        passage_box: Optional[dict] = None
        for j in range(max(0, start - 5), start):
            if para_info[j]["is_passage_box"]:
                passage_box = para_info[j]
                break

        question = _build_question(
            start_info=para_info[start],
            following=[para_info[j] for j in range(start + 1, end)],
            passage_box=passage_box,
            group_label=current_group,
        )
        questions.append(question)

    return questions


def _looks_like_question_start(text: str) -> bool:
    """이 단락이 새 문제의 시작인가?

    평가원 양식은 다양한 형태로 문제가 시작됩니다:
      "1. 다음을 듣고..."          (번호 + 질문 한 줄)
      "36."                          (번호만, 다음 단락에 본문)
      "31. Literature can be..."     (번호 + 영문 지문 시작)
    그래서 "숫자.+점/마침표" 형태면 일단 문제 시작으로 간주합니다.
    오탐을 피하려면 1~45 범위로 제한.
    """
    if not text:
        return False
    m = _PAT_QNUM.match(text)
    if not m:
        return False
    num = int(m.group("num"))
    if not (1 <= num <= 45):
        return False
    return True


def _is_group_continuation(_group: Optional[str], _q_idx: int) -> bool:
    """단순 휴리스틱: 그룹 레이블은 첫 문제에만 표시"""
    return False


def _build_question(start_info: dict, following: list[dict],
                    passage_box: Optional[dict], group_label: Optional[str]) -> Question:
    """단락 정보들 → Question 객체 조립"""
    q = Question()
    q.raw_paragraphs.append(start_info["text"])
    q.paragraph_indices.append(start_info["index"])
    q.group_label = group_label

    # 1. 시작 단락에서 번호 추출
    text = start_info["text"]
    m_num = _PAT_QNUM.match(text)
    if m_num:
        q.number = int(m_num.group("num"))
        rest = text[m_num.end():].strip()
    else:
        rest = text

    # 2. 시작 단락 또는 다음 단락에서 질문 + 배점 추출
    candidates = [rest] + [f["text"] for f in following[:3] if f["text"]]
    found_question = False
    for cand in candidates:
        if not cand:
            continue
        m_qwp = _PAT_QWITH.search(cand)
        if m_qwp:
            q.question_text = m_qwp.group("question").strip()
            q.points = float(m_qwp.group("pts"))
            found_question = True
            break
        m_qo = _PAT_QONLY.search(cand)
        if m_qo:
            q.question_text = m_qo.group("question").strip()
            m_pts = _PAT_PTS.search(cand)
            if m_pts:
                q.points = float(m_pts.group("pts"))
            found_question = True
            break
    if not found_question:
        # best-effort: 시작 단락의 번호 뒤 텍스트를 그대로 사용
        q.question_text = rest[:120] if rest else ""
        m_pts = _PAT_PTS.search(rest)
        if m_pts:
            q.points = float(m_pts.group("pts"))

    # 3. 지문 박스가 있으면 그 안의 텍스트를 지문으로
    if passage_box:
        q.has_passage_box = True
        q.passage = [line.strip() for line in passage_box["text"].split("\n") if line.strip()]
        if not q.passage and passage_box["text"]:
            q.passage = [passage_box["text"]]

    # 4. 다음 단락들에서 선택지 + (박스 없으면) 지문 추출
    pre_choice_paragraphs: list[str] = []   # 첫 ① 등장 전까지의 텍스트 = 지문 후보
    found_first_choice = False

    for info in following:
        text = info["text"]
        if not text:
            continue

        # 보기 기호 포함 → 선택지 추출 (한 단락 안에 ①②③ 여러 개 가능)
        choice_matches = list(_PAT_CMARK.finditer(text))
        if choice_matches:
            found_first_choice = True
            for cm in choice_matches:
                choice_text = cm.group("text").strip()
                if choice_text and len(q.choices) < 5:
                    q.choices.append(choice_text)
            if len(q.choices) >= 5:
                break
        elif not found_first_choice:
            # 아직 선택지 시작 전 → 지문 후보 (질문 자체는 제외)
            if text != q.question_text and not _PAT_QONLY.search(text):
                pre_choice_paragraphs.append(text)

    # 5. 박스가 없었다면 pre_choice 단락들을 지문으로 사용
    if not q.passage and pre_choice_paragraphs:
        q.passage = pre_choice_paragraphs

    # 6. 인라인 마커 / 빈칸 / 단서 감지
    full_passage = " ".join(q.passage)
    q.has_inline_markers = bool(_PAT_CSYM.search(full_passage))
    q.has_blanks         = bool(_PAT_BLANK.search(full_passage))

    return q


# ─── 디버깅 헬퍼 ──────────────────────────────────────────────────────────────

def print_questions(questions: list[Question]) -> None:
    print(f"\n=== 감지된 문제 {len(questions)}개 ===")
    for q in questions:
        print(f"  {_fmt_question(q)}")
        if q.choices:
            for i, c in enumerate(q.choices[:5]):
                print(f"    {['①','②','③','④','⑤'][i]} {c[:40]}")
        print()
