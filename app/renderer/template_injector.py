"""
Template injector — templates/template.hwpx의 Q1~Q10 슬롯 영역을
유형별 paraPr/charPr 단락 시퀀스로 동적 생성해 채운다.

세 가지 핵심 책임:
  1) header / 메타 placeholder ({{SUBJECT_NAME}} 등) 치환
  2) Q1~Q10 슬롯 영역을 N개 문항의 단락 시퀀스로 대체
     - 유형별 layout_pattern 에 따라 단락 종류와 paraPr/charPr 결정
     - 평가원 패턴: 질문(paraPr=30, h=1350 charPr=52) → 지문(paraPr=0, charPr=13)
       → 보기(paraPr=18, charPr=18) → 빈 단락(paraPr=18, charPr=15)
  3) 인라인 마커를 별도 hp:run으로 분리
     - _text_ → underline charPr (53)
     - ______ → underline run (53, 빈 텍스트가 아닌 nbSpace 6개로 표현)
     - 그 외 ①②③④⑤, (a)~(e), (A)~(D) 등은 LLM 출력 그대로 텍스트로 둠
       (이미 숫자/괄호 표기여서 굳이 스타일 분리 불필요)

template.hwpx 의 Contents/header.xml 은 평가원 샘플 그대로이므로
paraPr 0/18/19/30/77/78 과 charPr 13/18/52/53/15 가 모두 정의되어 있음 (수정 불필요).
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from shared.schemas.question import (
    BLANK_PLACEHOLDER, ExamMeta, Question, UNDERLINE_MARK,
    TYPES_WITH_BOX_PASSAGE, TYPES_WITH_GIVEN_BOX,
    TYPES_WITH_SUMMARY_BOX, TYPES_WITH_LONGSET_BOX,
)

_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "template.hwpx"

# section0.xml 단락 구조 (parser 분석 결과 고정값)
_BLOCK_PARA_COUNT  = 10                          # Q1 블록 = 10단락
_FIRST_Q_PARA_IDX  = 8                           # 헤더 8단락 후 Q1 시작
_TOTAL_Q_PARAS     = 10 * _BLOCK_PARA_COUNT      # Q1~Q10 = 100단락

_PARA_RE        = re.compile(r"<hp:p\b[^>]*>.*?</hp:p>", re.DOTALL)
_Q_SLOT_MARKER  = "<!--QUESTIONS_HERE-->"


# ─── header.xml 의 paraPr / charPr 매핑 (평가원 양식 그대로) ──────────────────
# 단락 스타일
PARA_QUESTION       = 30   # 추론형 질문 ("22. 다음 글의 요지로 ...")
PARA_QUESTION_SHORT = 77   # 짧은 질문 (목적/심경/주장)
PARA_PASSAGE_INDENT = 0    # 추론형 본문 (첫줄 들여쓰기 +1050)
PARA_PASSAGE_PLAIN  = 27   # 편지/안내문 본문 (들여쓰기 0)
PARA_LETTER_FIRST   = 78   # 편지 헤더 라인 ("Dear ...,")
PARA_CHOICE         = 18   # 보기 ① ~ ⑤ (hanging indent)
PARA_CHOICE_SHORT   = 78   # 짧은 답 한 줄 형식 (목적류 보기)
PARA_BLANK_LINE     = 38   # 빈 단락 (지문과 보기 사이 간격)
PARA_GROUP_LABEL    = 31   # [36~37] 같은 그룹 라벨
PARA_NOTE_RIGHT     = 19   # * widget: 제품 - 우측 정렬 각주

# 문자 스타일 (모두 underline=NONE 기본)
CHAR_NUMBER         = 52   # 문항 번호 "22." (h=1350, 큰 사이즈)
CHAR_QUESTION_GAP   = 5    # 번호와 질문 사이 공백 charPr (h=1148, 좁은)
CHAR_QUESTION       = 51   # 질문 본문 한국어 (h=1148)
CHAR_BODY           = 13   # 일반 본문 (영어/한글 공용, h=1150)
CHAR_BODY_KOR       = 18   # 보기 한국어 본문 (h=1150)
CHAR_UNDERLINE      = 53   # 밑줄 강조 (underline=BOTTOM, h=1148)
CHAR_BLANK          = 15   # 빈 단락의 끝 charPr (사이드 이펙트 없는 작은 사이즈)
CHAR_NOTE           = 14   # 각주용 (* widget:)


class _Cache:
    """모듈 로드 시점에 template.hwpx 분석한 결과를 보관."""
    raw_files: dict[str, bytes]    # template ZIP 내용 (mimetype 제외)
    section0_shell: str            # Q슬롯 영역을 marker로 대체한 section0.xml

    def __init__(self) -> None:
        with zipfile.ZipFile(_TEMPLATE_PATH) as zf:
            self.raw_files = {
                name: zf.read(name) for name in zf.namelist() if name != "mimetype"
            }
        section0 = self.raw_files["Contents/section0.xml"].decode("utf-8")
        self.section0_shell = self._build_shell(section0)

    @staticmethod
    def _build_shell(section0: str) -> str:
        """Q1~Q10 슬롯 영역(100단락)을 단일 marker로 치환한 shell 반환."""
        paras = list(_PARA_RE.finditer(section0))
        last_q_idx = _FIRST_Q_PARA_IDX + _TOTAL_Q_PARAS - 1
        if last_q_idx >= len(paras):
            raise RuntimeError(
                f"template.hwpx 가 Q10 슬롯까지 갖추지 못함: paras={len(paras)}"
            )
        slots_start = paras[_FIRST_Q_PARA_IDX].start()
        slots_end   = paras[last_q_idx].end()
        return section0[:slots_start] + _Q_SLOT_MARKER + section0[slots_end:]


_cache_state: tuple[float, _Cache] | None = None


def _cache() -> _Cache:
    """template.hwpx 의 mtime 이 바뀌면 자동으로 다시 로드.

    파일이 코드 외부에서 수정될 수 있으므로 (디자인 작업 등) lru_cache 만으로는
    부족하고 mtime 기반으로 무효화한다.
    """
    global _cache_state
    mtime = _TEMPLATE_PATH.stat().st_mtime
    if _cache_state is None or _cache_state[0] != mtime:
        _cache_state = (mtime, _Cache())
    return _cache_state[1]


# ─── XML escape (텍스트 노드용) ───────────────────────────────────────────────

_TRANS = str.maketrans({
    "&":  "&amp;",
    "<":  "&lt;",
    ">":  "&gt;",
    '"':  "&quot;",
    "\r": "",
    "\n": " ",
})


def _xe(s: object) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.translate(_TRANS)


# ─── 인라인 마커 분리 (밑줄/빈칸을 별도 hp:run으로) ─────────────────────────────
# 토큰 형식: ("plain"|"underline"|"blank", text)
_BLANK_RE     = re.compile(r"_{4,}")                    # _____ 4개 이상 → 빈칸
_UNDERLINE_RE = re.compile(r"_([^_\n][^_\n]*?)_")       # _foo bar_ → 밑줄


def _tokenize_inline(text: str) -> list[tuple[str, str]]:
    """본문 한 줄을 (kind, text) 토큰으로 분해.

    우선순위: blank(_____) > underline(_foo_) > plain.
    LLM 출력 안에 스코어 1개나 단어 중간 _는 underline으로 잡지 않도록 가드.
    """
    if not text:
        return []

    # 1) 먼저 빈칸을 sentinel 로 치환 (_____ 4개 이상)
    SENT = "\x00BLANK\x00"
    blanks: list[str] = []
    def _blank_repl(m: re.Match) -> str:
        blanks.append(m.group(0))
        return SENT
    masked = _BLANK_RE.sub(_blank_repl, text)

    # 2) 밑줄 마커 분해 (남은 _foo_ 패턴)
    tokens: list[tuple[str, str]] = []
    last = 0
    for m in _UNDERLINE_RE.finditer(masked):
        if m.start() > last:
            tokens.append(("plain", masked[last:m.start()]))
        tokens.append(("underline", m.group(1)))
        last = m.end()
    if last < len(masked):
        tokens.append(("plain", masked[last:]))

    # 3) sentinel 을 다시 blank 토큰으로 풀기
    out: list[tuple[str, str]] = []
    bi = 0
    for kind, t in tokens:
        if SENT not in t:
            if t:
                out.append((kind, t))
            continue
        # plain 안의 sentinel 분리
        parts = t.split(SENT)
        for i, p in enumerate(parts):
            if p:
                out.append((kind, p))
            if i < len(parts) - 1:
                # blank 자체는 길이를 보존 (LLM이 ____4 ~ ______6 다양함)
                out.append(("blank", blanks[bi]))
                bi += 1
    return out


def _runs_for_text(text: str, default_charpr: int) -> str:
    """본문 한 줄을 hp:run 시퀀스 XML 문자열로 변환.

    인라인 마커가 없으면 단일 run.
    있으면 plain 은 default_charpr, underline/blank 는 CHAR_UNDERLINE.
    """
    tokens = _tokenize_inline(text)
    if not tokens:
        return f'<hp:run charPrIDRef="{default_charpr}"><hp:t/></hp:run>'

    parts: list[str] = []
    for kind, t in tokens:
        if kind == "plain":
            parts.append(
                f'<hp:run charPrIDRef="{default_charpr}"><hp:t>{_xe(t)}</hp:t></hp:run>'
            )
        else:
            # underline 과 blank 모두 underline charPr 로 표시
            # blank 의 경우 텍스트 자체(언더스코어들)를 그대로 두면 hwp 에서 밑줄 친 빈 영역으로 보임
            parts.append(
                f'<hp:run charPrIDRef="{CHAR_UNDERLINE}"><hp:t>{_xe(t)}</hp:t></hp:run>'
            )
    return "".join(parts)


# ─── 단락 빌더 ─────────────────────────────────────────────────────────────────

def _para(parapr: int, runs_xml: str) -> str:
    """단일 hp:p 단락 생성. runs_xml 은 이미 만들어진 hp:run 시퀀스."""
    return (
        f'<hp:p id="0" paraPrIDRef="{parapr}" styleIDRef="0" '
        f'pageBreak="0" columnBreak="0" merged="0">{runs_xml}</hp:p>'
    )


def _para_text(parapr: int, charpr: int, text: str) -> str:
    """plain text 단일 run 단락 (인라인 마커 분리 없음 — 헤더/번호 등에 사용)."""
    return _para(
        parapr,
        f'<hp:run charPrIDRef="{charpr}"><hp:t>{_xe(text)}</hp:t></hp:run>',
    )


def _para_blank() -> str:
    """문항 사이 빈 단락 (간격용)."""
    return _para(
        PARA_BLANK_LINE,
        f'<hp:run charPrIDRef="{CHAR_BLANK}"><hp:t/></hp:run>',
    )


def _para_question_line(number: int, question_text: str, points: float | None) -> str:
    """'22. 다음 글의 요지로 ...   [3점]' 형식 단락.

    번호는 큰 charPr(52), 질문은 일반 charPr(51), 배점은 작은 charPr(5).
    평가원 양식: '22.<nbSpace><fwSpace>다음 글의 요지로 ...' 처럼 번호 뒤에
    nbSpace + fwSpace 조합으로 간격을 둔다.
    """
    runs: list[str] = []
    runs.append(
        f'<hp:run charPrIDRef="{CHAR_NUMBER}">'
        f'<hp:t>{number}.</hp:t></hp:run>'
    )
    runs.append(
        f'<hp:run charPrIDRef="{CHAR_QUESTION_GAP}">'
        f'<hp:t><hp:nbSpace/><hp:fwSpace/></hp:t></hp:run>'
    )
    runs.append(
        f'<hp:run charPrIDRef="{CHAR_QUESTION}">'
        f'<hp:t>{_xe(question_text)}</hp:t></hp:run>'
    )
    if points is not None:
        pts = int(points) if float(points).is_integer() else points
        runs.append(
            f'<hp:run charPrIDRef="{CHAR_QUESTION_GAP}">'
            f'<hp:t> [{pts}점]</hp:t></hp:run>'
        )
    return _para(PARA_QUESTION, "".join(runs))


def _para_passage(line: str, *, indent: bool, kor: bool = False) -> str:
    """본문 한 단락. indent=True 면 첫줄 들여쓰기 paraPr(0), False면 paraPr(27)."""
    parapr = PARA_PASSAGE_INDENT if indent else PARA_PASSAGE_PLAIN
    charpr = CHAR_BODY_KOR if kor else CHAR_BODY
    return _para(parapr, _runs_for_text(line, charpr))


_CIRCLED_PREFIX_RE = re.compile(r"^([①②③④⑤])\s*")


def _para_choice(line: str, *, kor_body: bool = False) -> str:
    """보기 한 단락 (paraPr=18, hanging indent).

    평가원 양식은 '①<nbSpace>본문' 형태. 라인이 원문자로 시작하면
    원문자와 본문 사이에 nbSpace 를 끼워 넣어 hanging indent 가 자연스러워지게.
    """
    charpr = CHAR_BODY_KOR if kor_body else CHAR_BODY
    m = _CIRCLED_PREFIX_RE.match(line)
    if m:
        prefix = m.group(1)
        rest = line[m.end():]
        runs: list[str] = []
        if rest:
            # 원문자 + nbSpace + 본문 (인라인 마커는 본문 부분에만 적용)
            runs.append(
                f'<hp:run charPrIDRef="{charpr}">'
                f'<hp:t>{_xe(prefix)}<hp:nbSpace/></hp:t></hp:run>'
            )
            runs.append(_runs_for_text(rest, charpr))
        else:
            runs.append(
                f'<hp:run charPrIDRef="{charpr}">'
                f'<hp:t>{_xe(prefix)}</hp:t></hp:run>'
            )
        return _para(PARA_CHOICE, "".join(runs))
    return _para(PARA_CHOICE, _runs_for_text(line, charpr))


def _para_group_label(label: str) -> str:
    """[36~37] 같은 그룹 라벨 단락."""
    return _para_text(PARA_GROUP_LABEL, CHAR_BODY_KOR, label)


# ─── 박스(네모 테두리) 단락 ──────────────────────────────────────────────────
# 평가원 양식의 박스는 1×1 hp:tbl, borderFillIDRef=6 (4면 SOLID 0.12mm 검정).
# treatAsChar=1 + textWrap=TOP_AND_BOTTOM 으로 본문 안에 인라인되어 있다.
# linesegarray 는 hp 가 자동 재계산하므로 빈 dummy 만 넣어도 된다.

_BOX_BORDER_FILL_ID = 6      # header.xml 의 4면 SOLID 0.12mm 검정 borderFill
_BOX_WIDTH          = 30614  # 평가원 본문폭 (units = 1/7200 inch)
_BOX_TBL_PARA       = 21     # 박스를 감싸는 hp:p 의 paraPr (sample 기준)
_BOX_TBL_RUN_CHAR   = 12     # 박스를 감싸는 hp:run 의 charPr
_BOX_INNER_PARA     = 27     # 박스 셀 안 본문 paraPr (들여쓰기 없음)


def _para_box(lines: list[str], *, kor: bool = False) -> str:
    """라인들을 1×1 hp:tbl 박스로 감싸 한 단락 XML로 반환.

    box 내부는 paraPr=27 (들여쓰기 없음) + 인라인 마커 분리(_xxx_, ______, ①②③④⑤).
    줄바꿈은 lines 배열 단위로 분리되며, 각 line 이 박스 안 한 단락이 된다.
    """
    charpr = CHAR_BODY_KOR if kor else CHAR_BODY
    inner_paras: list[str] = []
    for line in lines:
        if not line or not line.strip():
            inner_paras.append(_para(_BOX_INNER_PARA,
                f'<hp:run charPrIDRef="{charpr}"><hp:t/></hp:run>'))
            continue
        runs = _runs_for_text(line, charpr)
        # 박스 내부 단락은 lineseg 없이도 hp 가 자동 계산
        inner_paras.append(
            f'<hp:p id="0" paraPrIDRef="{_BOX_INNER_PARA}" styleIDRef="0" '
            f'pageBreak="0" columnBreak="0" merged="0">{runs}</hp:p>'
        )
    inner_xml = "".join(inner_paras)

    tbl = (
        f'<hp:tbl id="0" zOrder="0" numberingType="TABLE" '
        f'textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" lock="0" '
        f'dropcapstyle="None" pageBreak="CELL" repeatHeader="1" '
        f'rowCnt="1" colCnt="1" cellSpacing="0" '
        f'borderFillIDRef="{_BOX_BORDER_FILL_ID}" noAdjust="0">'
        f'<hp:sz width="{_BOX_WIDTH}" widthRelTo="ABSOLUTE" '
        f'height="2000" heightRelTo="ABSOLUTE" protect="0"/>'
        f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" '
        f'allowOverlap="0" holdAnchorAndSO="0" vertRelTo="PARA" horzRelTo="PARA" '
        f'vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
        f'<hp:outMargin left="0" right="0" top="0" bottom="0"/>'
        f'<hp:inMargin left="0" right="0" top="0" bottom="0"/>'
        f'<hp:tr><hp:tc name="" header="0" hasMargin="1" protect="0" '
        f'editable="0" dirty="0" borderFillIDRef="{_BOX_BORDER_FILL_ID}">'
        f'<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" '
        f'vertAlign="CENTER" linkListIDRef="0" linkListNextIDRef="0" '
        f'textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">'
        f'{inner_xml}'
        f'</hp:subList>'
        f'<hp:cellAddr colAddr="0" rowAddr="0"/>'
        f'<hp:cellSpan colSpan="1" rowSpan="1"/>'
        f'<hp:cellSz width="{_BOX_WIDTH}" height="2000"/>'
        f'<hp:cellMargin left="850" right="850" top="708" bottom="425"/>'
        f'</hp:tc></hp:tr></hp:tbl>'
    )

    # 박스를 한 단락의 hp:run 안에 인라인 (treatAsChar=1)
    return _para(
        _BOX_TBL_PARA,
        f'<hp:run charPrIDRef="{_BOX_TBL_RUN_CHAR}">{tbl}</hp:run>',
    )


# ─── layout 별 단락 시퀀스 빌더 ───────────────────────────────────────────────

def _is_korean_choice(choices: list[str]) -> bool:
    """보기에 한글이 절반 이상이면 한국어 본문 폰트(CHAR_BODY_KOR) 사용."""
    if not choices:
        return False
    kor = sum(
        1 for c in choices
        if any("가" <= ch <= "힯" for ch in c)
    )
    return kor * 2 >= len(choices)


_LEADING_CIRCLED_RE = re.compile(r"^\s*[①②③④⑤]\s*")


def _strip_leading_circled(text: str) -> str:
    """LLM이 보기 앞에 '① 본문' 처럼 원문자를 이미 박은 경우 제거.

    렌더러가 _para_choice 에서 표준 원문자(symbol) + nbSpace + 본문 형태로
    다시 부착하므로, 입력 단계에서 들어온 원문자는 중복을 피하기 위해 제거한다.
    """
    if not text:
        return text
    return _LEADING_CIRCLED_RE.sub("", text, count=1).strip()


def _build_choices(q: Question, *, choices: list[str] | None = None) -> list[str]:
    """보기 5개 단락. LLM이 이미 ①을 prepend 한 경우 중복 방지.

    choices 인자로 외부에서 보기 리스트를 명시할 수 있음 (sub_questions용).
    """
    src = choices if choices is not None else q.choices
    kor = _is_korean_choice(src)
    out: list[str] = []
    for i in range(5):
        symbol = "①②③④⑤"[i]
        text = _strip_leading_circled(src[i] if i < len(src) else "")
        line = f"{symbol} {text}".rstrip() if text else symbol
        out.append(_para_choice(line, kor_body=kor))
    return out


def _build_passage_indented(passage: list[str]) -> list[str]:
    """추론형/빈칸/마커형 본문: 첫 단락만 들여쓰기, 이후 단락은 일반 단락.

    평가원 양식에서 본문은 보통 1단락이지만, LLM이 줄바꿈을 넣으면 분리됨.
    """
    lines = [ln for ln in passage if ln and ln.strip()]
    if not lines:
        return [_para_text(PARA_PASSAGE_INDENT, CHAR_BODY, "")]
    return [_para_passage(line, indent=True) for line in lines]


def _build_passage_letter(passage: list[str]) -> list[str]:
    """편지/안내문: 들여쓰기 없이 줄바꿈 보존 (paraPr=27)."""
    lines = [ln for ln in passage if ln and ln.strip()]
    if not lines:
        return [_para_text(PARA_PASSAGE_PLAIN, CHAR_BODY, "")]
    return [_para_passage(line, indent=False) for line in lines]


def _prepend_label(passage: list[str], label: str) -> list[str]:
    """passage 첫 줄에 라벨이 없으면 'label ' 형태로 부착. 이미 있으면 그대로."""
    out = [ln for ln in (passage or []) if ln and ln.strip()]
    if not out:
        return [label]
    if out[0].lstrip().startswith(label):
        return out
    out[0] = f"{label} {out[0].lstrip()}"
    return out


def _build_segments(
    sub_passages: list[list[str]] | None, *, label_offset: int,
) -> list[str]:
    """순서배열/장문: (A)/(B)/(C)/(D) 분할 단락. 각 라벨 + 본문 합쳐 1단락."""
    if not sub_passages:
        return []
    labels = ["(A)", "(B)", "(C)", "(D)"]
    out: list[str] = []
    for i, sp in enumerate(sub_passages):
        if not sp:
            continue
        idx = label_offset + i
        label = labels[idx] if idx < len(labels) else f"({chr(65+idx)})"
        joined = " ".join(ln for ln in sp if ln.strip())
        if joined.startswith(label):
            joined = joined[len(label):].lstrip()
        line = f"{label} {joined}"
        out.append(_para_passage(line, indent=True))
    return out


def _build_question_paragraphs(q: Question) -> list[str]:
    """Question 객체 1개 → 단락 XML 리스트.

    layout_pattern + 박스 부착 분기:
      reasoning         : [group_label?, question, ...passage(indent), choices, blank]
      letter_box        : [group_label?, question, BOX(passage), choices, blank]
                          (목적/안내문 박스 안에 본문 전체)
      blank_inline      : [group_label?, question, ...passage(indent, ______), choices, blank]
      marker_inline     : [group_label?, question, BOX(given_sentence)?, ...passage, choices, blank]
                          (38/39 문장삽입은 given_sentence 박스가 본문 위에 들어감)
      passage_segments  : [group_label?, question, BOX(given)?, ...segments, BOX(summary)?, choices, blank]
                          (36/37 주어진 글 박스, 40 요약문 박스)
      long_set          : [group_label?, question, ...passage+segments, choices, sub_questions...]
    """
    paras: list[str] = []
    pattern = q.layout_pattern
    qtype   = q.type or ""
    number  = q.number if q.number is not None else 0

    # 1) group_label (필요 유형만, 빈 문자열은 skip)
    if q.group_label and q.group_label.strip():
        paras.append(_para_group_label(q.group_label.strip()))

    # 2) 질문
    paras.append(_para_question_line(number, q.question_text, q.points))

    # 3) 본문 (유형별 분기 + 박스 부착)
    if pattern == "letter_box":
        # 목적(18) / 안내문(27,28): 본문 전체를 박스 안에
        if qtype in TYPES_WITH_BOX_PASSAGE:
            kor = _is_korean_choice(q.passage) if q.passage else False
            paras.append(_para_box([ln for ln in q.passage if ln.strip()], kor=kor))
        else:
            paras.extend(_build_passage_letter(q.passage))

    elif pattern == "blank_inline":
        paras.extend(_build_passage_indented(q.passage))

    elif pattern == "marker_inline":
        # 문장삽입(38/39): 주어진 문장 박스 → 본문(①②③④⑤)
        if qtype in TYPES_WITH_GIVEN_BOX and q.given_sentence:
            paras.append(_para_box([q.given_sentence.strip()]))
        paras.extend(_build_passage_indented(q.passage))

    elif pattern == "passage_segments":
        # 순서배열(36,37): 주어진 글(passage) 박스 → (A)(B)(C) 분할 단락
        if qtype in TYPES_WITH_GIVEN_BOX:
            if q.passage:
                paras.append(_para_box([" ".join(ln for ln in q.passage if ln.strip())]))
            if q.sub_passages:
                paras.extend(_build_segments(q.sub_passages, label_offset=0))
        else:
            # 기존 동작 (다른 passage_segments 유형 대비)
            if q.given_sentence:
                paras.append(_para_passage(q.given_sentence.strip(), indent=False))
            if q.passage:
                paras.extend(_build_passage_indented(q.passage))
            if q.sub_passages:
                paras.extend(_build_segments(q.sub_passages, label_offset=0))

        # 요약문(40): 본문 다음에 요약문 박스
        if qtype in TYPES_WITH_SUMMARY_BOX and q.summary:
            paras.append(_para_box([q.summary.strip()]))
        elif q.summary:
            paras.append(_para_passage(q.summary.strip(), indent=True))

    elif pattern == "long_set":
        # 41-42 / 43-45: 본문은 박스 없이 일반 단락
        # 장문독해(43-45)는 question_text가 "주어진 글 (A) 다음에..."로 (A)를
        # 명시적으로 가리키므로 passage 첫 줄에 (A) 라벨을 자동 부착한다
        # (LLM이 이미 붙였으면 중복 부착하지 않음).
        passage = q.passage
        if passage and "장문독해" in qtype:
            passage = _prepend_label(passage, "(A)")
        if passage:
            paras.extend(_build_passage_indented(passage))
        if q.sub_passages:
            label_offset = 1 if "장문독해" in qtype else 0
            paras.extend(_build_segments(q.sub_passages, label_offset=label_offset))

    else:
        # fallback
        paras.extend(_build_passage_indented(q.passage))

    # 4) 보기 (메인 question 의 choices)
    # marker_inline 유형(어법29/어휘30/무관35/문장삽입38,39)은 본문 안에 ①~⑤가
    # 이미 박혀있고 그것이 보기를 대신하므로, 본문 뒤 빈 ①~⑤ 단락을 출력하지 않음.
    # (q.choices 데이터는 채점/답안지용으로 보존)
    if pattern != "marker_inline":
        paras.extend(_build_choices(q))

    # 5) sub_questions (장문 세트만)
    if pattern == "long_set" and q.sub_questions:
        for sub_idx, sq in enumerate(q.sub_questions):
            paras.append(_para_blank())
            sub_number = number + 1 + sub_idx  # 41-42: 41+1=42, 43-45: 43+1=44, +2=45
            paras.append(_para_question_line(sub_number, sq.question_text, None))
            paras.extend(_build_choices(q, choices=sq.choices))

    # 6) 문항 사이 빈 단락
    paras.append(_para_blank())

    return paras


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def render_section0(meta: ExamMeta, questions: list[Question]) -> str:
    """meta + 동적 N개 questions 로 section0.xml 텍스트 생성."""
    cache = _cache()

    # 1) 모든 문항의 단락 XML 펼치기 (q.number 보존, 없으면 enumerate idx)
    all_paras: list[str] = []
    for i, q in enumerate(questions):
        if q.number is None:
            q = q.model_copy(update={"number": i + 1})
        all_paras.extend(_build_question_paragraphs(q))
    questions_xml = "".join(all_paras)

    # 2) shell 의 marker 를 치환
    out = cache.section0_shell.replace(_Q_SLOT_MARKER, questions_xml)

    # 3) meta placeholder 치환
    out = _replace_meta(out, meta)
    return out


def render_masterpage(name: str, meta: ExamMeta) -> bytes:
    """masterpage*.xml 파일 1개에 meta 인젝션."""
    raw = _cache().raw_files[name].decode("utf-8")
    return _replace_meta(raw, meta).encode("utf-8")


def _replace_meta(text: str, meta: ExamMeta) -> str:
    return (text
            .replace("{{SUBJECT_NAME}}",    _xe(meta.subject_name))
            .replace("{{EXAM_TITLE}}",      _xe(meta.title))
            .replace("{{TOP_LEFT_LABEL}}",  _xe(meta.top_left_label))
            .replace("{{TOP_RIGHT_LABEL}}", _xe(meta.top_right_label)))


def get_template_files() -> dict[str, bytes]:
    """builder가 ZIP 재패키징 시 사용할 base 파일들."""
    return _cache().raw_files
