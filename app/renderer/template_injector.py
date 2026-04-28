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
from functools import lru_cache
from pathlib import Path

from shared.schemas.question import (
    BLANK_PLACEHOLDER, ExamMeta, Question, UNDERLINE_MARK,
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


@lru_cache(maxsize=1)
def _cache() -> _Cache:
    return _Cache()


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


def _build_choices(q: Question) -> list[str]:
    """보기 5개 단락. 누락분은 빈 ①~⑤ 만 표시."""
    kor = _is_korean_choice(q.choices)
    out: list[str] = []
    for i in range(5):
        symbol = "①②③④⑤"[i]
        text = q.choices[i] if i < len(q.choices) else ""
        # 평가원 양식: '①  text' (원문자 + nbSpace + 본문)
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

    layout_pattern 별 시퀀스:
      reasoning         : [group_label?, question, ...passage(indent), choices, blank]
      letter_box        : [group_label?, question, ...passage(plain), choices, blank]
      blank_inline      : [group_label?, question, ...passage(indent, ______), choices, blank]
      marker_inline     : [group_label?, question, ...passage(indent, ①②③④⑤), choices, blank]
      passage_segments  : [group_label?, question, given?, ...segments, summary?, choices, blank]
      long_set          : [group_label?, question, ...passage+segments, ...sub_questions, blank]
    """
    paras: list[str] = []
    pattern = q.layout_pattern
    number  = q.number if q.number is not None else 0

    # 1) group_label (필요 유형만, 빈 문자열은 skip)
    if q.group_label and q.group_label.strip():
        paras.append(_para_group_label(q.group_label.strip()))

    # 2) 질문
    paras.append(_para_question_line(number, q.question_text, q.points))

    # 3) 본문 (유형별 분기)
    if pattern == "letter_box":
        paras.extend(_build_passage_letter(q.passage))

    elif pattern in ("reasoning", "blank_inline", "marker_inline"):
        paras.extend(_build_passage_indented(q.passage))

    elif pattern == "passage_segments":
        # 38, 39: 주어진 문장이 있으면 본문 위에 표시
        if q.given_sentence:
            paras.append(_para_passage(q.given_sentence.strip(), indent=False))
        # 본문 (있으면)
        if q.passage:
            paras.extend(_build_passage_indented(q.passage))
        # 36, 37: (A)(B)(C) 분할 단락
        if q.sub_passages:
            paras.extend(_build_segments(q.sub_passages, label_offset=0))
        # 40: 요약문
        if q.summary:
            paras.append(_para_passage(q.summary.strip(), indent=True))

    elif pattern == "long_set":
        # 41-42, 43-45: passage 가 (A) 본문, sub_passages 가 (B)(C)(D)
        if q.passage:
            paras.extend(_build_passage_indented(q.passage))
        if q.sub_passages:
            label_offset = 1 if "장문독해" in (q.type or "") else 0
            paras.extend(_build_segments(q.sub_passages, label_offset=label_offset))

    else:
        # fallback
        paras.extend(_build_passage_indented(q.passage))

    # 4) 보기 (long_set 은 sub_questions 가 따로, 메인 choices 만 표시)
    paras.extend(_build_choices(q))

    # 5) sub_questions (장문 세트만)
    if pattern == "long_set" and q.sub_questions:
        for sub_idx, sq in enumerate(q.sub_questions):
            paras.append(_para_blank())
            sub_number = number + 1 + sub_idx  # 41-42: 41+1=42, 43-45: 43+1=44, +2=45
            paras.append(_para_question_line(sub_number, sq.question_text, None))
            kor = _is_korean_choice(sq.choices)
            for i in range(5):
                symbol = "①②③④⑤"[i]
                text = sq.choices[i] if i < len(sq.choices) else ""
                line = f"{symbol} {text}".rstrip() if text else symbol
                paras.append(_para_choice(line, kor_body=kor))

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
