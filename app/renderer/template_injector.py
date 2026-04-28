"""
Template injector — templates/template.hwpx 의 placeholder를 데이터로 치환.

templates/template.hwpx (= parser SSOT) 안에는 다음 placeholder가 박혀있음:

  머릿글/제목 (section0.xml + masterpage*.xml):
    {{SUBJECT_NAME}} {{EXAM_TITLE}} {{TOP_LEFT_LABEL}} {{TOP_RIGHT_LABEL}}

  Q1~Q10 슬롯 (section0.xml만, 각 10단락 묶음):
    {{Qn_GROUP_LABEL}} {{Qn_PASSAGE}} {{Qn_QUESTION}} {{Qn_POINTS}}
    {{Qn_CHOICE_1}} ~ {{Qn_CHOICE_5}}

이 모듈은 모듈 로드 시점에 template.hwpx를 분석해
  - shell: Q1 블록을 단일 정규화된 question_block으로 대체한 section0 텍스트
  - block: Q1 단락 10개를 placeholder({{Q_*}})로 정규화한 텍스트
를 메모리에 캐싱한 뒤, 요청마다 N개의 block을 만들어 shell에 끼워넣습니다.

renderer는 N의 상한이 없습니다 (요청 단의 GenerateRequest에서 max_length=50으로 제한).
"""
from __future__ import annotations

import re
import zipfile
from functools import lru_cache
from pathlib import Path

from shared.schemas.question import ExamMeta, Question

_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "template.hwpx"

# Q1 블록 단락 갯수 (template 분석 결과 고정값)
_BLOCK_PARA_COUNT = 10
# 첫 Q블록 시작 단락 idx (= header 단락 갯수)
_FIRST_Q_PARA_IDX = 8
# 마지막 단락(trailing 빈 단락) 직전까지가 Q 슬롯 영역
# section0 단락 총합 = 109 = 8 (header) + 10*10 (Q1~Q10) + 1 (trailing)

_PARA_RE = re.compile(r"<hp:p\b[^>]*>.*?</hp:p>", re.DOTALL)
_Q_SLOT_MARKER = "<!--QUESTIONS_HERE-->"


class _Cache:
    """모듈 로드 시점에 template.hwpx 분석한 결과를 보관."""
    raw_files: dict[str, bytes]   # template ZIP 내용 (mimetype 제외)
    section0_shell: str            # Q슬롯 영역을 _Q_SLOT_MARKER로 대체한 section0.xml
    question_block: str            # 정규화된 1문항 단락 묶음 (placeholder가 {{Q_*}})

    def __init__(self) -> None:
        with zipfile.ZipFile(_TEMPLATE_PATH) as zf:
            self.raw_files = {
                name: zf.read(name) for name in zf.namelist() if name != "mimetype"
            }
        section0 = self.raw_files["Contents/section0.xml"].decode("utf-8")
        self.section0_shell, self.question_block = self._split(section0)

    @staticmethod
    def _split(section0: str) -> tuple[str, str]:
        paras = list(_PARA_RE.finditer(section0))
        if len(paras) < _FIRST_Q_PARA_IDX + _BLOCK_PARA_COUNT:
            raise RuntimeError(
                f"template.hwpx 단락 수가 예상보다 적습니다: {len(paras)}"
            )

        # Q1 블록 = idx 8~17 (10단락)
        q1_start = paras[_FIRST_Q_PARA_IDX].start()
        q1_end_idx = _FIRST_Q_PARA_IDX + _BLOCK_PARA_COUNT - 1
        q1_end = paras[q1_end_idx].end()
        q1_block_xml = section0[q1_start:q1_end]

        # Q1 블록을 정규화 (Q1_ → Q_, '1.' → '{{Q_NUMBER}}.')
        normalized = q1_block_xml.replace("{{Q1_", "{{Q_")
        # number prefix '1. ' 를 '{{Q_NUMBER}}. ' 로 (질문 단락의 시작)
        normalized = normalized.replace(
            "<hp:t>1. {{Q_QUESTION}}",
            "<hp:t>{{Q_NUMBER}}. {{Q_QUESTION}}",
        )

        # shell: section0의 Q1 시작 ~ Q10 마지막 단락까지를 marker로 대체
        # Q10 마지막 단락 idx = 8 + 10*10 - 1 = 107
        q10_last_idx = _FIRST_Q_PARA_IDX + 10 * _BLOCK_PARA_COUNT - 1
        if q10_last_idx >= len(paras):
            raise RuntimeError(
                f"template.hwpx 가 Q10 슬롯까지 갖추지 못함: paras={len(paras)}"
            )
        slots_start = paras[_FIRST_Q_PARA_IDX].start()
        slots_end = paras[q10_last_idx].end()
        shell = section0[:slots_start] + _Q_SLOT_MARKER + section0[slots_end:]

        return shell, normalized


@lru_cache(maxsize=1)
def _cache() -> _Cache:
    return _Cache()


# ─── XML 이스케이프 ───────────────────────────────────────────────────────────

_TRANS = str.maketrans({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "\r": "",
    "\n": " ",
})


def _xe(s: object) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.translate(_TRANS)


# ─── 단일 블록 인젝션 ─────────────────────────────────────────────────────────

def _compose_passage(q: Question) -> str:
    """passage + given_sentence + sub_passages + summary 를 1슬롯 안에 라벨링해 합침.

    (ii) 차선책 — 가독성은 일부 손실, 텍스트는 모두 살아있음.
    추후 (C) 마이그레이션 시 유형별 블록 템플릿으로 분리 예정.
    """
    parts: list[str] = []

    # 38, 39: 주어진 문장
    if q.given_sentence:
        parts.append(f"[주어진 문장] {q.given_sentence.strip()}")

    # 본문
    if q.passage:
        body = " ".join(ln for ln in q.passage if ln.strip())
        if body:
            parts.append(body)

    # 36, 37: (A)(B)(C) 단락 / 43-45: (B)(C)(D) 단락
    if q.sub_passages:
        labels = ["(A)", "(B)", "(C)", "(D)"]
        # passage 가 (A)에 해당하면 sub_passages는 (B) 부터 시작
        # 36/37 형식: passage=주어진 글, sub_passages=[(A),(B),(C)] → 라벨 인덱스 0부터
        # 43-45 형식: passage=(A), sub_passages=[(B),(C),(D)] → 라벨 인덱스 1부터
        # 휴리스틱: sub_passages 길이가 3이면 36/37 (A부터), 길이가 3이고 type 이 장문독해면 (B부터)
        # 단순화: type 정보로 분기
        if q.type and "장문독해" in q.type:
            label_offset = 1   # (B), (C), (D)
        else:
            label_offset = 0   # (A), (B), (C)
        for i, sp in enumerate(q.sub_passages):
            if not sp:
                continue
            label_idx = label_offset + i
            label = labels[label_idx] if label_idx < len(labels) else f"({chr(65+label_idx)})"
            joined = " ".join(ln for ln in sp if ln.strip())
            if not joined:
                continue
            # LLM이 라벨을 그대로 넣었으면 제거 (안전망)
            if joined.startswith(label):
                joined = joined[len(label):].lstrip()
            parts.append(f"{label} {joined}")

    # 40: 요약문
    if q.summary:
        parts.append(f"[요약문] {q.summary.strip()}")

    return "\n\n".join(parts)


def _inject_question_block(template_block: str, q: Question) -> str:
    """정규화된 question_block 템플릿에 한 문항의 데이터를 박는다.

    문항 번호는 q.number 를 사용 (장문 세트 자동 번호 41/42/43/44/45 보존).
    """
    passage = _compose_passage(q)

    # choice가 5개 미만이면 빈 문자열로 채움
    choices = list(q.choices) + [""] * (5 - len(q.choices))

    # points가 None 이면 빈 문자열, 아니면 정수로 표시
    points_str = str(int(q.points)) if q.points is not None else ""

    number = q.number if q.number is not None else 0

    out = template_block
    out = out.replace("{{Q_NUMBER}}",      str(number))
    out = out.replace("{{Q_GROUP_LABEL}}", _xe(q.group_label or ""))
    out = out.replace("{{Q_PASSAGE}}",     _xe(passage))
    out = out.replace("{{Q_QUESTION}}",    _xe(q.question_text))
    out = out.replace("{{Q_POINTS}}",      _xe(points_str))
    for i in range(5):
        out = out.replace(f"{{{{Q_CHOICE_{i+1}}}}}", _xe(choices[i]))
    return out


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def render_section0(meta: ExamMeta, questions: list[Question]) -> str:
    """meta + 동적 N개 questions 로 section0.xml 텍스트 생성."""
    cache = _cache()

    # 1) N개 question_block 만들기 (q.number 보존, 없으면 enumerate idx)
    blocks: list[str] = []
    for i, q in enumerate(questions):
        if q.number is None:
            q = q.model_copy(update={"number": i + 1})
        blocks.append(_inject_question_block(cache.question_block, q))
    questions_xml = "".join(blocks)

    # 2) shell의 marker를 N개 블록으로 치환
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
