"""
유형별 본문 마커 검증.

prompts.py 에서 LLM 에게 본문에 박아 달라고 요구한 마커가 실제 출력에
들어왔는지 후검증한다. 누락이면 generator 가 재시도(최대 3회) 후 fallback.

검증은 'fail-fast': 첫 번째 위반을 문자열로 반환. 통과하면 None.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from shared.schemas.question import Question

# 본문 길이 hard limit (avg ± LENGTH_TOLERANCE 자)
# 평가원 5년치 실제 분포의 약 95%를 커버하는 폭 — 더 좁히면 평가원 출제도 reject 됨
LENGTH_TOLERANCE = 250

CIRCLED_5 = ("①", "②", "③", "④", "⑤")
ALPHA_5   = ("(a)", "(b)", "(c)", "(d)", "(e)")

_BLANK_RE          = re.compile(r"_{6}(?!_)")          # 정확히 6개의 _ (7개 이상은 제외)
_UNDERLINE_TOKEN_RE = re.compile(r"_([^_\n][^_\n]*?)_") # _foo bar_ 토큰
_ORDER_CHOICE_RE    = re.compile(r"^\([A-C]\)-\([A-C]\)-\([A-C]\)$")  # (A)-(B)-(C) 형식
_LABEL_UNDERLINED_RE = re.compile(r"_\([a-e]\)_")      # _(a)_ 처럼 라벨에 밑줄 들어간 패턴
# '①_word_' 처럼 원문자 마커 바로 뒤에 _..._ 토큰이 붙은 패턴 (마커 ↔ 밑줄 사이 공백 허용 X)
_CIRCLED_UNDERLINE_RE = re.compile(r"([①②③④⑤])_([^_\n][^_\n]*?)_")
# '(a) _word_' 처럼 (a)~(e) 라벨 뒤에 _..._ 가 오는 패턴 (라벨과 밑줄 사이 공백 1+ 허용)
_ALPHA_UNDERLINE_RE   = re.compile(r"\(([a-e])\)\s+_([^_\n][^_\n]*?)_")
# '_①' 처럼 마커가 밑줄 안으로 들어가버린 잘못된 패턴
_UNDERLINE_CIRCLED_RE = re.compile(r"_[①②③④⑤]")

# 본문 _..._ 밑줄 토큰이 허용되는 유형
_UNDERLINE_ALLOWED = frozenset({
    "밑줄함의(21)", "어법(29)", "어휘(30)",
    "장문(41-42)", "장문독해(43-45)",
})
# 본문 ①②③④⑤ 마커가 허용되는 유형
_CIRCLED_ALLOWED = frozenset({
    "어법(29)", "어휘(30)", "무관문장(35)",
    "문장삽입(38)", "문장삽입(39)",
})
# ______ 빈칸이 허용되는 유형 (passage/summary 어디든)
_BLANK_ALLOWED = frozenset({
    "빈칸-구(31)", "빈칸-절(32)", "빈칸-절(33)", "빈칸-절(34)",
    "요약문(40)",
})


def _passage_str(q: Question) -> str:
    return "\n".join(q.passage or [])


@lru_cache(maxsize=1)
def _type_profiles() -> dict:
    """templates/type_profiles.json — 평가원 5년치 본문 길이 통계."""
    path = Path(__file__).resolve().parents[2] / "templates" / "type_profiles.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _measured_length(q: Question) -> int:
    """학습 통계와 비교 가능한 형태로 본문 길이를 잰다.

    learn_from_pdfs 의 측정 단위와 일치시켜야 공정한 비교가 됨.

    - 장문(41-42), 장문독해(43-45): group_passage_len 기준 → passage + sub_passages 합산.
    - 순서배열(36, 37): 학습 시 PDF 텍스트는 (A)/(B)/(C) 가 합쳐진 한 덩어리였음.
      LLM 출력은 passage + sub_passages 분리 구조 → 합산해 동일 단위로 비교.
    - 요약문(40): 학습 시 본문 + 요약 박스 합본. LLM 출력은 passage + summary → 합산.
    - 그 외: passage 만 (마커형도 마커가 passage 안에 박혀 있으므로 그대로 OK).
    """
    if q.type in ("장문(41-42)", "장문독해(43-45)",
                  "순서배열(36)", "순서배열(37)"):
        parts: list[str] = list(q.passage or [])
        for sp in q.sub_passages or []:
            parts.extend(sp)
        return sum(len(p) for p in parts)
    if q.type == "요약문(40)":
        return len(_passage_str(q)) + len(q.summary or "")
    return len(_passage_str(q))


def _check_passage_length(q: Question) -> str | None:
    """본문 길이가 평가원 평균 ±LENGTH_TOLERANCE 자를 벗어나면 재생성 사유 반환."""
    prof = _type_profiles().get(q.type or "")
    if not prof:
        return None
    src = prof.get("group_passage_len") or prof.get("passage_len") or {}
    avg = src.get("avg", 0)
    if avg <= 0:
        return None
    actual = _measured_length(q)
    if actual == 0:
        return None  # 다른 검증에서 빈 본문은 이미 잡힘
    diff = actual - avg
    if abs(diff) > LENGTH_TOLERANCE:
        direction = "초과" if diff > 0 else "미달"
        return (
            f"{q.type}: 본문 길이 {actual}자 — 평가원 5년치 평균 {avg}자 대비 "
            f"{abs(diff)}자 {direction} (hard limit ±{LENGTH_TOLERANCE}자). "
            f"내용 비약 없이 길이를 평균에 맞춰 재작성할 것."
        )
    return None


def _has_all(text: str, needles: tuple[str, ...]) -> str | None:
    missing = [n for n in needles if n not in text]
    if missing:
        return f"본문 마커 누락: {missing}"
    return None


def _count(text: str, needle: str) -> int:
    return text.count(needle)


def _check_alpha_underline_form(text: str, qtype: str) -> str | None:
    """장문(41-42)/장문독해(43-45): '(a) _x_ ... (e) _x_' 형식 검증.

    - 라벨 (a)~(e) 각각이 바로 뒤 _..._ 토큰을 가진 형태로 등장하는지
    - 'word (a)' 처럼 라벨이 단어 뒤에 붙은 역전 패턴이 없는지
    """
    found = {m.group(1) for m in _ALPHA_UNDERLINE_RE.finditer(text)}
    expected = {"a", "b", "c", "d", "e"}
    missing = sorted(expected - found)
    if missing:
        return (f"{qtype}: 라벨 직후 '_word_' 형식 누락 → "
                f"{[f'({x})' for x in missing]}. "
                f"본문 안에 '(a) _word1_ ... (e) _word5_' 처럼 라벨 바로 뒤에 "
                f"공백 + 밑줄 단어/구가 와야 함 (라벨이 단어 뒤에 붙는 'word (a)' 형식 금지)")
    return None


def _is_alpha_choices(choices: list[str]) -> bool:
    """choices 가 ['(a)','(b)','(c)','(d)','(e)'] 형태인지 (앞뒤 공백 허용)."""
    if not choices or len(choices) != 5:
        return False
    for i, c in enumerate(choices):
        if (c or "").strip() != ALPHA_5[i]:
            return False
    return True


def validate_question(q: Question) -> str | None:
    """Question 객체 검증. 통과 시 None, 실패 시 사유 문자열."""
    t = q.type or ""
    passage = _passage_str(q)

    # 0) 공통 — 빈 응답(LLM이 빈 객체를 반환한 케이스) 거르기
    if not (q.question_text or "").strip():
        return f"{t or '?'}: question_text 가 비어있음"
    if not q.passage and not q.summary and not q.sub_passages:
        return f"{t or '?'}: passage/summary/sub_passages 모두 비어있음"

    # 본문 마커 침범 가드: 유형이 허용하지 않는 마커가 들어있으면 reject.
    # 빈칸 ______ 은 _..._ 토큰 정규식에 잡힐 수 있어 먼저 마스킹.
    masked = _BLANK_RE.sub(" ", passage)
    if t and t not in _UNDERLINE_ALLOWED:
        stray = _UNDERLINE_TOKEN_RE.findall(masked)
        if stray:
            return (f"{t}: 본문에 허용되지 않은 _..._ 밑줄 토큰 발견 → {stray[:3]} "
                    f"(이 유형은 본문 밑줄을 사용하지 않음 — 책 제목/고유명사도 평문)")
    if t and t not in _CIRCLED_ALLOWED:
        for sym in CIRCLED_5:
            if sym in passage:
                return (f"{t}: 본문에 허용되지 않은 원문자 마커 '{sym}' 발견 "
                        f"(이 유형은 본문에 ①②③④⑤ 사용 금지)")
    if t and t not in _BLANK_ALLOWED:
        if _BLANK_RE.search(passage):
            return (f"{t}: 본문에 허용되지 않은 ______ 빈칸 발견 "
                    f"(이 유형은 빈칸을 사용하지 않음)")

    # 빈칸형: ______ 6개 _ 가 본문 또는 summary 안에 있어야 함
    if t in ("빈칸-구(31)", "빈칸-절(32)", "빈칸-절(33)", "빈칸-절(34)"):
        if not _BLANK_RE.search(passage):
            return f"{t}: 본문에 ______ (정확히 6개 _) 가 없음"

    if t == "요약문(40)":
        summary = q.summary or ""
        if len(_BLANK_RE.findall(summary)) < 2:
            return "요약문(40): summary 에 ______ 가 2개 들어가야 함"

    # 마커형 (본문 ①②③④⑤ 5개)
    if t in ("어법(29)", "어휘(30)", "무관문장(35)", "문장삽입(38)", "문장삽입(39)"):
        miss = _has_all(passage, CIRCLED_5)
        if miss:
            return f"{t}: {miss}"

    # 어법/어휘: '①_word_' 형식 강제 — 마커 5개 각각 바로 뒤에 _..._ 토큰이 붙어야 함
    # (LLM이 '_①_word_' 처럼 마커를 밑줄 안에 넣으면 정규식 토글이 꼬여 본문 한 단락이
    #  통째로 밑줄 처리되는 케이스가 있었음 → 이걸 사전에 거부해 재시도 유도)
    if t in ("어법(29)", "어휘(30)"):
        # 1) '_①' 처럼 밑줄 안에 마커가 들어간 패턴 거부
        bad = _UNDERLINE_CIRCLED_RE.search(passage)
        if bad:
            return (f"{t}: 마커가 밑줄 토큰 안에 들어간 잘못된 패턴 '{bad.group(0)}' 발견 — "
                    f"올바른 형식은 '①_word_' (마커는 밑줄 밖, 단어만 _..._ 로 감쌀 것)")
        # 2) '_' 총 개수가 짝수인지 (홀수면 닫히지 않은 토큰이 있다는 뜻)
        if passage.count("_") % 2 != 0:
            return (f"{t}: 본문의 '_' 개수가 홀수 — 닫히지 않은 밑줄 토큰이 있음. "
                    f"각 단어/구는 '①_word_' 처럼 정확히 짝지어 밑줄 처리할 것")
        # 3) ①~⑤ 각 마커 바로 뒤에 _..._ 토큰이 붙어 있는지
        found_markers = {m.group(1) for m in _CIRCLED_UNDERLINE_RE.finditer(passage)}
        missing_markers = [c for c in CIRCLED_5 if c not in found_markers]
        if missing_markers:
            return (f"{t}: 마커 직후 '_word_' 형식 누락 → {missing_markers}. "
                    f"본문 안에 '①_word1_ ... ⑤_word5_' 처럼 각 마커 바로 뒤에 "
                    f"밑줄 단어/구가 붙어야 함 (마커-밑줄 사이 공백 금지)")

    # 21번 밑줄 강조: 본문 안에 _xxx_ 토큰이 정확히 1개,
    # 그 텍스트가 question_text 안에 그대로 등장해야 함
    if t == "밑줄함의(21)":
        tokens = _UNDERLINE_TOKEN_RE.findall(passage)
        if len(tokens) == 0:
            return "밑줄함의(21): 본문에 _..._ 밑줄 표현이 없음"
        if len(tokens) > 1:
            return f"밑줄함의(21): 본문 _..._ 밑줄 토큰이 {len(tokens)}개 (정확히 1개여야 함) → {tokens}"
        underline_text = tokens[0].strip()
        if underline_text not in (q.question_text or ""):
            return (f"밑줄함의(21): 본문 밑줄 표현 '{underline_text}' 가 "
                    f"question_text 와 일치하지 않음 → question_text={q.question_text!r}")

    # 문장삽입: given_sentence 필수
    if t in ("문장삽입(38)", "문장삽입(39)"):
        if not (q.given_sentence and q.given_sentence.strip()):
            return f"{t}: given_sentence 가 비어있음"

    # 순서배열: sub_passages 3개 + choices '(X)-(Y)-(Z)' 형식
    if t in ("순서배열(36)", "순서배열(37)"):
        if not q.sub_passages or len(q.sub_passages) != 3:
            return f"{t}: sub_passages 가 정확히 3개여야 함 (현재 {len(q.sub_passages or [])}개)"
        for i, c in enumerate(q.choices or []):
            if not _ORDER_CHOICE_RE.match((c or "").strip()):
                return (f"{t}: choices[{i}] 가 '(X)-(Y)-(Z)' 형식이 아님 (괄호 누락?) → {c!r}")

    # 장문(41-42): (a)~(e) 본문 전체에 5개, sub_questions 길이=1, 42번은 어휘
    if t == "장문(41-42)":
        miss = _has_all(passage, ALPHA_5)
        if miss:
            return f"장문(41-42): {miss}"
        if _LABEL_UNDERLINED_RE.search(passage):
            return ("장문(41-42): 라벨 자체가 밑줄로 감싸진 패턴 '_(x)_' 발견 — "
                    "라벨은 평문, 단어/구만 _..._ 로 밑줄 처리해야 함")
        form_err = _check_alpha_underline_form(passage, "장문(41-42)")
        if form_err:
            return form_err
        if not q.sub_questions or len(q.sub_questions) != 1:
            return f"장문(41-42): sub_questions 길이 1 필요 (현재 {len(q.sub_questions or [])})"
        # 42번 sub_question 은 어휘 — choices 가 ['(a)','(b)','(c)','(d)','(e)']
        sub42 = q.sub_questions[0]
        if not _is_alpha_choices(sub42.choices):
            return (f"장문(41-42): 42번 sub_question 의 choices 가 "
                    f"(a)~(e) 형식이 아님 → {sub42.choices}")

    # 장문독해(43-45): (A)~(D) 4단락, (a)~(e) 본문, sub_questions 길이=2 (지칭/일치)
    if t == "장문독해(43-45)":
        if not q.sub_passages or len(q.sub_passages) != 3:
            return f"장문독해(43-45): sub_passages 가 정확히 3개 (B/C/D)여야 함"
        all_text = passage + "\n" + "\n".join(
            "\n".join(sp) for sp in (q.sub_passages or [])
        )
        miss = _has_all(all_text, ALPHA_5)
        if miss:
            return f"장문독해(43-45): {miss}"
        if _LABEL_UNDERLINED_RE.search(all_text):
            return ("장문독해(43-45): 라벨 자체가 밑줄로 감싸진 패턴 '_(x)_' 발견 — "
                    "라벨은 평문, 대명사/명사구만 _..._ 로 밑줄 처리해야 함")
        form_err = _check_alpha_underline_form(all_text, "장문독해(43-45)")
        if form_err:
            return form_err
        if not q.sub_questions or len(q.sub_questions) != 2:
            return f"장문독해(43-45): sub_questions 길이 2 필요 (현재 {len(q.sub_questions or [])})"
        # 44번 = 지칭 → choices = (a)~(e), 45번 = 일치 → choices = ①~⑤ 텍스트 5개
        if not _is_alpha_choices(q.sub_questions[0].choices):
            return f"장문독해(43-45): 44번 choices 가 (a)~(e) 형식이 아님 → {q.sub_questions[0].choices}"
        if len(q.sub_questions[1].choices) != 5:
            return f"장문독해(43-45): 45번 choices 5개 필요 (현재 {len(q.sub_questions[1].choices)})"

    # 안내문(27/28): passage 가 최소 4줄 이상 (제목 + 헤딩들)
    if t in ("안내문(27)", "안내문(28)"):
        if len([p for p in (q.passage or []) if p.strip()]) < 3:
            return f"{t}: 안내문 본문이 최소 3줄 이상이어야 함"

    # 보기 5개
    if t and not t.startswith("장문"):
        if len(q.choices) != 5:
            return f"{t}: choices 개수 {len(q.choices)} (5개 필요)"

    # 마커형 보기는 정확히 ['①','②','③','④','⑤'] (앞뒤 공백 허용)
    if t in ("어법(29)", "어휘(30)", "무관문장(35)", "문장삽입(38)", "문장삽입(39)"):
        for i, c in enumerate(q.choices):
            stripped = (c or "").strip()
            if stripped not in (CIRCLED_5[i], CIRCLED_5[i] + ""):
                # 일부 LLM이 '① 본문' 처럼 본문을 붙이는데, 마커형은 본문 없음이 원칙
                if not stripped.startswith(CIRCLED_5[i]):
                    return f"{t}: choices[{i}] 가 '{CIRCLED_5[i]}' 가 아님 → {c!r}"

    # 본문 길이 검증 (평가원 5년치 평균 ±LENGTH_TOLERANCE 자)
    length_err = _check_passage_length(q)
    if length_err:
        return length_err

    return None
