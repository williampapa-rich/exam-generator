"""
PDF 5개년 모의/수능 영어 시험지에서 문항 패턴을 학습한다.

- 좌/우단을 분리해 추출 후 페이지 순서대로 이어붙임
- 합본 PDF (16p = 홀+짝)는 앞 절반만 사용
- 18~45번 독해 문항만 분리, classify.py 재사용해 유형 분류
- 그룹 지시문([31~34], [36~37], [38~39], [41~42], [43~45]) 캐시해 멤버에 분배
- 산출물:
    templates/type_profiles.json         재집계 (분포 P5/P50/P95 포함)
    tools/parser/_data/learning_report.json  분류 실패 / 표본 수 / 길이 분포 / 스키마 충족률
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "samples" / "KSAT and Mock Exam_2022~2026" / "Exam Sheet"
TEMPLATES_DIR = ROOT / "templates"
DATA_DIR = Path(__file__).resolve().parent / "_data"

sys.path.insert(0, str(ROOT))
from shared.schemas.question import Question  # noqa: E402
from tools.parser.classify import classify_type  # noqa: E402

CIRCLED = "①②③④⑤"
GROUP_RANGES = [(31, 34), (36, 37), (38, 39), (41, 42), (43, 45)]


@dataclass
class RawQuestion:
    number: int
    raw_text: str
    source_file: str
    group_instruction: str = ""

    def to_question(self) -> Question:
        # PDF 마지막 페이지 푸터/저작권 라인 제거
        text = re.sub(
            r"\n\s*\*\s*확인\s*사항.*$|\n\s*◦.*$|\n\s*\d+\s*$|\n\s*이\s*문제지에\s*관한.*$",
            "",
            self.raw_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        # 1) 첫 줄(또는 첫 ? 까지)이 질문 — 단, 비어 있으면 group_instruction 사용
        first_line = text.split("\n", 1)[0].strip()
        # "{num}." 제거
        first_line = re.sub(r"^\d+\.\s*", "", first_line)
        question_text = first_line if first_line else self.group_instruction
        if not question_text:
            question_text = self.group_instruction

        # 2) 선택지 분리 — ①…⑤
        choices: list[str] = []
        passage_part = text
        m = re.search(r"[①②③④⑤]", text)
        if m:
            passage_part = text[: m.start()]
            choice_block = text[m.start():]
            # ①~⑤ 단위로 split (구분자 보존)
            parts = re.split(r"([①②③④⑤])", choice_block)
            # parts: ['', '①', 'a', '②', 'b', ...]
            buf: list[str] = []
            cur_sym: str | None = None
            collected: list[str] = []
            for p in parts:
                if p in CIRCLED:
                    if cur_sym is not None:
                        buf.append("".join(collected).strip())
                    cur_sym = p
                    collected = []
                elif cur_sym is not None:
                    collected.append(p)
            if cur_sym is not None:
                buf.append("".join(collected).strip())
            choices = [c for c in buf if c]

        # 3) 지문 — 첫 줄(질문) 빼고 나머지
        lines = passage_part.split("\n")
        # 첫 줄이 질문이면 제거. 단, 첫 줄이 비고 그룹 지시문 사용한 경우는 제거 안 함.
        if lines and re.match(r"^\d+\.", lines[0]):
            lines = lines[1:]
        # 질문 텍스트가 첫 줄로 wrap된 경우(여러 줄 질문) → "?" 까지 모두 제거
        # 단순화를 위해: 첫 비어있지 않은 줄에 "?" 또는 "고르시오"가 있으면 제거
        while lines and ("?" in lines[0] or "고르시오" in lines[0]):
            lines.pop(0)
        passage = [ln.strip() for ln in lines if ln.strip()]

        # 배점 추출
        pts_m = re.search(r"\[(?P<pts>\d+\.?\d*)점\]", text)
        points = float(pts_m.group("pts")) if pts_m else None

        # 빈칸 / 인라인 마커 / 박스 휴리스틱
        has_blanks = bool(re.search(r"_{3,}|\(\s*[A-D]\s*\)", text))
        has_inline = sum(1 for c in passage_part if c in CIRCLED) >= 3

        return Question(
            number=self.number,
            type=None,
            points=points,
            passage=passage,
            question_text=question_text,
            choices=choices,
            has_blanks=has_blanks,
            has_inline_markers=has_inline,
            raw_paragraphs=text.split("\n"),
        )


def _extract_full_text(pdf_path: Path) -> str:
    """좌단→우단 순서로 페이지 텍스트를 이어붙인다.

    합본 PDF (페이지 수 16)는 홀수형(앞 8p)만 사용한다.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        if len(pages) >= 16:
            pages = pages[: len(pages) // 2]
        chunks: list[str] = []
        for pg in pages:
            W, H = pg.width, pg.height
            mid = W / 2
            for box in [(0, 0, mid, H), (mid, 0, W, H)]:
                t = pg.crop(box).extract_text() or ""
                chunks.append(t)
        return "\n".join(chunks)


def _split_questions(full_text: str) -> list[RawQuestion]:
    """18~45번만 추출. 그룹 지시문은 캐시 후 멤버에 부여."""
    # 그룹 지시문 캐시: {start_num: instruction}
    # 평가원 PDF에서 ~ 자리에는 다양한 물결문자가 옴: ~, ∼(U+223C), ～(U+FF5E)
    group_cache: dict[int, str] = {}
    for line in full_text.split("\n"):
        m = re.match(r"^\s*\[(\d+)\s*[~∼～]\s*(\d+)\]\s*(.+)$", line)
        if m:
            start = int(m.group(1))
            instr = m.group(3).strip()
            for s, e in GROUP_RANGES:
                if start == s:
                    group_cache[s] = instr
                    break

    # 문항 분할 — 줄 시작에서 18~45 번호 OR [N~M] 그룹라벨
    splits = re.split(
        r"\n(?=(?:(?:1[89]|[2-3]\d|4[0-5])\.\s|\[\d+\s*[~∼～]\s*\d+\]))",
        full_text,
    )
    out: list[RawQuestion] = []
    for s in splits:
        s = s.strip()
        m = re.match(r"^(\d+)\.", s)
        if not m:
            continue
        num = int(m.group(1))
        if not (18 <= num <= 45):
            continue
        # 그룹 멤버면 지시문 부여
        instr = ""
        for start, end in GROUP_RANGES:
            if start <= num <= end:
                instr = group_cache.get(start, "")
                break
        out.append(RawQuestion(number=num, raw_text=s, source_file="", group_instruction=instr))
    return out


MARKER_TYPES = {"어법(29)", "어휘(30)", "무관문장(35)", "문장삽입(38)", "문장삽입(39)"}


def _marker_passage_length(raw_text: str) -> int:
    """마커형 본문 길이 = 질문문 직후 ~ raw_text 끝.

    raw_text는 이미 (다음 문제 번호 직전)까지로 분리되어 있으므로
    여기서는 앞쪽의 질문문/번호/지시문만 제거하면 된다.
    """
    text = raw_text
    # 1) "29." 형식 번호 제거
    text = re.sub(r"^\d+\.\s*", "", text)
    # 2) 질문문 한 줄(또는 두 줄에 걸친) 제거 — '?' 또는 '?[3점]' 까지
    #    "다음 ... 적절하지 않은 것은? [3점]" 패턴
    m = re.search(r"\?(?:\s*\[\d+\.?\d*점\])?\s*\n", text)
    if m:
        text = text[m.end():]
    # 3) 38, 39처럼 질문문이 없는 경우 — 첫 줄이 빈 줄이면 제거
    text = text.lstrip("\n")
    # 4) 페이지 번호 / 각주 / 저작권 제거
    text = re.sub(r"\n\s*\d{1,2}\s*\n\s*\d{1,2}\s*$", "", text)
    text = re.sub(r"\n\s*\*+[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\*\s*확인\s*사항.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*이\s*문제지에\s*관한.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\d+\s*$", "", text)
    return len(text.strip())


def _extract_group_passage(full_text: str, start_num: int) -> str:
    """[41~42] / [43~45] 지시문 직후 ~ 첫 멤버 문항(41./43.) 직전까지를 그룹 본문으로 추출."""
    label_re = re.compile(rf"\[{start_num}\s*[~∼～]\s*\d+\]\s*[^\n]*\n")
    m = label_re.search(full_text)
    if not m:
        return ""
    body_start = m.end()
    end_re = re.compile(rf"\n\s*{start_num}\.\s")
    em = end_re.search(full_text, body_start)
    if not em:
        return ""
    return full_text[body_start: em.start()].strip()


def _percentile(vals: list[int], p: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return int(s[f] + (s[c] - s[f]) * (k - f))


@dataclass
class TypeStats:
    samples: int = 0
    passage_lens: list[int] = field(default_factory=list)
    choice_lens: list[int] = field(default_factory=list)
    has_box: int = 0
    has_blanks: int = 0
    has_inline: int = 0
    sample_questions: list[str] = field(default_factory=list)
    sample_choices: list[list[str]] = field(default_factory=list)

    def add(self, q: Question, override_passage_len: int | None = None) -> None:
        self.samples += 1
        self.passage_lens.append(
            override_passage_len if override_passage_len is not None else len(q.passage_text)
        )
        # 평균 선택지 길이
        if q.choices:
            self.choice_lens.append(int(statistics.mean(len(c) for c in q.choices)))
        if q.has_passage_box:
            self.has_box += 1
        if q.has_blanks:
            self.has_blanks += 1
        if q.has_inline_markers:
            self.has_inline += 1
        if len(self.sample_questions) < 3 and q.question_text:
            self.sample_questions.append(q.question_text)
        if len(self.sample_choices) < 3 and q.choices:
            self.sample_choices.append(q.choices[:5])

    def to_profile(self) -> dict:
        plens = self.passage_lens or [0]
        clens = self.choice_lens or [0]
        return {
            "samples": self.samples,
            "passage_len": {
                "avg": int(statistics.mean(plens)),
                "min": min(plens),
                "max": max(plens),
                "p5":  _percentile(plens, 0.05),
                "p50": _percentile(plens, 0.50),
                "p95": _percentile(plens, 0.95),
            },
            "choice_len": {
                "avg": int(statistics.mean(clens)),
                "min": min(clens),
                "max": max(clens),
            },
            "has_box_ratio":     round(self.has_box / max(self.samples, 1), 2),
            "has_blanks_ratio":  round(self.has_blanks / max(self.samples, 1), 2),
            "has_inline_ratio":  round(self.has_inline / max(self.samples, 1), 2),
            "sample_questions":  self.sample_questions,
            "sample_choices":    self.sample_choices,
        }


def main() -> None:
    pdfs = sorted(SAMPLES_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"❌ no PDFs in {SAMPLES_DIR}", file=sys.stderr)
        sys.exit(1)

    type_stats: dict[str, TypeStats] = {}
    failures: list[dict] = []   # 분류 실패 / 추출 이상
    by_file: list[dict] = []
    # 장문 세트 본문 길이 (그룹 첫 문항 직전까지의 공유 본문)
    group_passage_lens: dict[str, list[int]] = {
        "장문(41-42)": [],
        "장문독해(43-45)": [],
    }

    for pdf_path in pdfs:
        full = _extract_full_text(pdf_path)
        raws = _split_questions(full)
        # 그룹 본문 길이 측정
        for type_code, start in (("장문(41-42)", 41), ("장문독해(43-45)", 43)):
            body = _extract_group_passage(full, start)
            if body:
                group_passage_lens[type_code].append(len(body))
        per_file_types: dict[str, int] = {}
        for r in raws:
            r.source_file = pdf_path.name
            q = r.to_question()
            t = classify_type(q)
            q.type = t
            # 마커형은 본문 길이를 별도 정의로 재측정 (질문문 직후 ~ raw_text 끝)
            override_len = (
                _marker_passage_length(r.raw_text) if t in MARKER_TYPES else None
            )
            per_file_types[t] = per_file_types.get(t, 0) + 1
            ts = type_stats.setdefault(t, TypeStats())
            ts.add(q, override_passage_len=override_len)
            if t == "미분류":
                failures.append({
                    "file": pdf_path.name,
                    "number": q.number,
                    "question_head": (q.question_text or "")[:80],
                    "passage_head": q.passage_text[:80],
                })
            # 18~45 범위에서 선택지가 5개가 아닌 경우 추출 이상 의심
            if q.choices and len(q.choices) != 5:
                failures.append({
                    "file": pdf_path.name,
                    "number": q.number,
                    "issue": f"choices={len(q.choices)}",
                    "type": t,
                })
        by_file.append({
            "file": pdf_path.name,
            "questions_extracted": len(raws),
            "types": per_file_types,
        })

    # 1) type_profiles.json 갱신 (학습된 분포)
    profile_out = {t: ts.to_profile() for t, ts in sorted(type_stats.items())}
    # 장문 세트는 멤버 단위 본문 길이가 0이라 의미 없음 — 그룹 단위 본문 길이를 별도 주입
    for type_code, lens in group_passage_lens.items():
        if type_code in profile_out and lens:
            profile_out[type_code]["group_passage_len"] = {
                "n":   len(lens),
                "avg": int(statistics.mean(lens)),
                "min": min(lens),
                "max": max(lens),
                "p5":  _percentile(lens, 0.05),
                "p50": _percentile(lens, 0.50),
                "p95": _percentile(lens, 0.95),
            }
    (TEMPLATES_DIR / "type_profiles.json").write_text(
        json.dumps(profile_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) 학습 리포트
    DATA_DIR.mkdir(exist_ok=True)
    # 불일치/주의사항 자동 검출
    notes: list[str] = []
    expected = {
        "목적(18)", "심경(19)", "주장(20)", "밑줄함의(21)", "요지(22)", "주제(23)",
        "제목(24)", "도표(25)", "인물일치(26)", "안내문(27)", "안내문(28)",
        "어법(29)", "어휘(30)", "빈칸-구(31)", "빈칸-절(32)", "빈칸-절(33)",
        "빈칸-절(34)", "무관문장(35)", "순서배열(36)", "순서배열(37)",
        "문장삽입(38)", "문장삽입(39)", "요약문(40)",
        "장문(41-42)", "장문독해(43-45)",
    }
    missing = expected - set(type_stats)
    extra = set(type_stats) - expected
    if missing:
        notes.append(f"누락 유형: {sorted(missing)}")
    if extra:
        notes.append(f"표준 외 유형 검출: {sorted(extra)} — classify.py 매핑 확인 필요")
    if "어휘(30)" in type_stats and type_stats["어휘(30)"].passage_lens \
            and min(type_stats["어휘(30)"].passage_lens) < 100:
        notes.append(
            "어휘(30) 본문 길이 min < 100. 인라인 ① 마커가 본문 안에 있어 분리 시 본문이 잘려 보임. "
            "실제 본문은 이보다 김. 마커 처리 보강 필요."
        )

    report = {
        "pdfs_processed":    len(pdfs),
        "total_questions":   sum(ts.samples for ts in type_stats.values()),
        "type_count":        len(type_stats),
        "notes":             notes,
        "by_file":           by_file,
        "failures":          failures,
        "samples_per_type":  {t: ts.samples for t, ts in sorted(type_stats.items())},
    }
    (DATA_DIR / "learning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"✅ {len(pdfs)} PDFs processed")
    print(f"   {report['total_questions']} questions extracted")
    print(f"   {report['type_count']} types observed")
    print(f"   {len(failures)} failures/issues")
    print(f"\n   templates/type_profiles.json updated")
    print(f"   tools/parser/_data/learning_report.json written")


if __name__ == "__main__":
    main()
