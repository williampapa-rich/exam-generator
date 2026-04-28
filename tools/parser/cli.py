"""
Parser CLI — HWPX 표준 시험지를 분석해 templates/ 산출물을 생성합니다.

⚠️ 중요: templates/template.hwpx 는 SSOT (현재 = template_fixed.hwpx).
  통계/스타일 학습은 누적되지만, **template.hwpx 자체는 --set-template 명시 시에만 교체**됩니다.

진입점:
    # 학습 (통계만 누적, template.hwpx는 그대로)
    python -m tools.parser.cli learn <hwpx>           # replace (기존 학습 폐기)
    python -m tools.parser.cli learn <hwpx> --merge   # 누적

    # SSOT 템플릿 자체 교체 (위험: template.hwpx 덮어씀)
    python -m tools.parser.cli set-template <hwpx>

    # 단순 검사
    python -m tools.parser.cli detect <hwpx>          # 문제 감지 결과만 출력
    python -m tools.parser.cli classify <hwpx>        # 유형 분류 결과 출력
    python -m tools.parser.cli stats                  # 현재 누적 통계 요약

산출물 위치:
    templates/template.hwpx      # base ZIP (set-template 시에만 교체)
    templates/header.xml         # SSOT 템플릿에서 추출 (set-template 시 갱신)
    templates/styles.py          # 다수결로 결정된 ID 상수 (learn 시 갱신)
    templates/type_profiles.json # 유형별 통계 (learn 시 갱신)
    templates/schema.json        # Question.model_json_schema() (항상 자동)

비배포 누적 데이터:
    tools/parser/_data/type_profiles_raw.json   # 유형별 raw 측정값
    tools/parser/_data/styles_observations.json # 페이지/단 후보 관측값
    tools/parser/_data/samples_log.json         # 학습 이력
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.parser.unpack import unpack
from tools.parser.layout import extract_layout
from tools.parser.questions import detect_questions, print_questions
from tools.parser.classify import classify_type
from shared.schemas.question import Question


# ─── 경로 ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "templates"
DATA_DIR      = ROOT / "tools" / "parser" / "_data"

RAW_PROFILES_FILE = DATA_DIR / "type_profiles_raw.json"
STYLES_OBS_FILE   = DATA_DIR / "styles_observations.json"
SAMPLES_LOG_FILE  = DATA_DIR / "samples_log.json"


# ─── raw 데이터 I/O ───────────────────────────────────────────────────────────

def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── learn 명령: 통계/스타일 누적 (template.hwpx 미변경) ────────────────────

def cmd_learn(hwpx_path: Path, out_dir: Path, merge: bool) -> None:
    if not hwpx_path.exists():
        sys.exit(f"❌ 입력 파일 없음: {hwpx_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 학습 입력: {hwpx_path}")
    print(f"📤 출력: {out_dir}")
    print(f"🔀 모드: {'merge (누적)' if merge else 'replace (기존 학습 폐기)'}\n")

    hwpx = unpack(hwpx_path)
    questions = detect_questions(hwpx)
    annotated = [(q, classify_type(q)) for q in questions]
    layout = extract_layout(hwpx)

    print(f"  ✓ {len(questions)} 문제 감지")
    print(f"  ✓ 레이아웃: {layout.summary()}")
    print(f"  ℹ️  template.hwpx 는 변경하지 않습니다 (SSOT 보호).\n")

    # ─── 1. styles.py 누적 갱신 ─────────────────────────────────────────────
    obs = _load_json(STYLES_OBS_FILE, default={"layouts": [], "fonts": {}}) if merge \
          else {"layouts": [], "fonts": {}}
    obs["layouts"].append({
        "source": str(hwpx_path.name),
        "page_width":  layout.page.width,
        "page_height": layout.page.height,
        "margin_left": layout.page.margin_left,
        "margin_right": layout.page.margin_right,
        "margin_top":  layout.page.margin_top,
        "margin_bot":  layout.page.margin_bot,
        "header_len":  layout.page.header_len,
        "footer_len":  layout.page.footer_len,
        "col_count":   layout.columns.count,
        "col_gap":     layout.columns.gap,
        "has_header":  layout.header is not None,
        "has_footer":  layout.footer is not None,
        "header_has_pagenum": bool(layout.header and layout.header.has_pagenum),
        "footer_has_pagenum": bool(layout.footer and layout.footer.has_pagenum),
    })
    _save_json(STYLES_OBS_FILE, obs)

    styles_py = _render_styles_py(obs, sample_count=len(obs["layouts"]))
    sty_path = out_dir / "styles.py"
    sty_path.write_text(styles_py, encoding="utf-8")
    print(f"  ✅ {sty_path.relative_to(ROOT)} ({sty_path.stat().st_size:,} bytes)")

    # ─── 2. type_profiles.json 누적 갱신 ────────────────────────────────────
    raw = _load_json(RAW_PROFILES_FILE, default={"samples": [], "by_type": {}}) if merge \
          else {"samples": [], "by_type": {}}
    raw["samples"].append({
        "file": hwpx_path.name,
        "ingested_at": datetime.now().date().isoformat(),
        "questions_extracted": len(questions),
    })
    by_type: dict = raw["by_type"]
    for q, type_code in annotated:
        bucket = by_type.setdefault(type_code, {
            "passage_lens": [], "choice_lens": [], "had_box": [],
            "had_blanks": [], "had_inline_markers": [],
            "sample_questions": [], "sample_choices": [],
        })
        bucket["passage_lens"].append(len(q.passage_text))
        bucket["choice_lens"].append(
            int(sum(len(c) for c in q.choices) / len(q.choices)) if q.choices else 0
        )
        bucket["had_box"].append(q.has_passage_box)
        bucket["had_blanks"].append(q.has_blanks)
        bucket["had_inline_markers"].append(q.has_inline_markers)
        if q.question_text and len(bucket["sample_questions"]) < 3:
            bucket["sample_questions"].append(q.question_text)
        if q.choices and len(bucket["sample_choices"]) < 3:
            bucket["sample_choices"].append(q.choices)
    _save_json(RAW_PROFILES_FILE, raw)

    derived = _derive_type_profiles(raw)
    prof_path = out_dir / "type_profiles.json"
    _save_json(prof_path, derived)
    print(f"  ✅ {prof_path.relative_to(ROOT)} ({len(derived)} types, {prof_path.stat().st_size:,} bytes)")

    # ─── 3. schema.json (입력 무관, 항상 SSOT의 model_json_schema) ──────────
    schema_path = out_dir / "schema.json"
    _save_json(schema_path, Question.model_json_schema())
    print(f"  ✅ {schema_path.relative_to(ROOT)} ({schema_path.stat().st_size:,} bytes)")

    # ─── 4. samples_log.json (이력) ─────────────────────────────────────────
    log = _load_json(SAMPLES_LOG_FILE, default=[])
    log.append({
        "file":        hwpx_path.name,
        "ingested_at": datetime.now().isoformat(),
        "command":     "learn",
        "mode":        "merge" if merge else "replace",
        "questions":   len(questions),
    })
    _save_json(SAMPLES_LOG_FILE, log)
    print(f"  ✅ {SAMPLES_LOG_FILE.relative_to(ROOT)} ({len(log)} entries)\n")

    print(f"📊 누적 sample 수: {len(raw['samples'])}")


# ─── set-template 명령: SSOT 자체 교체 (위험) ────────────────────────────────

def cmd_set_template(hwpx_path: Path, out_dir: Path) -> None:
    if not hwpx_path.exists():
        sys.exit(f"❌ 입력 파일 없음: {hwpx_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"⚠️  template.hwpx (SSOT) 를 다음으로 교체합니다:")
    print(f"    {hwpx_path}\n")

    hwpx = unpack(hwpx_path)

    tpl_path = out_dir / "template.hwpx"
    tpl_path.write_bytes(hwpx_path.read_bytes())
    print(f"  ✅ {tpl_path.relative_to(ROOT)} ({tpl_path.stat().st_size:,} bytes)")

    if "Contents/header.xml" in hwpx.raw_bytes:
        hdr_path = out_dir / "header.xml"
        hdr_path.write_bytes(hwpx.raw_bytes["Contents/header.xml"])
        print(f"  ✅ {hdr_path.relative_to(ROOT)} ({hdr_path.stat().st_size:,} bytes)")

    log = _load_json(SAMPLES_LOG_FILE, default=[])
    log.append({
        "file":        hwpx_path.name,
        "ingested_at": datetime.now().isoformat(),
        "command":     "set-template",
    })
    _save_json(SAMPLES_LOG_FILE, log)
    print(f"\n📌 SSOT 교체 완료. renderer는 다음 요청부터 새 template.hwpx 사용.")


def _render_styles_py(obs: dict, sample_count: int) -> str:
    """관측 데이터에서 다수결로 styles.py 생성"""
    layouts = obs["layouts"]
    if not layouts:
        return "# (no observations)\n"

    def majority(key: str) -> Any:
        c = Counter(layout.get(key) for layout in layouts)
        return c.most_common(1)[0][0]

    return f'''"""
자동 추출된 스타일/레이아웃 상수
  분석된 샘플 수: {sample_count}
  자동 생성됨 — `python -m tools.parser.cli learn <hwpx>` 로 갱신
  수정하지 마세요. 다시 추출하면 덮어써집니다.
"""

# ─── 페이지 레이아웃 (다수결) ─────────────────────────────────────────────────
PAGE_WIDTH    = {majority("page_width")}
PAGE_HEIGHT   = {majority("page_height")}
MARGIN_L      = {majority("margin_left")}
MARGIN_R      = {majority("margin_right")}
MARGIN_T      = {majority("margin_top")}
MARGIN_B      = {majority("margin_bot")}
HEADER_H      = {majority("header_len")}
FOOTER_H      = {majority("footer_len")}

COL_COUNT     = {majority("col_count")}
COL_GAP       = {majority("col_gap")}

HAS_HEADER         = {majority("has_header")}
HAS_FOOTER         = {majority("has_footer")}
HEADER_HAS_PAGENUM = {majority("header_has_pagenum")}
FOOTER_HAS_PAGENUM = {majority("footer_has_pagenum")}
'''


def _derive_type_profiles(raw: dict) -> dict:
    """raw 누적 측정값 → derived 통계 (LLM prompt에 주입할 가벼운 형태)"""
    out: dict = {}
    for type_code, bucket in raw["by_type"].items():
        n = len(bucket["passage_lens"])
        if n == 0:
            continue
        plens = bucket["passage_lens"]
        clens = bucket["choice_lens"]
        out[type_code] = {
            "samples":          n,
            "avg_passage_len":  int(sum(plens) / n),
            "min_passage_len":  min(plens),
            "max_passage_len":  max(plens),
            "avg_choice_len":   int(sum(clens) / n) if clens else 0,
            "has_box_ratio":    round(sum(bucket["had_box"]) / n, 2),
            "has_blanks_ratio": round(sum(bucket["had_blanks"]) / n, 2),
            "has_inline_ratio": round(sum(bucket["had_inline_markers"]) / n, 2),
            "sample_questions": bucket["sample_questions"],
            "sample_choices":   bucket["sample_choices"],
        }
    return out


# ─── 단순 검사 명령 ───────────────────────────────────────────────────────────

def cmd_detect(hwpx_path: Path) -> None:
    hwpx = unpack(hwpx_path)
    qs = detect_questions(hwpx)
    print_questions(qs)


def cmd_classify(hwpx_path: Path) -> None:
    hwpx = unpack(hwpx_path)
    qs = detect_questions(hwpx)
    counts: dict = defaultdict(int)
    for q in qs:
        t = classify_type(q)
        counts[t] += 1
        print(f"  Q{q.number:>2} → {t}")
    print(f"\n분류 분포: {dict(counts)}")


def cmd_stats() -> None:
    raw = _load_json(RAW_PROFILES_FILE, default={"samples": [], "by_type": {}})
    log = _load_json(SAMPLES_LOG_FILE, default=[])
    print(f"📚 누적 샘플: {len(raw['samples'])}")
    print(f"📜 학습 이력: {len(log)} 회")
    if not raw["by_type"]:
        print("  (학습된 데이터 없음)")
        return
    print(f"\n유형별 통계:")
    for t, b in sorted(raw["by_type"].items()):
        n = len(b["passage_lens"])
        avg_p = int(sum(b["passage_lens"]) / n) if n else 0
        print(f"  {t:<25} n={n:>2}  avg_passage={avg_p:>5}자")


# ─── argparse ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="tools.parser.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("learn", help="통계/스타일 학습 (template.hwpx 미변경)")
    pl.add_argument("hwpx", type=Path)
    pl.add_argument("--out", type=Path, default=TEMPLATES_DIR)
    pl.add_argument("--merge", action="store_true", help="기존 누적 데이터에 추가")

    ps = sub.add_parser("set-template", help="⚠️ SSOT 템플릿(template.hwpx) 자체 교체")
    ps.add_argument("hwpx", type=Path)
    ps.add_argument("--out", type=Path, default=TEMPLATES_DIR)

    pd = sub.add_parser("detect", help="문제 감지 결과 출력")
    pd.add_argument("hwpx", type=Path)

    pc = sub.add_parser("classify", help="유형 분류 결과 출력")
    pc.add_argument("hwpx", type=Path)

    sub.add_parser("stats", help="현재 누적 통계 요약")

    args = p.parse_args()
    if args.cmd == "learn":
        cmd_learn(args.hwpx, args.out, args.merge)
    elif args.cmd == "set-template":
        cmd_set_template(args.hwpx, args.out)
    elif args.cmd == "detect":
        cmd_detect(args.hwpx)
    elif args.cmd == "classify":
        cmd_classify(args.hwpx)
    elif args.cmd == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()
