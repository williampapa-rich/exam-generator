"""
LLM 호출당 사용량/비용 누적 로그 (jsonl).

한 줄 = 한 호출. 분석은 pandas 한 줄로 충분:
    df = pd.read_json('tools/stats/usage_log.jsonl', lines=True)
    df.groupby('type')['cost_krw'].agg(['mean', 'p50', 'p95', 'count'])

서비스화 시 DB 마이그레이션 1시간 작업.

스레드/asyncio 동시성:
- jsonl append-only 라 race condition 영향이 있을 수 있으나, 한 줄이 한 syscall(write)
  로 떨어지면 OS 가 atomic 처리 (POSIX). asyncio 단일 이벤트 루프에선 사실상 직렬.
- 대량 동시 쓰기로 라인이 섞이면 파일 잠금 도입 검토.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "tools" / "stats" / "usage_log.jsonl"


def _path() -> Path:
    raw = os.environ.get("USAGE_LOG_PATH")
    return Path(raw) if raw else DEFAULT_PATH


def log_call(
    *,
    type:        str,
    provider:    str,
    model:       str,
    in_tokens:   int,
    out_tokens:  int,
    cost_krw:    float,
    attempt:     int,
    success:     bool,
    elapsed_ms:  int,
    extra:       dict[str, Any] | None = None,
) -> None:
    """LLM 호출 1회분을 jsonl 한 줄로 append. 실패해도 main flow 막지 않음."""
    record: dict[str, Any] = {
        "ts":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "type":        type,
        "provider":    provider,
        "model":       model,
        "in_tokens":   in_tokens,
        "out_tokens":  out_tokens,
        "cost_krw":    round(cost_krw, 4),
        "attempt":     attempt,
        "success":     success,
        "elapsed_ms":  elapsed_ms,
    }
    if extra:
        record.update(extra)

    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # 비용 로깅은 보조 기능 — 실패해도 본 흐름은 진행
        import logging
        logging.getLogger("llm.usage").warning("usage_log 쓰기 실패: %s", e)
