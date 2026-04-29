"""
LLM 호출당 원가 계산.

토큰 사용량 → USD → KRW.

- 모델별 $/1M token 테이블은 본 모듈 상수로 관리 (provider 공식 페이지 기준).
- 가격이 바뀌면 PRICE_TABLE 만 갱신하면 전 코드에 반영.
- 환율은 env(USD_KRW_RATE)로 외부 주입 가능, 미설정 시 보수적 기본값.

usage_log 와 짝지어 사용:
  cost_krw = compute_cost(model, in_tokens, out_tokens)
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """1M token 당 USD 단가."""
    input_per_1m:  float
    output_per_1m: float


# ─── 모델별 가격 테이블 (2026-04 시점, provider 공식 페이지 기준) ────────────
# 가격이 바뀌면 여기만 갱신.
PRICE_TABLE: dict[str, ModelPrice] = {
    # Gemini
    "gemini-2.5-flash":      ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
    "gemini-2.5-pro":        ModelPrice(1.25, 10.00),

    # OpenAI
    "gpt-4o":                ModelPrice(2.50, 10.00),
    "gpt-4o-mini":           ModelPrice(0.15, 0.60),
    "gpt-4.1":               ModelPrice(2.00, 8.00),
    "gpt-4.1-mini":          ModelPrice(0.40, 1.60),

    # Anthropic
    "claude-opus-4-7":       ModelPrice(15.00, 75.00),
    "claude-sonnet-4-6":     ModelPrice(3.00, 15.00),
    "claude-haiku-4-5":      ModelPrice(1.00, 5.00),
}

# 가격표에 없는 모델용 보수적 폴백 (Sonnet 가격 — 과대측정해야 비용 누락 안 됨)
FALLBACK_PRICE = ModelPrice(3.00, 15.00)

# 1$ → KRW. env USD_KRW_RATE 로 덮어쓸 수 있음.
DEFAULT_USD_KRW = 1400.0


def _usd_krw() -> float:
    raw = os.environ.get("USD_KRW_RATE", "")
    try:
        v = float(raw)
        if v > 0:
            return v
    except ValueError:
        pass
    return DEFAULT_USD_KRW


def _resolve_price(model: str) -> ModelPrice:
    """모델 ID 의 prefix 매칭으로 가격 조회. 미등록은 FALLBACK_PRICE."""
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    # 'gemini-2.5-flash-001' 같은 변종도 prefix 로 매칭
    for k, v in PRICE_TABLE.items():
        if model.startswith(k):
            return v
    return FALLBACK_PRICE


def compute_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    """모델 + 토큰 사용량 → KRW (소수 둘째자리까지 의미)."""
    p = _resolve_price(model)
    usd = (in_tokens * p.input_per_1m + out_tokens * p.output_per_1m) / 1_000_000
    return round(usd * _usd_krw(), 4)
