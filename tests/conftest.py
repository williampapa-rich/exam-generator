"""pytest 공통 설정.

- usage_log: 테스트 실행 시 실 운영 jsonl 을 더럽히지 않도록 항상 임시 경로 강제.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_usage_log(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("USAGE_LOG_PATH", str(tmp_path / "usage_log.jsonl"))
    yield
