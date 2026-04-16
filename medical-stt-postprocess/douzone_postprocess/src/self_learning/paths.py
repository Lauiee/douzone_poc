"""학습 로그·산출물 기본 경로."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def default_events_path() -> Path:
    """append-only JSONL: 한 줄에 CorrectionEvent JSON 하나."""
    return _PROJECT_ROOT / "data" / "learning" / "correction_events.jsonl"


def default_aggregate_output() -> Path:
    """집계 스크립트가 쓰는 후보 목록 (JSON)."""
    return _PROJECT_ROOT / "data" / "learning" / "aggregated_candidates.json"
