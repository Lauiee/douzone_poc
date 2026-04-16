"""교정 피드백을 JSONL에 누적. 자동 병합은 하지 않음(승인 후 반영)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.self_learning.models import CorrectionEvent
from src.self_learning.paths import default_events_path

logger = logging.getLogger(__name__)


def append_event(
    event: CorrectionEvent,
    path: Path | None = None,
) -> Path:
    """이벤트 한 줄 append. 디렉터리가 없으면 생성."""
    p = path or default_events_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event.to_json_dict(), ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.debug("correction event appended: %s", p)
    return p


def log_pair(
    original: str,
    corrected: str,
    *,
    source: str = "human",
    note: str = "",
    meta: dict | None = None,
    path: Path | None = None,
) -> Path:
    """편의 함수: (원문 구간, 확정 구간)만 넘겨 기록."""
    if original == corrected:
        logger.warning("log_pair: original == corrected, skip")
        return path or default_events_path()
    ev = CorrectionEvent(
        original=original.strip(),
        corrected=corrected.strip(),
        source=source,
        note=note,
        meta=meta or {},
    )
    return append_event(ev, path=path)
