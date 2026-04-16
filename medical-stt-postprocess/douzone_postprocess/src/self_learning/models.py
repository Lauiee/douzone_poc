"""자기학습(피드백 루프)용 이벤트 모델."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CorrectionEvent:
    """사람·외부 시스템이 남기는 (잘못된 표기 → 확정 표기) 한 건."""

    original: str
    corrected: str
    source: str = "human"  # human | api | batch_review 등
    created_at: str = field(default_factory=_utc_now_iso)
    note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class AggregatedPair:
    """집계 결과: 동일 (original → corrected) 빈도."""

    original: str
    corrected: str
    count: int
    last_seen: str | None = None
