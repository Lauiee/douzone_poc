"""JSONL 이벤트 집계 → 반복되는 (original→corrected) 후보만 추출."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from src.self_learning.models import AggregatedPair
from src.self_learning.paths import default_events_path

logger = logging.getLogger(__name__)


def load_events(path: Path | None = None) -> list[dict]:
    p = path or default_events_path()
    if not p.is_file():
        return []
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("skip bad jsonl line: %s", e)
    return rows


def aggregate(
    path: Path | None = None,
    min_count: int = 2,
) -> list[AggregatedPair]:
    """동일 (original, corrected) 출현 횟수 집계. min_count 이상만 반환."""
    rows = load_events(path)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    last_seen: dict[tuple[str, str], str] = {}
    for r in rows:
        o = (r.get("original") or "").strip()
        c = (r.get("corrected") or "").strip()
        if not o or not c or o == c:
            continue
        key = (o, c)
        counts[key] += 1
        ts = r.get("created_at") or ""
        if ts and (key not in last_seen or ts > last_seen[key]):
            last_seen[key] = ts
    out: list[AggregatedPair] = []
    for (o, c), n in counts.items():
        if n >= min_count:
            out.append(
                AggregatedPair(
                    original=o,
                    corrected=c,
                    count=n,
                    last_seen=last_seen.get((o, c)),
                )
            )
    out.sort(key=lambda x: (-x.count, x.original, x.corrected))
    return out


def aggregated_to_json_serializable(pairs: list[AggregatedPair]) -> list[dict]:
    return [
        {
            "original": p.original,
            "corrected": p.corrected,
            "count": p.count,
            "last_seen": p.last_seen,
        }
        for p in pairs
    ]
