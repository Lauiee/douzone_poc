#!/usr/bin/env python3
"""집계: data/learning/correction_events.jsonl → aggregated_candidates.json + 터미널 요약."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.self_learning.aggregate import aggregate, aggregated_to_json_serializable
from src.self_learning.merge_hint import format_python_dict_entries
from src.self_learning.paths import default_aggregate_output, default_events_path


def main() -> None:
    p = argparse.ArgumentParser(description="교정 이벤트 JSONL 집계")
    p.add_argument("--events", type=str, default=None, help="correction_events.jsonl 경로")
    p.add_argument("-o", "--output", type=str, default=None, help="aggregated_candidates.json")
    p.add_argument("--min-count", type=int, default=2, help="이 횟수 이상인 쌍만")
    p.add_argument("--print-hint", action="store_true", help="medical_confusion에 넣을 dict 줄 출력")
    args = p.parse_args()

    ev_path = Path(args.events) if args.events else default_events_path()
    out_path = Path(args.output) if args.output else default_aggregate_output()

    pairs = aggregate(ev_path, min_count=args.min_count)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(aggregated_to_json_serializable(pairs), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"집계 {len(pairs)}건 → {out_path}")

    if args.print_hint and pairs:
        print("\n# medical_confusion.py DEFAULT_MEDICAL_CONFUSION_SET 에 병합 검토용:\n")
        print(format_python_dict_entries(pairs))


if __name__ == "__main__":
    main()
