"""자기학습(피드백 루프) 골격.

- `capture`: 사람·API가 (원문 구간 → 확정 구간)을 JSONL로 누적
- `aggregate`: 로그를 집계해 반복되는 쌍만 후보로 추림
- `merge_hint`: 사전에 반영할 파이썬 dict 조각 문자열 생성 (자동 덮어쓰기 없음)

파이프라인은 기본적으로 여기를 호출하지 않는다. API에서 확정본과 비교해 `log_pair`만 호출하면 된다.
"""

from src.self_learning.aggregate import aggregate, aggregated_to_json_serializable, load_events
from src.self_learning.capture import append_event, log_pair
from src.self_learning.merge_hint import format_python_dict_entries
from src.self_learning.models import AggregatedPair, CorrectionEvent
from src.self_learning.paths import default_aggregate_output, default_events_path

__all__ = [
    "AggregatedPair",
    "CorrectionEvent",
    "aggregate",
    "aggregated_to_json_serializable",
    "append_event",
    "default_aggregate_output",
    "default_events_path",
    "format_python_dict_entries",
    "load_events",
    "log_pair",
]
