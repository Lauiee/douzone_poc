"""집계 결과를 `medical_confusion.DEFAULT_MEDICAL_CONFUSION_SET`에 넣을 수 있는 힌트 문자열 생성.

실제 파일 수정은 하지 않음 — 운영자가 검토 후 수동/스크립트 반영.
"""

from __future__ import annotations

from src.self_learning.models import AggregatedPair


def format_python_dict_entries(pairs: list[AggregatedPair]) -> str:
    """복사-붙여넣기용 파이썬 dict 한 줄씩 (값은 단일 원소 set)."""
    lines: list[str] = []
    for p in pairs:
        # 키/값에 따옴표 이스케이프
        o = p.original.replace("\\", "\\\\").replace('"', '\\"')
        c = p.corrected.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    "{o}": {{"{c}"}},  # count={p.count}')
    return "\n".join(lines)
