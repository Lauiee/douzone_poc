"""의료 용어 오인식 표 — 규칙 직후 **고정 치환**은 여기만 사용한다.

질환·소견·처치 등 **진짜 의료 용어** 매핑만 둔다. 구어 오인식(취한/체한, 개를/배를 등)은
`kobert_context` 등 **문맥 기반 교정** 단계에서 다룬다.

`apply_confusion_replacements` 기본값은 `DEFAULT_MEDICAL_CONFUSION_SET`만 적용한다.

`DEFAULT_CONFUSION_SET`은 MLM 모듈용으로, 의료 매핑 + MLM 표면 차단(빈 set)을 합친 것이다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# --- 고정 치환: 의료 용어만 (항상 적용) ---
DEFAULT_MEDICAL_CONFUSION_SET: dict[str, set[str]] = {
    "인구염": {"인후염"},
    "극성충수염": {"급성충수염"},
    "극성수수염": {"급성충수염"},
    "극성충수염입니다": {"급성충수염입니다"},
    "반동성": {"반동통"},
    "약풍": {"압통"},
    "돌중": {"통증"},
    "킥스": {"깁스"},
    "고석": {"고정"},
    "쳐단": {"처방"},
}

# --- MLM 단계 전용: 이 표면은 교정하지 않음 (빈 set). 고정 치환 대상 아님. ---
MLM_SURFACE_BLOCKS: dict[str, set[str]] = {
    "쪼이는": set(),
    "혹시": set(),
    "이번": set(),
    "수술을": set(),
    "수속하면": set(),
    "하나요": set(),
    "힘들어요": set(),
}

# MlmCorrector 등에서 legacy 이름으로 쓰는 합집합
DEFAULT_CONFUSION_SET: dict[str, set[str]] = {
    **DEFAULT_MEDICAL_CONFUSION_SET,
    **MLM_SURFACE_BLOCKS,
}


def apply_confusion_replacements(
    text: str,
    confusion_set: dict[str, set[str]] | None = None,
) -> tuple[str, list[dict]]:
    """긴 키를 우선하여 부분 치환이 꼬이지 않게 한다. 비어 있지 않은 value만 치환."""
    cmap = confusion_set if confusion_set is not None else DEFAULT_MEDICAL_CONFUSION_SET
    keys = sorted(
        (k for k, v in cmap.items() if v),
        key=len,
        reverse=True,
    )
    out = text
    changes: list[dict] = []
    for key in keys:
        allowed = cmap[key]
        if not allowed:
            continue
        if key not in out:
            continue
        target = sorted(allowed)[0]
        before = out
        out = out.replace(key, target)
        if out != before:
            n = before.count(key)
            changes.append(
                {
                    "type": "medical_confusion",
                    "original": key,
                    "corrected": target,
                    "occurrences": n,
                }
            )
            logger.info(
                "[Medical confusion] %s → %s (%d곳)",
                key,
                target,
                n,
            )
    return out, changes
