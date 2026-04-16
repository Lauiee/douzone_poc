"""파이프라인 루트 해석 및 MedicalSTTPipeline 싱글톤."""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipeline = None


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def resolve_medical_stt_root() -> Path:
    """후처리 루트: 이 디렉터리에 `src/pipeline.py`가 있으면 그 폴더가 루트(단독 배포 번들)."""
    bundle = Path(__file__).resolve().parent
    if (bundle / "src" / "pipeline.py").is_file():
        return bundle

    env = os.environ.get("MEDICAL_STT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "src" / "pipeline.py").is_file():
            raise RuntimeError(
                f"MEDICAL_STT_ROOT={p} 에 src/pipeline.py 가 없습니다."
            )
        return p

    # 예전: medical-stt-postprocess 루트에 douzone_postprocess/ 만 두고 src 는 상위에 둔 경우
    parent = bundle.parent
    if (parent / "src" / "pipeline.py").is_file():
        return parent

    raise RuntimeError(
        "src/pipeline.py 를 찾을 수 없습니다. "
        "douzone_postprocess 번들에 src/ 가 포함됐는지 확인하거나 MEDICAL_STT_ROOT 를 설정하세요."
    )


def ensure_src_on_path() -> Path:
    root = resolve_medical_stt_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def default_dict_path(root: Path) -> str | None:
    p = root / "data" / "medical_dict.txt"
    return str(p) if p.is_file() else None


def build_pipeline():
    root = ensure_src_on_path()
    from src.pipeline import MedicalSTTPipeline, default_kmbert_model_path

    dict_path = os.environ.get("MEDICAL_STT_DICT", "").strip() or default_dict_path(root)
    device = os.environ.get("MEDICAL_STT_DEVICE", "").strip() or None
    model = os.environ.get("MEDICAL_STT_MODEL", "").strip() or None
    mlm_path = os.environ.get("MEDICAL_STT_MLM_MODEL", "").strip() or None
    enable_mlm = not _env_bool("MEDICAL_STT_NO_MLM", False)

    return MedicalSTTPipeline(
        dict_path=dict_path,
        model_name=model or default_kmbert_model_path(),
        device=device,
        enable_bert=not _env_bool("MEDICAL_STT_NO_BERT", False),
        enable_mlm=enable_mlm,
        mlm_model_path=mlm_path if enable_mlm else None,
        enable_kogpt2=_env_bool("MEDICAL_STT_USE_KOGPT2", False),
        enable_kobert_context=_env_bool("MEDICAL_STT_USE_CONTEXT_MLM", True),
    )


def is_pipeline_loaded() -> bool:
    return _pipeline is not None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                logger.info("MedicalSTTPipeline 로드 중…")
                _pipeline = build_pipeline()
                logger.info("MedicalSTTPipeline 준비 완료")
    return _pipeline
