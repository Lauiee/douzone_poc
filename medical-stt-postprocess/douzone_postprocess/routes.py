"""FastAPI 라우터 (다른 앱에 include_router 용)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from douzone_postprocess.core import get_pipeline, is_pipeline_loaded, resolve_medical_stt_root

router = APIRouter(tags=["medical-stt-postprocess"])


class CorrectBody(BaseModel):
    text: str = Field(..., min_length=1, description="후처리할 문장/문단")


class BatchBody(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="후처리할 문자열 목록")


@router.get("/health")
def health():
    try:
        root = resolve_medical_stt_root()
    except RuntimeError as e:
        return {"status": "error", "detail": str(e)}
    return {
        "status": "ok",
        "medical_stt_root": str(root),
        "pipeline_loaded": is_pipeline_loaded(),
    }


@router.post("/v1/correct")
def correct(body: CorrectBody):
    p = get_pipeline()
    result = p.process_text(body.text)
    return result.to_dict()


@router.post("/v1/correct/batch")
def correct_batch(body: BatchBody):
    p = get_pipeline()
    results = p.process_batch(body.texts)
    return {"results": [r.to_dict() for r in results]}
