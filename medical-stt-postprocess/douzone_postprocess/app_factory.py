"""단독 실행용 FastAPI 앱 생성."""

from __future__ import annotations

from fastapi import FastAPI

from douzone_postprocess.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Medical STT Postprocess",
        description="의료 STT 후처리 파이프라인 (다른 프로젝트에서 복사한 douzone_postprocess)",
        version="1.0.0",
    )
    app.include_router(router)
    return app
