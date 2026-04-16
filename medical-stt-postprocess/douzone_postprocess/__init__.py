"""
다른 프로젝트에 이 폴더만 복사해 넣어 사용한다.

- 환경변수 MEDICAL_STT_ROOT 에 medical-stt-postprocess 루트 경로 지정 (필수: 복사만 한 경우)
- router 를 기존 FastAPI 앱에 include_router 하거나, create_app() 으로 단독 서비스 실행
"""

from douzone_postprocess.app_factory import create_app
from douzone_postprocess.routes import router

__all__ = ["create_app", "router"]
