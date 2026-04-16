"""
호환용 얇은 래퍼. 권장: `medical_stt_service` 패키지를 직접 사용.

  uvicorn medical_stt_service.app_factory:create_app --factory --host 0.0.0.0 --port 8080
"""

from medical_stt_service.app_factory import create_app

app = create_app()
