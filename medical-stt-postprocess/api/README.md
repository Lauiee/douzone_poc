# api/ (레거시 진입점)

새 구조는 상위의 **`medical_stt_service/`** 를 참고한다. 다른 프로젝트에 붙일 때는 그 폴더를 복사하고 `MEDICAL_STT_ROOT`를 설정한다.

이 저장소 루트에서 예전처럼 띄우려면:

```bash
pip install -r requirements.txt -r api/requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8080
```
