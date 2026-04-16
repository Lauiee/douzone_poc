# douzone_poc

의료 STT 후처리(`medical-stt-postprocess`) 실험·PoC 코드입니다.

## 대용량 모델 파일

Transformers가 Hugging Face에서 받는 가중치(`klue/roberta-large`, `skt/kogpt2-base-v2` 등)는 기본적으로 캐시에 내려받는다. 별도로 저장소에 넣어 둔 대용량 바이너리가 있다면 로컬 경로·환경변수로 지정하면 된다.

## 실행 예

```bash
cd medical-stt-postprocess
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

자세한 파이프라인은 `medical-stt-postprocess/PIPELINE.md`를 참고하세요.
