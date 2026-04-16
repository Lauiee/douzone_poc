# douzone_poc

의료 STT 후처리(`medical-stt-postprocess`) 실험·PoC 코드입니다.

## 대용량 모델 파일

Git에는 용량 제한으로 다음을 포함하지 않습니다. 로컬에 두거나 변환 스크립트로 준비하세요.

- `medical-stt-postprocess/third_party/pytorch_model.bin`
- `medical-stt-postprocess/third_party/kmbert_vocab.tar`
- `medical-stt-postprocess/models/kmbert-kurias-vocab-hf/model.safetensors`

토크나이저·설정(`config.json`, `tokenizer.json` 등)은 저장소에 포함됩니다.

## 실행 예

```bash
cd medical-stt-postprocess
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

자세한 파이프라인은 `medical-stt-postprocess/PIPELINE.md`를 참고하세요.
