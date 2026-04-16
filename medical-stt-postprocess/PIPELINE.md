# 의료 STT 후처리 파이프라인 문서

`main.py`, `src/pipeline.py` 기준 동작이다.

## 전체 흐름

1. 입력 로드 (`txt` 또는 `json`)
2. Rule-based 정규화
3. Medical confusion(고정 치환 표)
4. (선택) KoGPT2 PPL — `main.py`에서는 `--use-kogpt2`일 때만
5. (선택) Context MLM(KLUE-RoBERTa) — 기본 켬, `--no-kobert-context`로 끔
6. 결과 JSON 저장

```text
입력
  -> rule_based
  -> medical_confusion
  -> kogpt2_ppl (KoGPT2, CLI 기본 끔 / 파이프라인 클래스 기본값은 켜짐)
  -> kobert_context (KLUE-RoBERTa)
  -> 출력
```

---

## Stage 0. 입력

- `*.txt`: 파일 전체를 1개 샘플
- `*.json`: `[{ "text": "..." }, ...]` 또는 `{ "text": "..." }`
- 로더: `load_input()`

---

## Stage 1. Rule-based (`rule_based`)

숫자·단위·약어 등 규칙 정규화. `src/rule_based.py`의 `apply_rule_based()`.

---

## Stage 2. Medical confusion (`medical_confusion`)

의료 용어 오인식 표 기반 고정 치환. `src/medical_confusion.py`.

---

## Stage 3. KoGPT2 PPL (`kogpt2_ppl`)

의료 사전 자모 유사 후보와 문장 NLL 개선으로 스팬 교정. 선택적으로 앞 단계에서 로드한 KLUE-RoBERTa를 후보 보강에 재사용할 수 있다. `src/kogpt2_corrector.py`.

---

## Stage 4. Context MLM (`kobert_context`)

KLUE-RoBERTa 마스크 언어모델로 문맥 부적합 토큰 교정. `src/kobert_context_corrector.py`.

---

## CLI (`main.py`)

### 공통

- `-i, --input`, `-o, --output`, `--dict`, `--device`, `-v`

### KoGPT2

- `--use-kogpt2` (기본 끔)
- `--kogpt2-model`, `--kogpt2-top-k`, `--kogpt2-max-jamo-distance`, `--kogpt2-min-improve`, `--kogpt2-min-span-chars`

### Context MLM

- `--no-kobert-context` (기본 켬)
- `--kobert-model`, `--kobert-anomaly-threshold`, `--kobert-top-k`, `--kobert-min-cand-prob`, `--kobert-max-word-edit`, `--kobert-min-span-chars`, `--kobert-window`

---

## 결과 JSON (`PipelineResult.to_dict()`)

- `original`, `corrected`
- `stages`: `rule_based`, `medical_confusion`, `kogpt2_ppl`, `kobert_context`
  - 각 stage는 `output`, `changes`, 필요 시 `skipped` / `reason` / `error` 포함

---

## 알려진 한계

- 문맥·후보 품질은 모델·사전·임계값 설정에 민감하다.
- 과교정을 막는 게이트를 강하게 두면 누락이 늘 수 있다.
