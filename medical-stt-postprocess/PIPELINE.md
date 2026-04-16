# 의료 STT 후처리 파이프라인 문서

이 문서는 현재 코드(`main.py`, `src/pipeline.py`) 기준의 실제 동작 흐름을 설명한다.

## 전체 흐름

1. 입력 로드 (`txt` 또는 `json`)
2. Rule-based 정규화
3. KM-BERT MLM 토큰 확신도 기반 교정
4. (선택) KoGPT2 PPL — `--use-kogpt2`일 때만
5. (선택) Context MLM — `--use-kobert-context`일 때만
6. 결과 JSON 저장

```text
입력
  -> rule_based
  -> mlm_refine (KM-BERT MLM)
  -> kogpt2_ppl (KoGPT2, 기본 스킵)
  -> kobert_context (KLUE RoBERTa 등, 기본 스킵)
  -> 출력
```

---

## Stage 0. 입력

- 파일 형식:
  - `*.txt`: 텍스트 전체를 1개 샘플로 처리
  - `*.json`:
    - `[{ "text": "..." }, ...]`
    - 또는 `{ "text": "..." }`
- 로더 함수: `load_input()`

---

## Stage 1. Rule-based (`rule_based`)

핵심 역할은 숫자/단위 정규화다.

- 예시:
  - `오 분` -> `5분`
  - `백육십에 백` -> `160/100`
  - `한 번` -> `1번`
- 구현 파일: `src/rule_based.py`
- 엔트리: `apply_rule_based()`

출력은 `stages["rule_based"]`에 기록된다.

---

## Stage 2. KM-BERT MLM (`mlm_refine`)

토큰 자체가 어색한 경우를 교정하는 단계다.

동작 개요:

1. 한글 토큰(span) 순회
2. 각 토큰의 MLM score로 이상 여부 판단
3. 의심 토큰을 `[MASK]` 문맥으로 top-k 후보 생성
4. 후보 필터 통과 시 치환

주요 필터:

- 최소 후보 확률
- 조사/어미 일치
- 편집거리 제한
- 최소 토큰 길이
- (옵션) legacy dictionary gate

구현 파일:

- `src/mlm_corrector.py`
- 파이프라인 호출: `correct_by_token_confidence()`

---

## Stage 3. KoGPT2 PPL (`kogpt2_ppl`)

토큰은 정상처럼 보여도 문맥이 어색한 경우를 보정하는 단계다.

현재 구현 개요:

1. span window(기본 2어절) 단위 탐색
2. 각 어절 후보를 KM-BERT MLM top-k에서 생성
3. 제한적 삽입 후보 포함
4. 후보 조합별 문장 NLL 계산
5. 절대 개선 + 상대 개선율 모두 만족 시 반영

성능 보호 장치:

- 조합 상한(`max_combinations`)
- per-word 후보 상한
- NLL 캐시

구현 파일:

- `src/kogpt2_corrector.py`
- 파이프라인 호출: `correct_text()`

---

## 현재 비활성 단계

아래 단계는 현재 파이프라인에서 실행되지 않고 `skipped`로 남는다.

- `jamo_correction`
- `bert_mlm` (임베딩 기반 교정)

이 항목들은 호환을 위해 stage 키는 유지한다.

---

## CLI 옵션 요약

실행 파일: `main.py`

### 공통

- `-i, --input`: 입력 파일 경로
- `-o, --output`: 출력 JSON 경로
- `--device {cpu,cuda}`
- `-v, --verbose`

### KM-BERT MLM

- `--no-mlm`
- `--mlm-model`
- `--mlm-anomaly-threshold`
- `--mlm-top-k`
- `--mlm-min-cand-prob`
- `--mlm-min-span-chars`
- `--mlm-max-word-edit`
- `--mlm-window`
- `--mlm-legacy-dict-gate`

### KoGPT2 PPL

- `--use-kogpt2` (기본 끔)
- `--kogpt2-model`
- `--kogpt2-top-k`
- `--kogpt2-max-word-edit`
- `--kogpt2-min-improve`
- `--kogpt2-min-span-chars`

---

## 결과 JSON 구조

출력은 샘플 배열이며 각 항목은 아래 구조를 가진다.

- `original`: 원문
- `corrected`: 최종 결과
- `stages`:
  - `rule_based`
  - `jamo_correction` (보통 skipped)
  - `bert_mlm` (보통 skipped)
  - `mlm_refine`
  - `kogpt2_ppl`

각 stage는 다음 필드를 가질 수 있다.

- `output`: 해당 단계 출력 문자열
- `changes`: 교정 리스트
- `skipped`: 스킵 여부
- `reason`: 스킵 사유
- `error`: 예외 메시지

---

## GPU 메모리 상한 (다른 프로세스와 VRAM 나누기)

`MedicalSTTPipeline` 초기화 시(첫 GPU 텐서보다 앞서) PyTorch 프로세스당 VRAM 상한을 둘 수 있다.

- **`CUDA_MEMORY_LIMIT_GB`**: 이 프로세스가 쓰고 싶은 대략적인 GiB. 예: 16GB 카드에서 최대 8GB만 쓰려면 `8` 로 두면, 카드 총량 대비 비율이 자동 계산된다.
- **`CUDA_MEMORY_FRACTION`**: 카드 총 VRAM 대비 비율 (0 초과 1 이하). 예: `0.5` 는 약 절반. **`CUDA_MEMORY_LIMIT_GB`가 설정되어 있으면 그쪽이 우선**한다.

내부적으로 `torch.cuda.set_per_process_memory_fraction`을 사용한다. 상한을 넘기려는 할당이 있으면 OOM이 날 수 있으므로, 모델을 모두 올리기 어렵다면 단계 비활성화(기본으로 KoGPT2는 끔)나 더 작은 모델을 검토한다.

---

## 알려진 한계

- 문맥 오류는 후보 생성 품질에 크게 의존한다.
- span/삽입 후보를 늘리면 품질이 오르지만 계산량이 크게 증가한다.
- 과교정 방지를 위해 게이트가 강하면 교정 누락이 생길 수 있다.
