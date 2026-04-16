# douzone_postprocess

**이 폴더만** 다른 프로젝트(또는 서버)로 복사하면 된다. **`MEDICAL_STT_ROOT` 없이** 이 디렉터리가 곧 후처리 루트다.

## 포함되는 것

| 경로 | 설명 |
|------|------|
| `src/` | Rule-based, Medical confusion, KoGPT2, Context MLM(KLUE-RoBERTa) 등 **전체 후처리 코드** |
| `data/` | `medical_dict.txt` 등 |
| `*.py` | FastAPI 라우터·싱글톤 로더 |

`models/`에 로컬 체크포인트를 두고 `MEDICAL_STT_MODEL` 등으로 경로를 지정할 수 있다. 없으면 Hugging Face 허브 기본값을 쓴다.

## 의존성

상위 `medical-stt-postprocess/requirements.txt`와 동일하게 맞춘다( `torch`, `transformers`, `jamo`, `python-Levenshtein` … ). API용은:

```bash
pip install -r requirements.txt
```

## 다른 앱에 붙이기

프로젝트 루트가 `PYTHONPATH`에 잡히게 한 뒤:

```python
from douzone_postprocess import router as douzone_router

app.include_router(douzone_router, prefix="/stt-postprocess")
```

## 단독 실행

```bash
cd /path/to/parent-of-douzone_postprocess   # douzone_postprocess 의 부모
uvicorn douzone_postprocess.app_factory:create_app --factory --host 0.0.0.0 --port 8080
```

`MEDICAL_STT_ROOT`는 **선택**: 다른 경로의 후처리 트리를 쓰고 싶을 때만 설정한다.

## 이 저장소에서 개발할 때

`douzone_postprocess/` 안에 **`src/`가 들어 있으면** 그걸 우선 사용한다.  
상위 `medical-stt-postprocess`에만 `src/`가 있고 이 폴더에는 없으면(옛 구조), 자동으로 **부모 디렉터리**를 루트로 쓴다.

## 상위 저장소와 `src/` 동기화

상위에서 수정한 뒤 번들에 반영하려면:

```bash
# medical-stt-postprocess 루트에서
rsync -a --delete src/ douzone_postprocess/src/
rsync -a --delete data/ douzone_postprocess/data/
```
