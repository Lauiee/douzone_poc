"""DAPT 학습용 원본 말뭉치 생성 스크립트.

medsub 디렉터리 하위의 모든 JSON 파일에서 전사정보.LabelText 를 추출하고,
QualityStatus == "Good" 인 발화 중 품질 조건을 만족하는 텍스트만
train / eval 파일로 분할 저장한다.

실행 예:
    python scripts/prepare_dapt_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT_ROOT / "training_data" / "medsub"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"

HANGUL_RE = re.compile(r"[가-힣]")
# 전사 노이즈: (웃음), (기침), (헛기침), [unintelligible], <noise> 등
NOISE_RE = re.compile(r"[\(\[\<][^\(\)\[\]\<\>]{0,40}?[\)\]\>]")
WS_RE = re.compile(r"\s+")

MIN_LEN = 5
EVAL_RATIO = 0.02
EVAL_MIN = 1000
RANDOM_SEED = 42


def clean_text(text: str) -> str:
    """전사 노이즈 마커 제거 및 공백 정리."""
    # 중첩 노이즈 마커를 위해 반복 적용
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = NOISE_RE.sub(" ", cur)
    return WS_RE.sub(" ", cur).strip()


def is_valid(text: str) -> bool:
    if len(text) < MIN_LEN:
        return False
    if not HANGUL_RE.search(text):
        return False
    return True


def process_file(path: str) -> str | None:
    """단일 JSON 파일에서 유효한 발화 텍스트를 추출한다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    quality = (data.get("기타정보") or {}).get("QualityStatus")
    if quality != "Good":
        return None

    label = (data.get("전사정보") or {}).get("LabelText")
    if not isinstance(label, str):
        return None

    cleaned = clean_text(label)
    if not is_valid(cleaned):
        return None
    return cleaned


def iter_json_files(src: Path):
    yield from (str(p) for p in src.rglob("*.json"))


def chunked(iterable, n):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def process_chunk(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        text = process_file(p)
        if text is not None:
            out.append(text)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    src: Path = args.src
    out_dir: Path = args.out_dir
    if not src.exists():
        print(f"[ERROR] 원본 경로가 없음: {src}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "dapt_raw_train.txt"
    eval_path = out_dir / "dapt_raw_eval.txt"

    print(f"[INFO] 스캔 경로: {src}")
    print("[INFO] 파일 목록 수집 중...")
    files = list(iter_json_files(src))
    total_files = len(files)
    print(f"[INFO] 총 JSON 파일: {total_files:,}개")
    if total_files == 0:
        print("[ERROR] JSON 파일을 찾지 못함", file=sys.stderr)
        return 1

    collected: list[str] = []
    seen: set[str] = set()
    processed = 0
    duplicates = 0

    chunks = list(chunked(files, args.chunk_size))
    print(f"[INFO] 청크 수: {len(chunks)} (워커 {args.workers}개)")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_chunk, c) for c in chunks]
        for i, fut in enumerate(as_completed(futures), 1):
            texts = fut.result()
            for t in texts:
                if t in seen:
                    duplicates += 1
                    continue
                seen.add(t)
                collected.append(t)
            processed += 1
            if processed % 50 == 0 or processed == len(futures):
                print(
                    f"[INFO] 청크 {processed}/{len(futures)} | "
                    f"유효 발화 {len(collected):,} | 중복 {duplicates:,}",
                    flush=True,
                )

    n_valid = len(collected)
    print(f"[INFO] 필터 통과 발화: {n_valid:,} / {total_files:,}")
    if n_valid == 0:
        print("[ERROR] 유효 발화가 없음", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(collected)

    eval_n = max(EVAL_MIN, int(round(n_valid * EVAL_RATIO)))
    eval_n = min(eval_n, n_valid - 1)
    train_texts = collected[eval_n:]
    eval_texts = collected[:eval_n]

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))
        f.write("\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write("\n".join(eval_texts))
        f.write("\n")

    print(f"[DONE] train {len(train_texts):,} -> {train_path}")
    print(f"[DONE] eval  {len(eval_texts):,} -> {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
