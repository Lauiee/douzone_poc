#!/usr/bin/env python3
"""
KM-BERT MLM 스모크 테스트 — Hugging Face 허브 ID 또는 로컬 변환 모델 경로에서
[MASK] 위치의 top-k 토큰을 출력한다.

KU-RIAS 원본(model.bin)을 transformers 형식으로 변환한 뒤 --model 에 로컬 경로를
넘기면 HF 체크포인트(madatnlp/km-bert)와 출력을 나란히 비교할 수 있다.

사용 예:
  python scripts/mlm_smoke.py --model madatnlp/km-bert \\
    --text "혈압이 [MASK] 입니다."

  python scripts/mlm_smoke.py --compare madatnlp/km-bert ./converted_kmbert \\
    --text "반동성 [MASK]이 뚜렷합니다."
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_TEXTS = [
    "혈압이 [MASK] 입니다.",
    "반동성 [MASK]이 뚜렷합니다.",
    "맹장염 즉 [MASK]입니다.",
]


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_mlm(model_id_or_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_id_or_path)
    model.eval()
    model.to(device)
    return tok, model


def mlm_topk_loaded(
    tok,
    model,
    text: str,
    top_k: int,
    device: torch.device,
) -> tuple[list[tuple[str, float]], str]:
    """이미 로드된 tokenizer/model로 top-k 반환. (오류 시 빈 리스트 + 메시지)"""
    if tok.mask_token is None:
        return [], "토크나이저에 mask_token 이 없습니다."

    line = text
    if "[MASK]" not in line and tok.mask_token not in line:
        line = line.replace("[MASK]", tok.mask_token)

    enc = tok(line, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    mask_id = tok.mask_token_id
    if mask_id is None:
        return [], "mask_token_id 가 None 입니다."

    positions = (input_ids == mask_id).nonzero(as_tuple=True)
    if input_ids.shape[0] != 1 or len(positions[0]) != 1:
        return [], f"마스크 토큰은 정확히 1개여야 합니다. (현재: {len(positions[0])}개)"

    batch_i = int(positions[0][0])
    pos = int(positions[1][0])
    if batch_i != 0:
        return [], "배치 1개만 지원합니다."

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, pos]

    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k=min(top_k, logits.shape[0]))

    results: list[tuple[str, float]] = []
    for i in range(vals.shape[0]):
        tid = int(idx[i].item())
        piece = tok.convert_ids_to_tokens(tid)
        results.append((piece, float(vals[i].item())))

    return results, ""


def run_one(label: str, model_id: str, texts: list[str], top_k: int, device: torch.device) -> None:
    print(f"\n=== 모델: {label} ({model_id}) ===")
    try:
        tok, model = _load_mlm(model_id, device)
    except Exception as e:
        print(f"로드 실패: {e}")
        return

    for t in texts:
        print(f"\n문장: {t}")
        rows, err = mlm_topk_loaded(tok, model, t, top_k, device)
        if err:
            print(f"  오류: {err}")
            continue
        for rank, (tok_str, p) in enumerate(rows, start=1):
            print(f"  {rank:2d}. {tok_str!r}  p={p:.6f}")


def main() -> int:
    p = argparse.ArgumentParser(description="KM-BERT MLM 스모크 (top-k)")
    p.add_argument("--model", default=None, help="단일 모델: 허브 ID 또는 로컬 경로")
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="두 모델을 같은 문장으로 비교 (허브 ID 또는 로컬 경로)",
    )
    p.add_argument("--text", action="append", help="테스트 문장 ([MASK] 포함). 여러 번 지정 가능")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", default=None, help="cuda / cpu / cuda:0 등")
    args = p.parse_args()

    if not args.model and not args.compare:
        print("--model 또는 --compare 가 필요합니다.", file=sys.stderr)
        return 2

    texts = args.text if args.text else DEFAULT_TEXTS
    device = _device(args.device)

    if args.compare:
        a, b = args.compare
        run_one("A", a, texts, args.top_k, device)
        run_one("B", b, texts, args.top_k, device)
    else:
        run_one("single", args.model, texts, args.top_k, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
