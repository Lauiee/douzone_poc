#!/usr/bin/env python3
"""RoBERTa MLM: 특정 어절 위치에서 top-k 토큰·확률 진단 (판결→한결, 타신 등 원인 분리용)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_KOREAN = re.compile(r"^[가-힣]+$")


def _strip_wp(tok: str) -> str:
    return tok[2:] if tok.startswith("##") else tok


def _mask_index(input_ids: torch.Tensor, mask_id: int) -> int:
    pos = (input_ids == mask_id).nonzero(as_tuple=False)
    if pos.numel() == 0:
        raise RuntimeError("[MASK] 토큰 없음")
    return int(pos[0, -1].item())


def check_topk(
    text: str,
    needle: str,
    model_name: str = "klue/roberta-large",
    top_k: int = 50,
    device: str | None = None,
) -> tuple[list[tuple[str, float]], dict[str, tuple[int | None, float | None]]]:
    """needle 첫 등장을 [MASK]로 바꾼 뒤 MLM top-k 및 관심 토큰 순위.

    반환: (top_k 리스트, interest 이름 -> (순위 또는 None, 확률))
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(dev)

    if needle not in text:
        raise ValueError(f"문장에 '{needle}' 이 없습니다: {text!r}")

    start = text.index(needle)
    end = start + len(needle)
    masked = text[:start] + tok.mask_token + text[end:]
    enc = tok(masked, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(dev)

    mid = tok.mask_token_id
    midx = _mask_index(enc["input_ids"][0], mid)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[0, midx]

    probs = F.softmax(logits.float(), dim=-1)
    k = min(top_k, probs.shape[0])
    top_p, top_i = torch.topk(probs, k=k)

    results: list[tuple[str, float]] = []
    for i in range(k):
        tid = int(top_i[i].item())
        p = float(top_p[i].item())
        raw = tok.convert_ids_to_tokens(tid) or ""
        surface = _strip_wp(raw)
        if surface and _KOREAN.fullmatch(surface):
            results.append((surface, p))
        else:
            results.append((raw or f"<id={tid}>", p))

    interest: dict[str, tuple[int | None, float | None]] = {}
    for name in ("한결", "탓인", "타신", "타"):
        ids = tok.encode(name, add_special_tokens=False)
        if len(ids) != 1:
            interest[name] = (None, None)
            continue
        tid = ids[0]
        p = float(probs[tid].item())
        r = int((probs > probs[tid]).sum().item()) + 1
        interest[name] = (r, p)

    return results, interest


def main() -> None:
    ap = argparse.ArgumentParser(description="RoBERTa MLM top-k 진단")
    ap.add_argument("--text", type=str, default="주사 맞고 가시면 판결 가벼우실 겁니다")
    ap.add_argument("--needle", type=str, default="판결")
    ap.add_argument("--model", type=str, default="klue/roberta-large")
    ap.add_argument("-k", "--top-k", type=int, default=50)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    needle_ids = tok.encode(args.needle, add_special_tokens=False)
    needle_toks = tok.convert_ids_to_tokens(needle_ids)

    print("=== needle 토큰화 (서브워드 개수) ===")
    print(f"  needle={args.needle!r} -> ids={needle_ids} tokens={needle_toks}")
    if len(needle_ids) > 1:
        print(
            "  주의: needle이 서브워드 2개 이상이면, 문자열 전체를 한 [MASK]로 바꿔도 "
            "모델은 마스크 **한 칸**만 예측합니다. kobert_context의 "
            "top-k에 표면이 needle 문자열 전체와 일치하는 항목이 나오기 어렵고, "
            "original_prob==0으로 스킵되는 구조일 수 있습니다."
        )

    print()
    print("=== RoBERTa top-%d (마스크 1칸 기준) ===" % args.top_k)
    results, interest = check_topk(
        args.text, args.needle, model_name=args.model, top_k=args.top_k
    )
    for t, prob in results:
        print(f"  {t}: {prob:.6f}")

    tokens = [t for t, _ in results]
    print()
    print(f"top-{args.top_k} 표면에 '한결' 포함: {'한결' in tokens}")
    for label in ("한결", "탓인", "타신", "타"):
        r, p = interest.get(label, (None, None))
        ids = tok.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            print(f"{label}: 단일 토큰 아님 ids={ids} {tok.convert_ids_to_tokens(ids)} -> 단일 [MASK]와 직접 비교 불가")
        elif p is not None:
            print(f"{label}: 순위={r}, prob={p:.6f} (해당 vocab id 한 칸 기준)")
        else:
            print(f"{label}: 순위={r}, prob={p}")

    # 타신: 전체 문자열이 top-k에 있을 수 없음 (2 서브워드)
    if args.needle == "타신":
        print()
        print("=== 타신 케이스 요약 ===")
        print(
            "  original_prob(kobert): top-k에서 surface=='타신' 조회 -> "
            "통상 **불가**(표면이 한 토큰이 아님) -> **0.0** -> original_prob<=0 게이트에서 스킵."
        )
        rt, pt = interest.get("타", (None, None))
        if pt is not None:
            print(
                f"  첫 서브워드 '타'의 MLM prob≈{pt:.6e}, 순위≈{rt} "
                "(이 [MASK] 위치는 원래 '타' 한 칸에 대응)"
            )


if __name__ == "__main__":
    main()
