#!/usr/bin/env python3
"""MLM 교정 프로토타입 CLI (쿠리아스 변환 모델 등)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mlm_corrector import MlmCorrector  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--span", nargs=2, type=int)
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-prob", type=float, default=0.02)
    p.add_argument("--max-jamo", type=int, default=6)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    corr = MlmCorrector(str(args.model.resolve()), device=args.device)
    max_j = None if args.max_jamo < 0 else args.max_jamo
    if args.span:
        prop = corr.propose_correction(
            args.text, args.span[0], args.span[1],
            top_k=args.top_k, min_prob=args.min_prob, max_jamo_dist=max_j,
        )
        print(json.dumps({
            "original_span": prop.original,
            "chosen": prop.chosen,
            "confidence": prop.score,
            "topk": [{"surface": c.surface, "prob": c.prob, "token": c.token} for c in prop.topk],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.heuristic:
        t, ch = corr.correct_text(
            args.text, True,
            top_k=args.top_k, min_prob=args.min_prob, max_jamo_dist=max_j,
        )
        print(json.dumps({"corrected": t, "changes": ch}, ensure_ascii=False, indent=2))
        return 0
    print("--span 또는 --heuristic 필요", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
