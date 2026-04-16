#!/usr/bin/env python3
"""한 문장(또는 파일)에 대해 rule_based → medical_confusion → kogpt2_ppl → kobert_context 단계별 교정을 출력."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline import MedicalSTTPipeline  # noqa: E402


def _print_changes(stage: str, data: dict) -> None:
    if data.get("skipped"):
        print(f"  [{stage}] 비활성: {data.get('reason', '')}")
        return
    if data.get("error"):
        print(f"  [{stage}] 오류: {data['error']}")
        return
    ch = data.get("changes") or []
    if not ch:
        print(f"  [{stage}] 교정 없음")
        return
    print(f"  [{stage}] {len(ch)}건")
    for c in ch:
        o = c.get("original", "")
        n = c.get("corrected", "")
        extra = []
        if "improve" in c:
            extra.append(f"NLL개선={c['improve']:.4f}")
        if "confidence" in c:
            extra.append(f"conf={c['confidence']}")
        if "selected_prob" in c:
            extra.append(f"p={c['selected_prob']}")
        tail = f" ({', '.join(extra)})" if extra else ""
        print(f"    {o!r} → {n!r}{tail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", type=str, help="처리할 한 문장")
    ap.add_argument("--file", type=Path, help="utf-8 텍스트 파일")
    args = ap.parse_args()

    if bool(args.text) == bool(args.file):
        raise SystemExit("--text 또는 --file 중 하나만 지정하세요.")

    if args.file:
        raw = args.file.read_text(encoding="utf-8").strip()
        if raw.startswith("[") or raw.startswith("{"):
            try:
                data = json.loads(raw)
                if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
                    texts = [str(x["text"]) for x in data]
                elif isinstance(data, dict) and "text" in data:
                    texts = [str(data["text"])]
                else:
                    texts = [raw]
            except json.JSONDecodeError:
                texts = [raw]
        else:
            texts = [raw]
    else:
        texts = [args.text]

    p = MedicalSTTPipeline()
    for i, text in enumerate(texts, 1):
        if len(texts) > 1:
            print(f"\n######## 블록 {i}/{len(texts)} ########")
        r = p.process_text(text)
        print("\n[원문]\n", text[:500] + ("…" if len(text) > 500 else ""), sep="")
        print("\n[단계별 교정]")
        for name in ("rule_based", "medical_confusion", "kogpt2_ppl", "kobert_context"):
            _print_changes(name, r.stages.get(name, {}))
        print("\n[최종]\n", r.corrected[:500] + ("…" if len(r.corrected) > 500 else ""), sep="")


if __name__ == "__main__":
    main()
