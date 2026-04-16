"""KoGPT2 단계 효과 요약: Context MLM을 끈 상태에서 KoGPT2만 켜고 끄고 비교."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.pipeline import MedicalSTTPipeline, load_input  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="json 입력 (load_input 형식)")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--skip-kogpt2",
        action="store_true",
        help="KoGPT2 로드/실행 생략(의료 혼동까지 동일 기준선)",
    )
    args = ap.parse_args()

    texts = load_input(args.input)
    pipeline = MedicalSTTPipeline(
        device=args.device,
        enable_kogpt2=not args.skip_kogpt2,
        enable_kobert_context=False,
    )

    rows = []
    for i, text in enumerate(texts):
        r = pipeline.process_text(text)
        mc = r.stages.get("medical_confusion", {})
        mc_out = mc.get("output", r.corrected)
        kg = r.stages.get("kogpt2_ppl", {})
        kg_changes = kg.get("changes") or []
        delta = r.corrected != mc_out
        rows.append(
            {
                "index": i,
                "kogpt2_skipped": kg.get("skipped"),
                "text_changed_after_medical_confusion": delta,
                "kogpt2_change_count": len(kg_changes),
                "after_confusion_preview": mc_out[:120] + ("..." if len(mc_out) > 120 else ""),
                "final_preview": r.corrected[:120] + ("..." if len(r.corrected) > 120 else ""),
            }
        )

    any_delta = any(x["text_changed_after_medical_confusion"] for x in rows)
    total_kg = sum(x["kogpt2_change_count"] for x in rows)

    print(
        json.dumps(
            {
                "samples": len(texts),
                "any_kogpt2_text_delta": any_delta,
                "total_kogpt2_changes": total_kg,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
