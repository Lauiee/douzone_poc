"""KoGPT2 단계 유효성: MLM 직후 문자열 vs KoGPT2 이후( Context MLM 끔 )."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.pipeline import MedicalSTTPipeline, load_input, default_kmbert_model_path, default_mlm_model_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="json 입력 (load_input 형식)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-mlm", action="store_true")
    ap.add_argument(
        "--skip-kogpt2",
        action="store_true",
        help="이 스크립트에서만 KoGPT2 로드/실행 생략(MLM만 비교)",
    )
    args = ap.parse_args()

    texts = load_input(args.input)
    pipeline = MedicalSTTPipeline(
        model_name=default_kmbert_model_path(),
        device=args.device,
        enable_mlm=not args.no_mlm,
        mlm_model_path=None,
        enable_kogpt2=not args.skip_kogpt2,
        enable_kobert_context=False,
    )

    rows = []
    for i, text in enumerate(texts):
        r = pipeline.process_text(text)
        mlm = r.stages.get("mlm_refine", {})
        mlm_out = mlm.get("output", r.corrected)
        kg = r.stages.get("kogpt2_ppl", {})
        kg_changes = kg.get("changes") or []
        delta = r.corrected != mlm_out
        rows.append(
            {
                "index": i,
                "mlm_skipped": mlm.get("skipped"),
                "kogpt2_skipped": kg.get("skipped"),
                "text_changed_after_mlm": delta,
                "kogpt2_change_count": len(kg_changes),
                "mlm_output_preview": mlm_out[:120] + ("..." if len(mlm_out) > 120 else ""),
                "final_preview": r.corrected[:120] + ("..." if len(r.corrected) > 120 else ""),
            }
        )

    any_delta = any(x["text_changed_after_mlm"] for x in rows)
    total_kg = sum(x["kogpt2_change_count"] for x in rows)

    print(json.dumps({"samples": len(texts), "any_kogpt2_text_delta": any_delta, "total_kogpt2_changes": total_kg, "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
