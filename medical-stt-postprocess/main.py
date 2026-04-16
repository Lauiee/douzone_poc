"""
의료 STT 후처리 POC - 메인 실행 파일

현재 파이프라인:
- Rule-based
- Medical confusion
- KoGPT2 PPL (기본 끔)
- Context MLM(KLUE RoBERTa, 기본 켬)
"""

import argparse
import json
import logging
from pathlib import Path

from src.pipeline import MedicalSTTPipeline, format_comparison, load_input

SAMPLE_TEXTS = [
    (
        "오늘 어디가 불편하셔서 오셨어요? 일주일 정도 된 것 같은데 계단 오르내리면 "
        "가슴이 두근거리면서 쪼이는 통증이 있어요. 왼쪽 어깨랑 팔까지 아플 때도 있어요. "
        "오 분, 십 분 잠깐 쉬면 또 괜찮더라고요. 고혈압약은 오 년 전부터 먹고 있는데 "
        "먹다 안 먹다 해요. 혈압이 백육십에 백인데 안정 한번 하시고 다시 한번 재보고 "
        "심장 쪽에 문제 있는지 혈액 검사랑 심전도 한 번 해보시죠."
    ),
]


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="의료 STT 후처리 POC")
    parser.add_argument("-i", "--input", type=str, help="입력 파일 (txt/json)")
    parser.add_argument("-o", "--output", type=str, help="결과 JSON 경로")
    parser.add_argument("--dict", type=str, help="의료 사전 경로")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument(
        "--use-kogpt2",
        action="store_true",
        help="KoGPT2 PPL 단계 활성화 (기본: 끔)",
    )
    parser.add_argument("--kogpt2-model", type=str, default="skt/kogpt2-base-v2", help="KoGPT2 모델 경로/허브 ID")
    parser.add_argument("--kogpt2-top-k", type=int, default=40, help="KoGPT2 후보 개수")
    parser.add_argument(
        "--kogpt2-max-jamo-distance",
        type=int,
        default=2,
        help="의료 사전 자모 후보 최대 거리 (기본 2; kobert와 별개)",
    )
    parser.add_argument(
        "--kogpt2-roberta-max-jamo-distance",
        type=int,
        default=2,
        help="RoBERTa MLM 후보 자모 거리 상한 (기본 2; kobert와 별개)",
    )
    parser.add_argument(
        "--kogpt2-roberta-no-full-vocab",
        action="store_true",
        help="RoBERTa 후보를 softmax top-k만 사용 (기본: vocab 전체에서 MLM 하한+자모거리)",
    )
    parser.add_argument(
        "--kogpt2-roberta-vocab-mlm-floor",
        type=float,
        default=0.3,
        help="vocab 전체 스캔 시 MLM 확률 하한 (기본 0.3)",
    )
    parser.add_argument(
        "--kogpt2-roberta-full-vocab-max-cand",
        type=int,
        default=512,
        help="자모 필터 통과 후보 최대 개수 (기본 512)",
    )
    parser.add_argument("--kogpt2-min-improve", type=float, default=0.15, help="KoGPT2 교정 최소 NLL 개선치")
    parser.add_argument(
        "--kogpt2-min-improve-ratio",
        type=float,
        default=0.05,
        help="KoGPT2 교정 최소 상대 NLL 개선 (base 대비)",
    )
    parser.add_argument("--kogpt2-min-span-chars", type=int, default=2, help="KoGPT2 교정 최소 단어 길이")

    parser.add_argument(
        "--no-kobert-context",
        action="store_true",
        help="Context MLM(KLUE RoBERTa 등) 문맥 교정 끄기 (기본: 켜짐)",
    )
    parser.add_argument("--kobert-model", type=str, default="klue/roberta-large", help="Context MLM 모델 경로/허브 ID")
    parser.add_argument(
        "--kobert-anomaly-threshold",
        type=float,
        default=0.01,
        help=(
            "원문 토큰의 마스크 위치 MLM 확률이 이 값 미만이면 이상으로 보고 교정 후보 탐색. "
            "값을 올리면 교정 시도가 늘고, 내리면 줄어듦(기본 0.01)."
        ),
    )
    parser.add_argument("--kobert-top-k", type=int, default=50, help="Context MLM 후보 상위 k개(기본 50)")
    parser.add_argument(
        "--kobert-min-cand-prob",
        type=float,
        default=0.05,
        help="후보 채택 최소 확률(기본 0.05)",
    )
    parser.add_argument("--kobert-max-word-edit", type=int, default=2)
    parser.add_argument("--kobert-min-span-chars", type=int, default=2)
    parser.add_argument("--kobert-window", type=int, default=72)

    args = parser.parse_args()
    setup_logging(args.verbose)

    log = logging.getLogger("main")
    log.info("의료 STT 후처리 POC 시작")
    log.info("KoGPT2 PPL: %s", f"켜짐 — {args.kogpt2_model}" if args.use_kogpt2 else "끔 (기본)")
    use_kobert = not args.no_kobert_context
    log.info("Context MLM: %s", f"켜짐 — {args.kobert_model}" if use_kobert else "끔 (--no-kobert-context)")

    if args.input:
        texts = load_input(args.input)
    else:
        texts = SAMPLE_TEXTS

    pipeline = MedicalSTTPipeline(
        dict_path=args.dict,
        device=args.device,
        enable_kogpt2=args.use_kogpt2,
        kogpt2_model_name=args.kogpt2_model,
        kogpt2_top_k=args.kogpt2_top_k,
        kogpt2_max_jamo_distance=args.kogpt2_max_jamo_distance,
        kogpt2_roberta_max_jamo_distance=args.kogpt2_roberta_max_jamo_distance,
        kogpt2_roberta_full_vocab_jamo=not args.kogpt2_roberta_no_full_vocab,
        kogpt2_roberta_vocab_mlm_floor=args.kogpt2_roberta_vocab_mlm_floor,
        kogpt2_roberta_full_vocab_max_cand=args.kogpt2_roberta_full_vocab_max_cand,
        kogpt2_min_improve=args.kogpt2_min_improve,
        kogpt2_min_improve_ratio=args.kogpt2_min_improve_ratio,
        kogpt2_min_span_chars=args.kogpt2_min_span_chars,
        enable_kobert_context=use_kobert,
        kobert_model_name=args.kobert_model,
        kobert_anomaly_threshold=args.kobert_anomaly_threshold,
        kobert_top_k=args.kobert_top_k,
        kobert_min_candidate_prob=args.kobert_min_cand_prob,
        kobert_max_word_edit_distance=args.kobert_max_word_edit,
        kobert_min_span_chars=args.kobert_min_span_chars,
        kobert_window_chars=args.kobert_window,
    )

    results = pipeline.process_batch(texts)

    print("\n" + "=" * 70)
    print(" 의료 STT 후처리 결과")
    print("=" * 70)
    for i, result in enumerate(results):
        print(f"\n[샘플 {i+1}]")
        print(format_comparison(result))

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("결과 저장: %s", out)


if __name__ == "__main__":
    main()
