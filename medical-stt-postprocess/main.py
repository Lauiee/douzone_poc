"""
의료 STT 후처리 POC - 메인 실행 파일

기본 KM-BERT 경로: 프로젝트 내 models/kmbert-kurias-vocab-hf (쿠리아스 가중치 변환본)
없으면 madatnlp/km-bert (임베딩 단계만, MLM 없음)

환경변수 KMBERT_MODEL 로 경로/허브 ID 지정 가능.
BertForMaskedLM refine 은 교정 품질 우선으로 기본 켜짐. 끄려면 --no-mlm 만 사용.
로컬 MLM 가중치가 없으면 해당 단계는 자동 스킵. KMBERT_MLM_MODEL 로 MLM 전용 폴더 지정 가능.
KoGPT2 PPL 단계는 기본 끔(실측 유효성 낮음·VRAM 부담). 켜려면 --use-kogpt2.
Context MLM 문맥 교정은 기본 켜짐. 끄려면 --no-kobert-context.
"""

import argparse
import json
import logging
from pathlib import Path

from src.pipeline import (
    MedicalSTTPipeline,
    load_input,
    format_comparison,
    default_kmbert_model_path,
    default_mlm_model_path,
)

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
    parser.add_argument("--no-bert", action="store_true", help="BERT 단계 비활성화")
    parser.add_argument("--model", type=str, default=None, help="KM-BERT 경로 또는 허브 ID (기본: 로컬 쿠리아스 변환본 또는 madatnlp)")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--max-edit-distance", type=int, default=2)
    parser.add_argument("--dict", type=str, help="의료 사전 경로")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--bert-trace", action="store_true")
    parser.add_argument(
        "--no-mlm",
        action="store_true",
        help="BertForMaskedLM refine 끄기 (속도·디버깅용; 기본은 켜짐)",
    )
    parser.add_argument(
        "--use-kogpt2",
        action="store_true",
        help="KoGPT2 PPL 단계 활성화 (기본: 끔)",
    )
    parser.add_argument(
        "--no-kobert-context",
        action="store_true",
        help="Context MLM(KLUE RoBERTa 등) 문맥 교정 끄기 (기본: 켜짐)",
    )
    parser.add_argument(
        "--mlm-model",
        type=str,
        default=None,
        help="MLM 전용 폴더 (기본: 로컬 쿠리아스 변환본 또는 KMBERT_MLM_MODEL)",
    )
    parser.add_argument("--mlm-anomaly-threshold", type=float, default=0.03)
    parser.add_argument("--mlm-top-k", type=int, default=30)
    parser.add_argument(
        "--mlm-min-cand-prob",
        type=float,
        default=0.12,
        help="MLM 후보 최소 확률 (기본 0.12, 과교정 완화)",
    )
    parser.add_argument(
        "--mlm-min-span-chars",
        type=int,
        default=3,
        metavar="N",
        help="MLM으로 바꿀 한국어 연속 구간 최소 길이(글자 수); 기본 3, 1이면 한 글자도 허용 (--mlm-legacy-dict-gate면 1)",
    )
    parser.add_argument("--mlm-window", type=int, default=36, help="이상 감지/예측 시 앞뒤 문맥 글자 수")
    parser.add_argument(
        "--no-mlm-dict-gate",
        action="store_true",
        help="MLM 의료 사전/화이트리스트 게이트 비활성화 (기본: 활성화)",
    )
    parser.add_argument(
        "--mlm-max-word-edit",
        type=int,
        default=1,
        metavar="N",
        help="사전·화이트리스트 없이 MLM 쓸 때 단어별 허용 편집거리 상한 (기본 1; --mlm-legacy-dict-gate면 무시)",
    )
    parser.add_argument("--kogpt2-model", type=str, default="skt/kogpt2-base-v2", help="KoGPT2 모델 경로/허브 ID")
    parser.add_argument("--kogpt2-top-k", type=int, default=40, help="KoGPT2 후보 개수")
    parser.add_argument("--kogpt2-max-word-edit", type=int, default=1, help="KoGPT2 교정 후보 최대 편집거리")
    parser.add_argument("--kogpt2-min-improve", type=float, default=0.06, help="KoGPT2 교정 최소 NLL 개선치")
    parser.add_argument("--kogpt2-min-span-chars", type=int, default=2, help="KoGPT2 교정 최소 단어 길이")
    parser.add_argument("--kobert-model", type=str, default="klue/roberta-large", help="Context MLM 모델 경로/허브 ID")
    parser.add_argument(
        "--kobert-anomaly-threshold",
        type=float,
        default=0.000002,
        help=(
            "원문 토큰 MLM 확률이 이 값 **미만**일 때만 교정 시도. "
            "값을 **낮출수록** 교정이 줄어듦(기본 0.000002, '이번→입원' 재현 최소치)."
        ),
    )
    parser.add_argument("--kobert-top-k", type=int, default=1500)
    parser.add_argument(
        "--kobert-min-cand-prob",
        type=float,
        default=0.05,
        help="후보 채택 최소 확률(기본 0.05)",
    )
    parser.add_argument("--kobert-max-word-edit", type=int, default=2)
    parser.add_argument("--kobert-min-span-chars", type=int, default=2)
    parser.add_argument("--kobert-window", type=int, default=36)

    args = parser.parse_args()
    setup_logging(args.verbose)

    log = logging.getLogger("main")
    log.info("의료 STT 후처리 POC 시작")

    model = args.model if args.model else default_kmbert_model_path()
    log.info(f"KM-BERT(BERT 단계): {model}")
    enable_mlm = not args.no_mlm
    mlm_path = default_mlm_model_path(args.mlm_model) if enable_mlm else None
    if enable_mlm:
        log.info(
            "MLM refine: 켜짐(기본) — %s",
            mlm_path or "가중치 없음(해당 단계 스킵)",
        )
    else:
        log.info("MLM refine: 끔 (--no-mlm)")
    log.info("KoGPT2 PPL: %s", f"켜짐 — {args.kogpt2_model}" if args.use_kogpt2 else "끔 (기본)")
    use_kobert = not args.no_kobert_context
    log.info(
        "Context MLM: %s",
        f"켜짐 — {args.kobert_model}" if use_kobert else "끔 (--no-kobert-context)",
    )

    if args.input:
        texts = load_input(args.input)
    else:
        texts = SAMPLE_TEXTS

    pipeline = MedicalSTTPipeline(
        dict_path=args.dict,
        model_name=model,
        device=args.device,
        confidence_threshold=args.confidence,
        max_edit_distance=args.max_edit_distance,
        enable_bert=not args.no_bert,
        bert_trace=args.bert_trace,
        enable_mlm=enable_mlm,
        mlm_model_path=args.mlm_model,
        mlm_anomaly_threshold=args.mlm_anomaly_threshold,
        mlm_top_k=args.mlm_top_k,
        mlm_min_candidate_prob=args.mlm_min_cand_prob,
        mlm_window_chars=args.mlm_window,
        mlm_legacy_dict_gate=not args.no_mlm_dict_gate,
        mlm_max_word_edit_distance=args.mlm_max_word_edit,
        mlm_min_span_chars=args.mlm_min_span_chars,
        enable_kogpt2=args.use_kogpt2,
        kogpt2_model_name=args.kogpt2_model,
        kogpt2_top_k=args.kogpt2_top_k,
        kogpt2_max_word_edit_distance=args.kogpt2_max_word_edit,
        kogpt2_min_improve=args.kogpt2_min_improve,
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
        log.info(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
