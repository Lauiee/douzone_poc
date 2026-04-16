"""
전체 후처리 파이프라인 (KM-BERT 제거, KLUE-RoBERTa 단일화)
1. Rule-based: STT 구문 + 숫자/단위/약어
2. Medical confusion: 의료 용어 오인식 표만 고정 치환
3. KoGPT2 PPL Span Correction (선택): 자모 유사도 후보 + GPT2 NLL 검증
4. Context MLM (선택): KLUE-RoBERTa 문맥 기반 교정
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.kobert_context_corrector import KoBERTContextCorrector
from src.kogpt2_corrector import KoGPT2Corrector
from src.medical_confusion import DEFAULT_MEDICAL_CONFUSION_SET, apply_confusion_replacements
from src.rule_based import apply_rule_based

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    original: str
    corrected: str
    stages: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "corrected": self.corrected,
            "stages": self.stages,
        }


class MedicalSTTPipeline:
    def __init__(
        self,
        dict_path: str | Path | None = None,
        device: str | None = None,
        # KoGPT2
        enable_kogpt2: bool = True,
        kogpt2_model_name: str = "skt/kogpt2-base-v2",
        kogpt2_top_k: int = 40,
        kogpt2_max_jamo_distance: int = 2,
        kogpt2_min_improve: float = 0.04,
        kogpt2_min_span_chars: int = 2,
        # Context MLM (KLUE-RoBERTa)
        enable_kobert_context: bool = True,
        kobert_model_name: str = "klue/roberta-large",
        kobert_anomaly_threshold: float = 0.01,
        kobert_top_k: int = 50,
        kobert_min_candidate_prob: float = 0.05,
        kobert_max_word_edit_distance: int = 2,
        kobert_min_span_chars: int = 2,
        kobert_window_chars: int = 72,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 의료 사전 로드 (jamo_corrector 없이 직접)
        from src.jamo_corrector import load_medical_dict
        self.medical_terms = load_medical_dict(dict_path)

        self.enable_kogpt2 = enable_kogpt2
        self.kogpt2_model_name = kogpt2_model_name
        self.kogpt2_top_k = kogpt2_top_k
        self.kogpt2_max_jamo_distance = kogpt2_max_jamo_distance
        self.kogpt2_min_improve = kogpt2_min_improve
        self.kogpt2_min_span_chars = kogpt2_min_span_chars

        self.enable_kobert_context = enable_kobert_context
        self.kobert_model_name = kobert_model_name
        self.kobert_anomaly_threshold = kobert_anomaly_threshold
        self.kobert_top_k = kobert_top_k
        self.kobert_min_candidate_prob = kobert_min_candidate_prob
        self.kobert_max_word_edit_distance = kobert_max_word_edit_distance
        self.kobert_min_span_chars = kobert_min_span_chars
        self.kobert_window_chars = kobert_window_chars

        self._kogpt2_corrector = None
        self._kobert_context_corrector = None

        logger.info("=== 의료 STT 후처리 파이프라인 초기화 ===")
        logger.info("의료 사전: %d개 용어", len(self.medical_terms))

        if enable_kogpt2:
            logger.info("KoGPT2 PPL 준비: %s", kogpt2_model_name)
        else:
            logger.info("KoGPT2 PPL 비활성화")

        if enable_kobert_context:
            logger.info("Context MLM 준비: %s", kobert_model_name)
        else:
            logger.info("Context MLM 비활성화")

        logger.info("파이프라인 초기화 완료")

    def _get_kogpt2_corrector(self) -> KoGPT2Corrector | None:
        if not self.enable_kogpt2:
            return None
        if self._kogpt2_corrector is None:
            logger.info("KoGPT2 로드: %s", self.kogpt2_model_name)
            # proposal_model로 KLUE-RoBERTa 사용
            # kobert_context가 이미 로드했다면 재사용, 아니면 여기서 로드
            kobert = self._get_kobert_context_corrector()
            self._kogpt2_corrector = KoGPT2Corrector(
                model_name=self.kogpt2_model_name,
                device=self._device,
                medical_terms=self.medical_terms,
                max_jamo_distance=self.kogpt2_max_jamo_distance,
                # KLUE-RoBERTa 인스턴스 주입 (모델 중복 로드 방지)
                proposal_model=kobert.model if kobert else None,
                proposal_tokenizer=kobert.tokenizer if kobert else None,
            )
        return self._kogpt2_corrector

    def _get_kobert_context_corrector(self) -> KoBERTContextCorrector | None:
        if not self.enable_kobert_context:
            return None
        if self._kobert_context_corrector is None:
            logger.info("Context MLM 로드: %s", self.kobert_model_name)
            self._kobert_context_corrector = KoBERTContextCorrector(
                model_name=self.kobert_model_name,
                device=self._device,
                medical_terms=self.medical_terms,
                protected_surfaces={"수속", "수속하면"},
                jamo_max_edit_distance=2,
                alpha_mlm=1.0,
                beta_jamo=0.8,
                medical_bonus=0.25,
            )
        return self._kobert_context_corrector

    def process_text(self, text: str) -> PipelineResult:
        original = text
        stages: dict = {}

        logger.info("\n%s", "=" * 60)
        logger.info("[입력] %s", text[:100] + ("..." if len(text) > 100 else ""))

        # Stage 1: Rule-based
        text, rule_changes = apply_rule_based(text)
        stages["rule_based"] = {"output": text, "changes": rule_changes}
        if rule_changes:
            logger.info("[Stage 1 - Rule-based] %d건 교정", len(rule_changes))

        # Stage 2: Medical confusion
        text, medical_confusion_changes = apply_confusion_replacements(
            text, confusion_set=DEFAULT_MEDICAL_CONFUSION_SET
        )
        stages["medical_confusion"] = {"output": text, "changes": medical_confusion_changes}
        if medical_confusion_changes:
            logger.info("[Stage 2 - Medical confusion] %d종 치환", len(medical_confusion_changes))

        # Stage 3: KoGPT2 PPL Span Correction
        kg = self._get_kogpt2_corrector()
        if kg is not None:
            try:
                text, kg_changes = kg.correct_text(
                    text,
                    top_k=self.kogpt2_top_k,
                    min_improve=self.kogpt2_min_improve,
                    min_span_chars=self.kogpt2_min_span_chars,
                )
                stages["kogpt2_ppl"] = {"output": text, "changes": kg_changes}
                if kg_changes:
                    logger.info("[Stage 3 - KoGPT2 PPL] %d건 교정", len(kg_changes))
                else:
                    stages["kogpt2_ppl"]["skipped"] = False
            except Exception as e:
                logger.exception("KoGPT2 PPL 교정 실패: %s", e)
                stages["kogpt2_ppl"] = {"output": text, "changes": [], "error": str(e)}
        else:
            stages["kogpt2_ppl"] = {"output": text, "changes": [], "skipped": True, "reason": "disabled"}

        # Stage 4: Context MLM (KLUE-RoBERTa)
        kb = self._get_kobert_context_corrector()
        if kb is not None:
            try:
                text, kb_changes = kb.correct_text(
                    text,
                    anomaly_threshold=self.kobert_anomaly_threshold,
                    top_k=self.kobert_top_k,
                    min_candidate_prob=self.kobert_min_candidate_prob,
                    max_word_edit_distance=self.kobert_max_word_edit_distance,
                    min_span_chars=self.kobert_min_span_chars,
                    window_chars=self.kobert_window_chars,
                )
                stages["kobert_context"] = {"output": text, "changes": kb_changes}
                if kb_changes:
                    logger.info("[Stage 4 - Context MLM] %d건 교정", len(kb_changes))
            except Exception as e:
                logger.exception("Context MLM 교정 실패: %s", e)
                stages["kobert_context"] = {"output": text, "changes": [], "error": str(e)}
        else:
            stages["kobert_context"] = {"output": text, "changes": [], "skipped": True, "reason": "disabled"}

        logger.info("[출력] %s", text[:100] + ("..." if len(text) > 100 else ""))
        return PipelineResult(original=original, corrected=text, stages=stages)

    def process_batch(self, texts: list[str]) -> list[PipelineResult]:
        results = []
        for i, text in enumerate(texts):
            logger.info("\n--- 텍스트 %d/%d 처리 중 ---", i + 1, len(texts))
            results.append(self.process_text(text))
        return results


def load_input(file_path: str | Path) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    content = path.read_text(encoding="utf-8").strip()

    if path.suffix.lower() == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return [item["text"] if isinstance(item, dict) else str(item) for item in data]
        if isinstance(data, dict) and "text" in data:
            return [data["text"]]
        raise ValueError("JSON 형식이 올바르지 않습니다.")

    if path.suffix.lower() == ".txt":
        if content.startswith("[") or content.startswith("{"):
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return [item["text"] if isinstance(item, dict) else str(item) for item in data]
                if isinstance(data, dict) and "text" in data:
                    return [data["text"]]
            except json.JSONDecodeError:
                pass
        return [content]

    return [content]


def format_comparison(result: PipelineResult) -> str:
    lines = ["=" * 70, "[원문]", result.original, ""]
    for stage_name, stage_data in result.stages.items():
        changes = stage_data.get("changes", [])
        if stage_data.get("skipped", False):
            lines.append(f"[{stage_name}] (비활성화)")
            continue
        if changes:
            lines.append(f"[{stage_name}] {len(changes)}건 교정:")
            for c in changes:
                detail = f"  - '{c['original']}' → '{c['corrected']}'"
                if "edit_distance" in c:
                    detail += f" (편집거리: {c['edit_distance']})"
                if "confidence" in c:
                    detail += f" (신뢰도: {c['confidence']})"
                if "improve" in c:
                    detail += f" (NLL개선: {c['improve']:.4f})"
                lines.append(detail)
        else:
            lines.append(f"[{stage_name}] 교정 없음")
    lines += ["", "[교정 결과]", result.corrected, "=" * 70]
    return "\n".join(lines)
