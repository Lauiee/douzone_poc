"""
전체 후처리 파이프라인
1. Rule-based: STT 구문 + 숫자/단위/약어
1b. Medical confusion: 의료 용어 오인식 표만 고정 치환
2. KM-BERT MLM: 토큰 확신도 기반 의심 탐지 + [MASK] top-k 교정
3. KoGPT2 PPL(선택): span 문맥 후보 비교 교정 — 기본 비활성
4. Context MLM(선택): 구어·문맥 오류는 문맥 기반 교정
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.bert_corrector import BertCorrector
from src.jamo_corrector import JamoCorrector
from src.kobert_context_corrector import KoBERTContextCorrector
from src.kogpt2_corrector import KoGPT2Corrector
from src.medical_confusion import DEFAULT_MEDICAL_CONFUSION_SET, apply_confusion_replacements
from src.rule_based import apply_rule_based

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KURIAS_HF = _PROJECT_ROOT / "models" / "kmbert-kurias-vocab-hf"


def _has_local_kurias_weights(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any(path.glob("model*.safetensors")) or (path / "pytorch_model.bin").exists()


def default_kmbert_model_path() -> str:
    env = os.environ.get("KMBERT_MODEL", "").strip()
    if env:
        return env
    if _has_local_kurias_weights(_DEFAULT_KURIAS_HF):
        return str(_DEFAULT_KURIAS_HF)
    return "madatnlp/km-bert"


def default_mlm_model_path(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    env = os.environ.get("KMBERT_MLM_MODEL", "").strip()
    if env and Path(env).exists():
        return env
    if _has_local_kurias_weights(_DEFAULT_KURIAS_HF):
        return str(_DEFAULT_KURIAS_HF)
    return None


def _cuda_device_index(device: str) -> int:
    d = (device or "cpu").strip().lower()
    if d == "cpu":
        return 0
    if d == "cuda":
        return 0
    if d.startswith("cuda:"):
        return int(d.split(":", 1)[1])
    return 0


def apply_cuda_memory_limit(device: str) -> None:
    """첫 GPU 텐서 할당 전에 호출. 환경변수로 프로세스당 VRAM 상한 설정.

    - CUDA_MEMORY_LIMIT_GB: 카드 전체 용량 대비 목표 GiB (예: 16GB 카드에서 8만 쓰려면 8)
    - CUDA_MEMORY_FRACTION: 카드 총 VRAM 대비 비율 (예: 0.5). LIMIT_GB가 있으면 그쪽이 우선.
    """
    if not torch.cuda.is_available():
        return
    d = (device or "cpu").lower()
    if d == "cpu":
        return
    idx = _cuda_device_index(device)
    limit_gb = os.environ.get("CUDA_MEMORY_LIMIT_GB", "").strip()
    frac_env = os.environ.get("CUDA_MEMORY_FRACTION", "").strip()
    if not limit_gb and not frac_env:
        return
    try:
        if limit_gb:
            total = torch.cuda.get_device_properties(idx).total_memory
            want = float(limit_gb) * (1024**3)
            frac = min(1.0, want / float(total))
            torch.cuda.set_per_process_memory_fraction(frac, idx)
            logger.info(
                "CUDA 메모리 상한: 약 %.2f GiB (디바이스 %.2f GiB 중 비율 %.3f, cuda:%s)",
                want / (1024**3),
                total / (1024**3),
                frac,
                idx,
            )
        else:
            frac = float(frac_env)
            if not (0.0 < frac <= 1.0):
                logger.warning("CUDA_MEMORY_FRACTION은 (0,1] 이어야 함: %s 무시", frac_env)
                return
            torch.cuda.set_per_process_memory_fraction(frac, idx)
            total = torch.cuda.get_device_properties(idx).total_memory
            logger.info(
                "CUDA 메모리 비율: %.3f (디바이스 %s, 상한 약 %.2f GiB)",
                frac,
                idx,
                total * frac / (1024**3),
            )
    except Exception as e:
        logger.warning("CUDA 메모리 상한 설정 실패(무시): %s", e)


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
        model_name: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.45,
        max_edit_distance: int = 2,
        enable_bert: bool = True,
        bert_trace: bool = False,
        enable_mlm: bool = True,
        mlm_model_path: str | None = None,
        mlm_anomaly_threshold: float = 0.03,
        mlm_top_k: int = 30,
        mlm_min_candidate_prob: float = 0.12,
        mlm_window_chars: int = 72,
        mlm_legacy_dict_gate: bool = True,
        mlm_max_word_edit_distance: int = 1,
        mlm_min_span_chars: int = 3,
        enable_kogpt2: bool = False,
        kogpt2_model_name: str = "skt/kogpt2-base-v2",
        kogpt2_top_k: int = 40,
        kogpt2_max_word_edit_distance: int = 1,
        kogpt2_min_improve: float = 0.06,
        kogpt2_min_span_chars: int = 2,
        enable_kobert_context: bool = True,
        kobert_model_name: str = "klue/roberta-large",
        kobert_anomaly_threshold: float = 0.01,
        kobert_top_k: int = 50,
        kobert_min_candidate_prob: float = 0.05,
        kobert_max_word_edit_distance: int = 2,
        kobert_min_span_chars: int = 2,
        kobert_window_chars: int = 72,
    ):
        self.enable_bert = enable_bert
        self.bert_trace = bert_trace
        self.model_name = model_name if model_name is not None else default_kmbert_model_path()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        apply_cuda_memory_limit(self._device)

        self.enable_mlm = enable_mlm
        self.mlm_model_path = default_mlm_model_path(mlm_model_path) if enable_mlm else None
        self.mlm_anomaly_threshold = mlm_anomaly_threshold
        self.mlm_top_k = mlm_top_k
        self.mlm_min_candidate_prob = mlm_min_candidate_prob
        self.mlm_window_chars = mlm_window_chars
        self.mlm_legacy_dict_gate = mlm_legacy_dict_gate
        self.mlm_max_word_edit_distance = mlm_max_word_edit_distance
        self.mlm_min_span_chars = mlm_min_span_chars

        self.enable_kogpt2 = enable_kogpt2
        self.kogpt2_model_name = kogpt2_model_name
        self.kogpt2_top_k = kogpt2_top_k
        self.kogpt2_max_word_edit_distance = kogpt2_max_word_edit_distance
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

        self._mlm_corrector = None
        self._kogpt2_corrector = None
        self._kobert_context_corrector = None

        logger.info("=== 의료 STT 후처리 파이프라인 초기화 ===")
        logger.info("Step 1/5: Jamo Corrector 초기화")
        self.jamo_corrector = JamoCorrector(dict_path=dict_path, max_edit_distance=max_edit_distance)

        if self.enable_bert:
            logger.info("Step 2/5: KM-BERT Corrector 초기화")
            self.bert_corrector = BertCorrector(
                model_name=self.model_name,
                device=self._device,
                confidence_threshold=confidence_threshold,
                medical_terms=self.jamo_corrector.medical_terms,
            )
        else:
            logger.info("Step 2/5: KM-BERT 비활성화됨")
            self.bert_corrector = None

        if self.mlm_model_path:
            logger.info("Step 3/5: BertForMaskedLM 준비: %s", self.mlm_model_path)
        elif self.enable_mlm:
            logger.warning("Step 3/5: MLM refine 스킵 - 로컬 가중치 없음")
        else:
            logger.info("Step 3/5: MLM refine 비활성화됨")

        if self.enable_kogpt2:
            logger.info("Step 4/5: KoGPT2 PPL 준비: %s", self.kogpt2_model_name)
        else:
            logger.info("Step 4/5: KoGPT2 PPL 비활성화됨")

        if self.enable_kobert_context:
            logger.info("Context MLM 문맥 교정 준비: %s", self.kobert_model_name)

        logger.info("Step 5/5: 파이프라인 초기화 완료")

    def _get_mlm_corrector(self):
        if not self.enable_mlm or not self.mlm_model_path:
            return None
        if self._mlm_corrector is None:
            from src.mlm_corrector import MlmCorrector

            logger.info("BertForMaskedLM 로드: %s", self.mlm_model_path)
            self._mlm_corrector = MlmCorrector(
                self.mlm_model_path,
                device=self._device,
                medical_terms=self.jamo_corrector.medical_terms,
                require_medical_dict_match=self.mlm_legacy_dict_gate,
                confusion_whitelist_only=self.mlm_legacy_dict_gate,
                max_word_edit_distance=(None if self.mlm_legacy_dict_gate else self.mlm_max_word_edit_distance),
                min_span_chars=(1 if self.mlm_legacy_dict_gate else self.mlm_min_span_chars),
            )
        return self._mlm_corrector

    def _get_kogpt2_corrector(self):
        if not self.enable_kogpt2:
            return None
        if self._kogpt2_corrector is None:
            logger.info("KoGPT2 로드: %s", self.kogpt2_model_name)
            self._kogpt2_corrector = KoGPT2Corrector(
                model_name=self.kogpt2_model_name,
                proposal_model_name=self.model_name,
                device=self._device,
            )
        return self._kogpt2_corrector

    def _get_kobert_context_corrector(self):
        if not self.enable_kobert_context:
            return None
        if self._kobert_context_corrector is None:
            logger.info("Context MLM 교정기 로드: %s", self.kobert_model_name)
            self._kobert_context_corrector = KoBERTContextCorrector(
                model_name=self.kobert_model_name,
                device=self._device,
                medical_terms=self.jamo_corrector.medical_terms,
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

        text, rule_changes = apply_rule_based(text)
        stages["rule_based"] = {"output": text, "changes": rule_changes}
        if rule_changes:
            logger.info("[Stage 1 - Rule-based] %d건 교정", len(rule_changes))

        text, medical_confusion_changes = apply_confusion_replacements(
            text, confusion_set=DEFAULT_MEDICAL_CONFUSION_SET
        )
        stages["medical_confusion"] = {"output": text, "changes": medical_confusion_changes}
        if medical_confusion_changes:
            logger.info(
                "[Stage 1b - Medical confusion] %d종 치환",
                len(medical_confusion_changes),
            )

        stages["jamo_correction"] = {"output": text, "changes": [], "skipped": True, "reason": "disabled_in_new_flow"}
        stages["bert_mlm"] = {"output": text, "changes": [], "skipped": True, "reason": "disabled_in_new_flow"}

        mlm = self._get_mlm_corrector()
        if mlm is not None:
            try:
                text, mlm_changes = mlm.correct_by_token_confidence(
                    text,
                    detect_threshold=self.mlm_anomaly_threshold,
                    top_k=self.mlm_top_k,
                    min_candidate_prob=self.mlm_min_candidate_prob,
                    window_chars=self.mlm_window_chars,
                    max_word_edit_distance=self.mlm_max_word_edit_distance,
                    min_span_chars=self.mlm_min_span_chars,
                )
                stages["mlm_refine"] = {"output": text, "changes": mlm_changes}
                if mlm_changes:
                    logger.info("[Stage 2 - KM-BERT MLM] %d건 교정", len(mlm_changes))
            except Exception as e:
                logger.exception("MLM refine 실패: %s", e)
                stages["mlm_refine"] = {"output": text, "changes": [], "error": str(e)}
        else:
            skip_info: dict = {"skipped": True}
            if not self.enable_mlm:
                skip_info["reason"] = "disabled"
            elif not self.mlm_model_path:
                skip_info["reason"] = "no_local_mlm_weights"
            stages["mlm_refine"] = {"output": text, "changes": [], **skip_info}

        kg = self._get_kogpt2_corrector()
        if kg is not None:
            try:
                text, kg_changes = kg.correct_text(
                    text,
                    top_k=self.kogpt2_top_k,
                    max_word_edit_distance=self.kogpt2_max_word_edit_distance,
                    min_improve=self.kogpt2_min_improve,
                    min_span_chars=self.kogpt2_min_span_chars,
                    span_words=2,
                    per_word_top_k=3,
                    max_combinations=8,
                    min_improve_ratio=0.03,
                )
                stages["kogpt2_ppl"] = {"output": text, "changes": kg_changes}
                if kg_changes:
                    logger.info("[Stage 3 - KoGPT2 PPL] %d건 교정", len(kg_changes))
            except Exception as e:
                logger.exception("KoGPT2 PPL 교정 실패: %s", e)
                stages["kogpt2_ppl"] = {"output": text, "changes": [], "error": str(e)}
        else:
            stages["kogpt2_ppl"] = {"output": text, "changes": [], "skipped": True, "reason": "disabled"}

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
                logger.exception("Context MLM 문맥 교정 실패: %s", e)
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
        raise ValueError("JSON 형식이 올바르지 않습니다. list 또는 {'text': ...} 형태여야 합니다.")

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

    logger.warning("알 수 없는 확장자: %s, 텍스트 파일로 처리합니다.", path.suffix)
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
                lines.append(detail)
        else:
            lines.append(f"[{stage_name}] 교정 없음")
    lines += ["", "[교정 결과]", result.corrected, "=" * 70]
    return "\n".join(lines)
