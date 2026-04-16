"""KU-RIAS 변환 BertForMaskedLM 기반 이상 감지 교정.

고정 의료 치환은 `medical_confusion`(의료 용어만)에서 먼저 적용된다.
MLM은 문맥 기반 보정; `DEFAULT_CONFUSION_SET`에는 의료 매핑 + MLM 표면 차단(빈 set)이 합쳐져 있다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
import Levenshtein
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.korean_text_utils import speech_endings_compatible, split_josa
from src.medical_confusion import DEFAULT_CONFUSION_SET

logger = logging.getLogger(__name__)

_KOREAN = re.compile(r"[가-힣]+")


def _strip_wordpiece(t: str) -> str:
    return t[2:] if t.startswith("##") else t


@dataclass
class MlmCandidate:
    token: str
    surface: str
    prob: float


@dataclass
class MlmCorrection:
    start: int
    end: int
    original: str
    chosen: str | None
    anomaly_score: float
    threshold: float
    topk: list[MlmCandidate]
    filtered_topk: list[MlmCandidate]


class MlmCorrector:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        medical_terms: set[str] | None = None,
        confusion_set: dict[str, set[str]] | None = None,
        *,
        require_medical_dict_match: bool = False,
        confusion_whitelist_only: bool = False,
        max_word_edit_distance: int | None = 1,
        min_span_chars: int = 3,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
        self.medical_terms = medical_terms or set()
        self.confusion_set = confusion_set if confusion_set is not None else DEFAULT_CONFUSION_SET
        self.require_medical_dict_match = require_medical_dict_match
        self.confusion_whitelist_only = confusion_whitelist_only
        self.max_word_edit_distance = (
            None if max_word_edit_distance is None else max(0, int(max_word_edit_distance))
        )
        self.min_span_chars = max(1, int(min_span_chars))
        self._mecab = self._try_load_mecab()
        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError("mask_token 없음")

    def word_spans(self, text: str) -> list[tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group()) for m in _KOREAN.finditer(text)]

    def _mlm_logits_at_single_mask(self, masked_text: str):
        enc = self.tokenizer(masked_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        mid = self.tokenizer.mask_token_id
        positions = (input_ids == mid).nonzero(as_tuple=True)
        if input_ids.shape[0] != 1 or len(positions[0]) != 1:
            return None, None
        pos = int(positions[1][0])
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, pos]
        return logits, pos

    def _masked_context(self, text: str, start: int, end: int, window_chars: int) -> str:
        left = max(0, start - window_chars)
        right = min(len(text), end + window_chars)
        return text[left:start] + self.tokenizer.mask_token + text[end:right]

    def _topk_from_logits(self, logits: torch.Tensor, top_k: int) -> list[MlmCandidate]:
        if logits is None:
            return []
        probs = F.softmax(logits.float(), dim=-1)
        k = min(top_k, probs.shape[0])
        vals, idx = torch.topk(probs, k=k)
        out: list[MlmCandidate] = []
        for i in range(k):
            tid = int(idx[i].item())
            piece = self.tokenizer.convert_ids_to_tokens(tid)
            if piece is None:
                continue
            out.append(MlmCandidate(piece, _strip_wordpiece(piece), float(vals[i].item())))
        return out

    def _try_load_mecab(self):
        try:
            from konlpy.tag import Mecab  # type: ignore

            logger.info("MLM POS 게이트: MeCab 활성화")
            return Mecab()
        except Exception:
            logger.info("MLM POS 게이트: MeCab 미활성화 (설치되지 않음)")
            return None

    def _extract_josa(self, word: str) -> tuple[str, str]:
        return split_josa(word)

    def _mecab_pos_head(self, word: str) -> str:
        if self._mecab is None:
            return ""
        try:
            tags = self._mecab.pos(word)
            if not tags:
                return ""
            pos = tags[0][1] or ""
            return pos.split("+")[0]
        except Exception:
            return ""

    def _allow_pos(self, original: str, candidate: str) -> bool:
        # S3: MeCab 없으면 통과.
        if self._mecab is None:
            return True
        po = self._mecab_pos_head(original)
        pc = self._mecab_pos_head(candidate)
        if not po or not pc:
            return True
        # 거친 품사군 정렬: 첫 문자 동일하면 같은 군으로 판단.
        return po[0] == pc[0]

    def mlm_score(
        self,
        text: str,
        start: int,
        end: int,
        top_k: int = 50,
        window_chars: int = 72,
    ) -> tuple[float, list[MlmCandidate], bool]:
        original = text[start:end]
        orig_stem, _ = self._extract_josa(original)
        context = self._masked_context(text, start, end, window_chars=window_chars)
        logits, _ = self._mlm_logits_at_single_mask(context)
        if logits is None:
            return 0.0, [], False
        topk = self._topk_from_logits(logits, top_k=top_k)
        if not topk:
            return 0.0, [], False
        original_p = 0.0
        # S1: 원형(원문 또는 어간)이 top-k에 있으면 비정상 아님으로 간주
        keep_original = False
        for c in topk:
            if c.surface == original:
                original_p = c.prob
                keep_original = True
            cand_stem, _ = self._extract_josa(c.surface)
            if cand_stem == orig_stem:
                keep_original = True
        return original_p, topk, keep_original

    def propose_correction(
        self,
        text: str,
        start: int,
        end: int,
        anomaly_threshold: float = 0.03,
        top_k: int = 30,
        min_candidate_prob: float = 0.05,
        window_chars: int = 72,
    ) -> MlmCorrection:
        original = text[start:end]
        if len(original) < self.min_span_chars:
            return MlmCorrection(
                start, end, original, None, 0.0, anomaly_threshold, [], [],
            )
        score, cands, keep_original = self.mlm_score(text, start, end, top_k=top_k, window_chars=window_chars)
        if keep_original or score >= anomaly_threshold:
            return MlmCorrection(start, end, original, None, score, anomaly_threshold, cands, [])

        filtered: list[MlmCandidate] = []
        orig_stem, orig_josa = self._extract_josa(original)
        allowed = self.confusion_set.get(original)
        if allowed is not None and len(allowed) == 0:
            return MlmCorrection(start, end, original, None, score, anomaly_threshold, cands, [])
        if self.confusion_whitelist_only and allowed is None:
            return MlmCorrection(start, end, original, None, score, anomaly_threshold, cands, [])
        for c in cands:
            if not _KOREAN.fullmatch(c.surface):
                continue
            if c.prob < min_candidate_prob:
                continue
            if c.surface == original:
                continue
            cand_stem, cand_josa = self._extract_josa(c.surface)
            # S2: 조사/어미 보존
            if orig_josa != cand_josa:
                continue
            if self.max_word_edit_distance is not None:
                if Levenshtein.distance(original, c.surface) > self.max_word_edit_distance:
                    continue
            if self.confusion_whitelist_only and allowed is not None and len(allowed) > 0:
                if c.surface not in allowed and cand_stem not in allowed:
                    continue
            if not self._allow_pos(orig_stem, cand_stem):
                continue
            if (
                self.require_medical_dict_match
                and self.medical_terms
                and c.surface not in self.medical_terms
                and cand_stem not in self.medical_terms
            ):
                continue
            filtered.append(c)
        filtered = [c for c in filtered if speech_endings_compatible(original, c.surface)]
        chosen = filtered[0].surface if filtered else None
        return MlmCorrection(start, end, original, chosen, score, anomaly_threshold, cands, filtered[:top_k])

    def correct_text(
        self,
        text: str,
        anomaly_threshold: float = 0.03,
        top_k: int = 30,
        min_candidate_prob: float = 0.05,
        window_chars: int = 72,
    ) -> tuple[str, list[dict]]:
        spans = self.word_spans(text)
        if not spans:
            return text, []
        spans.sort(key=lambda x: x[0], reverse=True)
        out = text
        changes: list[dict] = []
        for start, end, _word in spans:
            prop = self.propose_correction(
                out,
                start,
                end,
                anomaly_threshold=anomaly_threshold,
                top_k=top_k,
                min_candidate_prob=min_candidate_prob,
                window_chars=window_chars,
            )
            if prop.chosen is None or prop.chosen == prop.original:
                continue
            out = out[:start] + prop.chosen + out[end:]
            changes.append({
                "type": "mlm_topk",
                "start": start,
                "end": end,
                "original": prop.original,
                "corrected": prop.chosen,
                "anomaly_score": round(prop.anomaly_score, 6),
                "threshold": prop.threshold,
                "selected_prob": round(prop.filtered_topk[0].prob, 6) if prop.filtered_topk else 0.0,
                "alternatives": [c.surface for c in prop.filtered_topk[1:6]],
                "topk": [asdict(c) for c in prop.topk[:10]],
                "filtered_topk": [asdict(c) for c in prop.filtered_topk[:10]],
            })
        changes.reverse()
        return out, changes

    def correct_by_token_confidence(
        self,
        text: str,
        detect_threshold: float = 0.03,
        top_k: int = 30,
        min_candidate_prob: float = 0.05,
        window_chars: int = 72,
        max_word_edit_distance: int = 2,
        min_span_chars: int = 3,
    ) -> tuple[str, list[dict]]:
        """
        STT 토큰별 확신도 기반 교정.
        - 원문 토큰 확률(score)이 detect_threshold 미만이면 의심 토큰으로 간주
        - 의심 토큰을 [MASK] 문맥으로 top-k 후보 추출 후 가장 확률 높은 후보로 교정
        """
        spans = self.word_spans(text)
        if not spans:
            return text, []
        spans.sort(key=lambda x: x[0], reverse=True)

        out = text
        changes: list[dict] = []
        max_ed = max(0, int(max_word_edit_distance))
        min_chars = max(1, int(min_span_chars))

        for start, end, _ in spans:
            original = out[start:end]
            if len(original) < min_chars:
                continue

            score, cands, keep_original = self.mlm_score(
                out,
                start,
                end,
                top_k=top_k,
                window_chars=window_chars,
            )
            if keep_original or score >= detect_threshold:
                continue

            orig_stem, orig_josa = self._extract_josa(original)
            filtered: list[MlmCandidate] = []
            for c in cands:
                if not _KOREAN.fullmatch(c.surface):
                    continue
                if c.surface == original:
                    continue
                if c.prob < min_candidate_prob:
                    continue
                cand_stem, cand_josa = self._extract_josa(c.surface)
                if cand_josa != orig_josa:
                    continue
                if Levenshtein.distance(original, c.surface) > max_ed:
                    continue
                if not self._allow_pos(orig_stem, cand_stem):
                    continue
                if not speech_endings_compatible(original, c.surface):
                    continue
                filtered.append(c)

            if not filtered:
                continue

            chosen = filtered[0].surface
            out = out[:start] + chosen + out[end:]
            changes.append(
                {
                    "type": "kmbert_mlm_token",
                    "start": start,
                    "end": end,
                    "original": original,
                    "corrected": chosen,
                    "confidence": round(score, 6),
                    "threshold": detect_threshold,
                    "selected_prob": round(filtered[0].prob, 6),
                    "alternatives": [c.surface for c in filtered[1:6]],
                    "topk": [asdict(c) for c in cands[:10]],
                }
            )

        changes.reverse()
        return out, changes
