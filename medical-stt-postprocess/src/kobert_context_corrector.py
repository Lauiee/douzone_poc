"""Context MLM 기반 문맥 부적합 토큰 교정."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

import Levenshtein
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.korean_text_utils import harmonize_josa, speech_endings_compatible, split_josa
from src.jamo_corrector import to_jamo

logger = logging.getLogger(__name__)

_KOREAN = re.compile(r"[가-힣]+")


def _strip_wordpiece(token: str) -> str:
    return token[2:] if token.startswith("##") else token


@dataclass
class Candidate:
    surface: str
    prob: float


class KoBERTContextCorrector:
    def __init__(
        self,
        model_name: str = "klue/roberta-large",
        device: str | None = None,
        medical_terms: set[str] | None = None,
        protected_surfaces: set[str] | None = None,
        jamo_max_edit_distance: int = 2,
        alpha_mlm: float = 1.0,
        beta_jamo: float = 0.8,
        medical_bonus: float = 0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.medical_terms = medical_terms or set()
        self.protected_surfaces = protected_surfaces or set()
        self.jamo_max_edit_distance = max(0, int(jamo_max_edit_distance))
        self.alpha_mlm = float(alpha_mlm)
        self.beta_jamo = float(beta_jamo)
        self.medical_bonus = float(medical_bonus)
        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError("Context MLM tokenizer에 [MASK] 토큰이 없습니다.")

    def _word_spans(self, text: str) -> list[tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group()) for m in _KOREAN.finditer(text)]

    def _mask_logits(self, text: str, start: int, end: int, window_chars: int) -> torch.Tensor | None:
        left = max(0, start - window_chars)
        right = min(len(text), end + window_chars)
        masked = text[left:start] + self.tokenizer.mask_token + text[end:right]
        enc = self.tokenizer(masked, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        mid = self.tokenizer.mask_token_id
        pos = (input_ids == mid).nonzero(as_tuple=True)
        if input_ids.shape[0] != 1 or len(pos[0]) != 1:
            return None
        mpos = int(pos[1][0])
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
        return out.logits[0, mpos]

    def _topk_candidates(self, logits: torch.Tensor, top_k: int) -> list[Candidate]:
        probs = F.softmax(logits.float(), dim=-1)
        k = min(top_k, int(probs.shape[0]))
        vals, idx = torch.topk(probs, k=k)
        out: list[Candidate] = []
        seen = set()
        for i in range(k):
            tid = int(idx[i].item())
            tok = self.tokenizer.convert_ids_to_tokens(tid) or ""
            surface = _strip_wordpiece(tok)
            if not surface or not _KOREAN.fullmatch(surface):
                continue
            if surface in seen:
                continue
            seen.add(surface)
            out.append(Candidate(surface=surface, prob=float(vals[i].item())))
        return out

    def correct_text(
        self,
        text: str,
        anomaly_threshold: float = 0.03,
        top_k: int = 30,
        min_candidate_prob: float = 0.05,
        max_word_edit_distance: int = 2,
        min_span_chars: int = 2,
        window_chars: int = 72,
    ) -> tuple[str, list[dict]]:
        spans = self._word_spans(text)
        if not spans:
            return text, []
        spans.sort(key=lambda x: x[0], reverse=True)

        out = text
        changes: list[dict] = []
        for start, end, original in spans:
            if len(original) < min_span_chars:
                continue
            # 문맥 모델 과교정 방지: 보호 표면은 교정 대상에서 제외.
            if original in self.protected_surfaces:
                continue
            logits = self._mask_logits(out, start, end, window_chars=window_chars)
            if logits is None:
                continue
            cands = self._topk_candidates(logits, top_k=top_k)
            if not cands:
                continue

            original_prob = 0.0
            for c in cands:
                if c.surface == original:
                    original_prob = c.prob
                    break
            # top-k에 원문 표면이 없으면 original_prob==0 → 교정하지 않음(과교정 방지).
            if original_prob <= 0.0:
                continue
            if original_prob >= anomaly_threshold:
                continue

            chosen: Candidate | None = None
            best_score = -1e9
            orig_stem, orig_josa = split_josa(original)
            o_jamo = to_jamo(orig_stem)
            for c in cands:
                if c.surface == original:
                    continue
                if c.prob < min_candidate_prob:
                    continue
                cand_stem, _cand_josa = split_josa(c.surface)
                if abs(len(cand_stem) - len(orig_stem)) > 1:
                    continue
                stem_edit = Levenshtein.distance(orig_stem, cand_stem)
                if stem_edit > max_word_edit_distance:
                    continue
                c_jamo = to_jamo(cand_stem)
                jdist = Levenshtein.distance(o_jamo, c_jamo)
                if jdist > self.jamo_max_edit_distance:
                    continue
                max_len = max(len(o_jamo), len(c_jamo), 1)
                jamo_sim = 1.0 - (jdist / max_len)
                score = self.alpha_mlm * c.prob + self.beta_jamo * jamo_sim
                if c.surface in self.medical_terms:
                    score += self.medical_bonus
                if score > best_score:
                    best_score = score
                    chosen = c

            if chosen is None:
                continue

            chosen_stem, _ = split_josa(chosen.surface)
            chosen_josa = harmonize_josa(chosen_stem, orig_josa)
            corrected_surface = chosen_stem + chosen_josa
            if not speech_endings_compatible(original, corrected_surface):
                continue
            out = out[:start] + corrected_surface + out[end:]
            changes.append(
                {
                    "type": "context_mlm",
                    "start": start,
                    "end": end,
                    "original": original,
                    "corrected": corrected_surface,
                    "confidence": round(original_prob, 6),
                    "threshold": anomaly_threshold,
                    "selected_prob": round(chosen.prob, 6),
                    "topk": [asdict(c) for c in cands[:10]],
                }
            )

        changes.reverse()
        return out, changes

