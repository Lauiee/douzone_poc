"""KoGPT2 기반 PPL Span Correction.

의료 사전 자모 후보 + KLUE-RoBERTa MLM 후보를 쓰되, RoBERTa 쪽은 원문 어간과
자모 거리가 일정 이하일 때만 채택해 무제한 일반어 과교정을 막는다. 최종 채택은
KoGPT2 NLL 개선(min_improve 등)으로 제한하며, NLL이 동률이면 MLM 확률로 타이브레이크한다.
의료 사전 후보만 MLM이 충분할 때 NLL이 소폭 악화돼도 채택할 수 있다(medical_nll_relax).
"""

from __future__ import annotations

import logging
import math
import re

import Levenshtein
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.jamo_corrector import to_jamo
from src.korean_text_utils import (
    harmonize_josa,
    looks_like_verb_conjugation,
    speech_endings_compatible,
    split_josa,
)

logger = logging.getLogger(__name__)

_KOREAN = re.compile(r"[가-힣]+")

# NLL 동률일 때만 MLM 타이브레이크 허용; 너무 낮은 MLM 점수는 무시(무의미한 동률 교정 방지)
_MLM_TIE_MIN_PROB = 0.05

# 짧은 의존명사·위치 어휘: split_josa 어간이 여기에 있으면 KoGPT2 스팬 교정 스킵
# (예: 전→정, 전부터→정부터 MLM 과교정 방지)
KOGPT2_PROTECTED_STEMS = frozenset({
    "전",
    "후",
    "중",
    "내",
    "간",
    "상",
    "하",
    "좌",
    "우",
    "초",
    "말",
    "초기",
    "말기",
})


class KoGPT2Corrector:
    def __init__(
        self,
        model_name: str = "skt/kogpt2-base-v2",
        device: str | None = None,
        medical_terms: set[str] | None = None,
        max_jamo_distance: int = 2,
        roberta_max_jamo_distance: int = 2,
        # RoBERTa: 전체 vocab에서 MLM≥하한 + 자모거리로 후보 (top-k만 쓰려면 full_vocab=False)
        roberta_full_vocab_jamo: bool = True,
        roberta_vocab_mlm_floor: float = 0.3,
        roberta_full_vocab_max_cand: int = 512,
        # KLUE-RoBERTa 인스턴스 주입 (MLM 후보; proposal 없으면 RoBERTa 경로 스킵)
        proposal_model=None,
        proposal_tokenizer=None,
        # 의료사전 후보만: NLL이 소폭 악화돼도 MLM이 충분하면 채택 (예: 이번→입원)
        max_nll_penalty_for_medical: float = 0.6,
        medical_relax_mlm_min_prob: float = 0.05,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # KoGPT2: NLL 계산 전용
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        self.medical_terms = medical_terms or set()
        self.max_jamo_distance = max_jamo_distance
        self.roberta_max_jamo_distance = roberta_max_jamo_distance
        self.roberta_full_vocab_jamo = roberta_full_vocab_jamo
        self.roberta_vocab_mlm_floor = roberta_vocab_mlm_floor
        self.roberta_full_vocab_max_cand = roberta_full_vocab_max_cand

        # 자모 캐시 미리 빌드
        self._jamo_cache: dict[str, str] = {}
        for term in self.medical_terms:
            self._jamo_cache[term] = to_jamo(term)

        # KLUE-RoBERTa (optional — MLM 후보 보조용, 없어도 동작)
        self.proposal_model = proposal_model
        self.proposal_tokenizer = proposal_tokenizer
        self.max_nll_penalty_for_medical = float(max_nll_penalty_for_medical)
        self.medical_relax_mlm_min_prob = float(medical_relax_mlm_min_prob)

        self._nll_cache: dict[str, float] = {}
        logger.info(
            "KoGPT2Corrector 초기화 완료 (의료사전 %d개, 의료자모≤%d, RoBERTa자모≤%d, "
            "RoBERTa후보=%s, MLM=%s)",
            len(self.medical_terms),
            max_jamo_distance,
            roberta_max_jamo_distance,
            (
                f"vocab전체≥{roberta_vocab_mlm_floor}"
                if roberta_full_vocab_jamo
                else "top-k만"
            ),
            "주입" if proposal_model else "없음",
        )

    def _word_spans(self, text: str) -> list[tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group()) for m in _KOREAN.finditer(text)]

    def _candidate_in_medical_dict(self, surface: str) -> bool:
        """교정 후보 표면의 어간(또는 전체)이 의료 사전에 있으면 True."""
        st, _ = split_josa(surface)
        return st in self.medical_terms or surface in self.medical_terms

    def _sent_nll(self, text: str) -> float:
        if text in self._nll_cache:
            return self._nll_cache[text]
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        nll = float(out.loss.item())
        self._nll_cache[text] = nll
        return nll

    def _jamo_candidates(self, word: str, max_jamo_dist: int | None = None) -> list[tuple[str, int]]:
        """의료 사전·자모 유사도 기반 후보 생성. 멀티토큰 단어도 후보에 포함."""
        dist_limit = max_jamo_dist if max_jamo_dist is not None else self.max_jamo_distance
        word_stem, _ = split_josa(word)
        word_jamo = to_jamo(word_stem)
        candidates = []

        for term in self.medical_terms:
            if term == word_stem:
                continue
            if abs(len(term) - len(word_stem)) > dist_limit + 1:
                continue
            term_jamo = self._jamo_cache.get(term) or to_jamo(term)
            jdist = Levenshtein.distance(word_jamo, term_jamo)
            if jdist <= dist_limit:
                candidates.append((term, jdist))

        # 거리 오름차순
        candidates.sort(key=lambda x: x[1])
        return candidates

    def _mlm_probs_at_mask(self, text: str, start: int, end: int) -> torch.Tensor | None:
        """마스크 위치 전체 vocab softmax (1D). RoBERTa 없으면 None."""
        if self.proposal_model is None or self.proposal_tokenizer is None:
            return None
        mask_tok = self.proposal_tokenizer.mask_token
        mask_id = self.proposal_tokenizer.mask_token_id
        if not mask_tok or mask_id is None:
            return None
        masked = text[:start] + mask_tok + text[end:]
        enc = self.proposal_tokenizer(masked, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        pos = (input_ids == mask_id).nonzero(as_tuple=True)
        if len(pos[0]) != 1:
            return None
        midx = int(pos[1][0])
        with torch.no_grad():
            out = self.proposal_model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0, midx]
        return F.softmax(logits.float(), dim=-1)

    def _surface_mlm_prob(self, surface: str, mlm_probs: torch.Tensor | None) -> float:
        """표면 문자열에 대응하는 토큰 ID(들)의 MLM 확률. 다토큰이면 곱으로 근사."""
        if mlm_probs is None or self.proposal_tokenizer is None:
            return 0.0
        ids = self.proposal_tokenizer.encode(surface, add_special_tokens=False)
        if not ids:
            return 0.0
        p = 1.0
        for tid in ids:
            if tid < 0 or tid >= mlm_probs.shape[0]:
                return 0.0
            p *= float(mlm_probs[tid].item())
        return p

    def _build_candidates(
        self,
        text: str,
        start: int,
        end: int,
        original: str,
        top_k: int,
    ) -> tuple[list[tuple[str, float]], torch.Tensor | None]:
        """(후보 표면, 해당 위치 MLM 확률) 목록 + 마스크 MLM 분포(타이브레이크·원문 확률용)."""
        orig_stem, orig_josa = split_josa(original)
        orig_jamo = to_jamo(orig_stem)
        scored: list[tuple[str, float]] = []
        seen: set[str] = set()

        mlm_probs = self._mlm_probs_at_mask(text, start, end)

        for term, _ in self._jamo_candidates(original):
            if term in seen:
                continue
            seen.add(term)
            restored = term + harmonize_josa(term, orig_josa)
            mp = self._surface_mlm_prob(restored, mlm_probs)
            scored.append((restored, mp))

        rj_limit = self.roberta_max_jamo_distance
        if mlm_probs is not None:
            if self.roberta_full_vocab_jamo:
                probs_cpu = mlm_probs.detach().float().cpu().view(-1)
                floor = float(self.roberta_vocab_mlm_floor)
                mask = probs_cpu >= floor
                cand_idx = torch.nonzero(mask, as_tuple=False).view(-1)
                if cand_idx.numel() > 0:
                    sub = probs_cpu[cand_idx]
                    order = torch.argsort(sub, descending=True)
                    cand_idx = cand_idx[order]
                tok = self.proposal_tokenizer
                n_added = 0
                for ii in range(cand_idx.shape[0]):
                    if n_added >= self.roberta_full_vocab_max_cand:
                        break
                    tid = int(cand_idx[ii].item())
                    p = float(probs_cpu[tid].item())
                    raw = tok.convert_ids_to_tokens(tid) or ""
                    word = raw[2:] if raw.startswith("##") else raw
                    if not word or not _KOREAN.fullmatch(word):
                        continue
                    word_stem, _ = split_josa(word)
                    word_jamo = to_jamo(word_stem)
                    if Levenshtein.distance(orig_jamo, word_jamo) > rj_limit:
                        continue
                    if word_stem in seen:
                        continue
                    seen.add(word_stem)
                    restored = word_stem + harmonize_josa(word_stem, orig_josa)
                    if restored == original:
                        continue
                    scored.append((restored, p))
                    n_added += 1
            else:
                k = min(top_k * 2, int(mlm_probs.shape[0]))
                top_p, top_i = torch.topk(mlm_probs, k=k)
                for j in range(k):
                    tid = int(top_i[j].item())
                    p = float(top_p[j].item())
                    raw = self.proposal_tokenizer.convert_ids_to_tokens(tid) or ""
                    word = raw[2:] if raw.startswith("##") else raw
                    if not word or not _KOREAN.fullmatch(word):
                        continue
                    word_stem, _ = split_josa(word)
                    word_jamo = to_jamo(word_stem)
                    if Levenshtein.distance(orig_jamo, word_jamo) > rj_limit:
                        continue
                    if word_stem in seen:
                        continue
                    seen.add(word_stem)
                    restored = word_stem + harmonize_josa(word_stem, orig_josa)
                    if restored == original:
                        continue
                    scored.append((restored, p))

        merged: dict[str, float] = {}
        for surf, mp in scored:
            merged[surf] = max(merged.get(surf, 0.0), mp)
        scored_out = sorted(merged.items(), key=lambda x: -x[1])[:top_k]
        return scored_out, mlm_probs

    def correct_text(
        self,
        text: str,
        top_k: int = 40,
        min_improve: float = 0.15,
        min_span_chars: int = 2,
        min_improve_ratio: float = 0.05,
    ) -> tuple[str, list[dict]]:
        self._nll_cache.clear()
        out = text
        changes: list[dict] = []

        # 역순 처리로 offset 관리 단순화
        spans = self._word_spans(out)
        spans.sort(key=lambda x: x[0], reverse=True)

        for start, end, original in spans:
            if len(original) < min_span_chars:
                continue
            if looks_like_verb_conjugation(original):
                continue
            stem, _ = split_josa(original)
            if stem in KOGPT2_PROTECTED_STEMS or original in KOGPT2_PROTECTED_STEMS:
                continue

            candidates_scored, mlm_probs = self._build_candidates(
                out, start, end, original, top_k=top_k
            )
            if not candidates_scored:
                continue

            orig_mlm_prob = self._surface_mlm_prob(original, mlm_probs)

            base_nll = self._sent_nll(out)
            best_nll = base_nll
            best_cand: str | None = None
            best_mlm_prob = -1.0

            for cand, mlm_p in candidates_scored:
                if cand == original:
                    continue
                if not speech_endings_compatible(original, cand):
                    continue
                trial = out[:start] + cand + out[end:]
                nll = self._sent_nll(trial)
                if nll < best_nll or (
                    math.isclose(nll, best_nll, rel_tol=0.0, abs_tol=1e-6)
                    and mlm_p > best_mlm_prob
                ):
                    best_nll = nll
                    best_cand = cand
                    best_mlm_prob = mlm_p

            passes_medical_nll_relax = False
            if best_cand is None:
                # NLL이 개선되는 후보가 없을 때: 의료사전 표면 + MLM 하한 + NLL 악화 상한
                relax: list[tuple[float, float, str, float, float]] = []
                for cand, mlm_p in candidates_scored:
                    if cand == original:
                        continue
                    if not speech_endings_compatible(original, cand):
                        continue
                    if not self._candidate_in_medical_dict(cand):
                        continue
                    if mlm_p < self.medical_relax_mlm_min_prob:
                        continue
                    trial = out[:start] + cand + out[end:]
                    tn = self._sent_nll(trial)
                    penalty = tn - base_nll
                    if penalty <= self.max_nll_penalty_for_medical:
                        relax.append((penalty, -mlm_p, cand, tn, mlm_p))
                if relax:
                    relax.sort(key=lambda x: (x[0], x[1]))
                    _, _, best_cand, best_nll, best_mlm_prob = relax[0]
                    passes_medical_nll_relax = True

            if best_cand is None:
                continue

            improve = base_nll - best_nll
            improve_ratio = improve / base_nll if base_nll > 0 else 0.0

            passes_nll = improve >= min_improve and improve_ratio >= min_improve_ratio
            passes_mlm_tie = (
                math.isclose(improve, 0.0, rel_tol=0.0, abs_tol=1e-5)
                and best_mlm_prob > orig_mlm_prob + 1e-15
                and best_mlm_prob >= _MLM_TIE_MIN_PROB
            )

            if not passes_nll and not passes_mlm_tie and not passes_medical_nll_relax:
                continue

            out = out[:start] + best_cand + out[end:]
            changes.append({
                "type": "kogpt2_ppl_span",
                "start": start,
                "end": end,
                "original": original,
                "corrected": best_cand,
                "nll_before": round(base_nll, 6),
                "nll_after": round(best_nll, 6),
                "improve": round(improve, 6),
                "improve_ratio": round(improve_ratio, 6),
                "mlm_prob_original": round(orig_mlm_prob, 8),
                "mlm_prob_corrected": round(best_mlm_prob, 8),
                "mlm_tiebreak": passes_mlm_tie and not passes_nll,
                "medical_nll_relax": passes_medical_nll_relax,
            })
            log_extra = ""
            if passes_mlm_tie and not passes_nll:
                log_extra = ", MLM타이브레이크"
            elif passes_medical_nll_relax:
                log_extra = ", 의료NLL완화"
            logger.info(
                "[KoGPT2] '%s' → '%s' (NLL개선 %.4f, MLM 원/교 %.4f/%.4f%s)",
                original,
                best_cand,
                improve,
                orig_mlm_prob,
                best_mlm_prob,
                log_extra,
            )

        changes.reverse()
        return out, changes
