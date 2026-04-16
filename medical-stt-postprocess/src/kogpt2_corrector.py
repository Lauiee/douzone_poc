"""KoGPT2 기반 PPL Span Correction.

변경사항 (KM-BERT 제거):
- proposal_model(KM-BERT) 제거
- 후보 생성: MASK 방식 → 자모 유사도 기반으로 교체
  → 멀티토큰 단어(탓인, 입원 등)도 후보로 생성 가능
- proposal_model 자리에 KLUE-RoBERTa 인스턴스 주입 옵션 유지
  (kobert_context와 모델 공유, 중복 로드 방지)
"""

from __future__ import annotations

import logging
import re

import Levenshtein
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.jamo_corrector import to_jamo
from src.korean_text_utils import speech_endings_compatible, split_josa

logger = logging.getLogger(__name__)

_KOREAN = re.compile(r"[가-힣]+")


class KoGPT2Corrector:
    def __init__(
        self,
        model_name: str = "skt/kogpt2-base-v2",
        device: str | None = None,
        medical_terms: set[str] | None = None,
        max_jamo_distance: int = 2,
        # KLUE-RoBERTa 인스턴스 주입 (optional, 모델 공유용)
        proposal_model=None,
        proposal_tokenizer=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # KoGPT2: NLL 계산 전용
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        self.medical_terms = medical_terms or set()
        self.max_jamo_distance = max_jamo_distance

        # 자모 캐시 미리 빌드
        self._jamo_cache: dict[str, str] = {}
        for term in self.medical_terms:
            self._jamo_cache[term] = to_jamo(term)

        # KLUE-RoBERTa (optional — MLM 후보 보조용, 없어도 동작)
        self.proposal_model = proposal_model
        self.proposal_tokenizer = proposal_tokenizer

        self._nll_cache: dict[str, float] = {}
        logger.info(
            "KoGPT2Corrector 초기화 완료 (의료사전 %d개, max_jamo_dist=%d, proposal_model=%s)",
            len(self.medical_terms),
            max_jamo_distance,
            "RoBERTa(주입)" if proposal_model else "없음",
        )

    def _word_spans(self, text: str) -> list[tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group()) for m in _KOREAN.finditer(text)]

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
        """자모 유사도 기반 후보 생성.
        
        KM-BERT MASK 방식 대신 사용.
        멀티토큰 단어(탓인, 입원 등)도 후보로 잡힘.
        """
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

    def _roberta_candidates(self, text: str, start: int, end: int, top_k: int = 30) -> list[str]:
        """KLUE-RoBERTa MASK 후보 (주입된 경우만 사용, 단일토큰 한정)."""
        if self.proposal_model is None or self.proposal_tokenizer is None:
            return []
        import torch.nn.functional as F
        mask_tok = self.proposal_tokenizer.mask_token
        mask_id = self.proposal_tokenizer.mask_token_id
        if not mask_tok or mask_id is None:
            return []
        masked = text[:start] + mask_tok + text[end:]
        enc = self.proposal_tokenizer(masked, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        pos = (input_ids == mask_id).nonzero(as_tuple=True)
        if len(pos[0]) != 1:
            return []
        midx = int(pos[1][0])
        with torch.no_grad():
            out = self.proposal_model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0, midx]
        probs = F.softmax(logits.float(), dim=-1)
        k = min(top_k, int(probs.shape[0]))
        _, idx = torch.topk(probs, k=k)
        words: list[str] = []
        seen: set[str] = set()
        for i in range(k):
            tid = int(idx[i].item())
            tok = self.proposal_tokenizer.convert_ids_to_tokens(tid) or ""
            tok = tok[2:] if tok.startswith("##") else tok
            if not tok or not _KOREAN.fullmatch(tok):
                continue
            if tok in seen:
                continue
            seen.add(tok)
            words.append(tok)
        return words

    def _build_candidates(
        self,
        text: str,
        start: int,
        end: int,
        original: str,
        top_k: int,
    ) -> list[str]:
        """자모 유사도 + RoBERTa(optional) 후보 통합."""
        orig_stem, orig_josa = split_josa(original)
        candidates: list[str] = []
        seen: set[str] = set()

        # 1. 자모 유사도 기반 (멀티토큰 포함)
        for term, _ in self._jamo_candidates(original):
            if term in seen:
                continue
            seen.add(term)
            # 조사 복원
            from src.korean_text_utils import harmonize_josa
            restored = term + harmonize_josa(term, orig_josa)
            candidates.append(restored)

        # 2. RoBERTa MASK 기반 (단일토큰, optional 보완)
        for word in self._roberta_candidates(text, start, end, top_k=top_k):
            if word in seen or word == original:
                continue
            seen.add(word)
            candidates.append(word)

        return candidates[:top_k]

    def correct_text(
        self,
        text: str,
        top_k: int = 40,
        min_improve: float = 0.04,
        min_span_chars: int = 2,
        min_improve_ratio: float = 0.02,
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

            candidates = self._build_candidates(out, start, end, original, top_k=top_k)
            if not candidates:
                continue

            base_nll = self._sent_nll(out)
            best_nll = base_nll
            best_cand: str | None = None

            for cand in candidates:
                if cand == original:
                    continue
                if not speech_endings_compatible(original, cand):
                    continue
                trial = out[:start] + cand + out[end:]
                nll = self._sent_nll(trial)
                if nll < best_nll:
                    best_nll = nll
                    best_cand = cand

            if best_cand is None:
                continue

            improve = base_nll - best_nll
            improve_ratio = improve / base_nll if base_nll > 0 else 0.0

            if improve < min_improve or improve_ratio < min_improve_ratio:
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
            })
            logger.info("[KoGPT2] '%s' → '%s' (NLL개선 %.4f)", original, best_cand, improve)

        changes.reverse()
        return out, changes
