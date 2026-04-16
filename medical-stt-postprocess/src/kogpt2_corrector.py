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
    # 1인칭 대명사: 존댓말 문맥에서 '내가/뭐가' 등으로 과교정되는 것을 차단
    "제가",
    "저는",
    "저를",
    "저도",
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
        # 멀티토큰(2-subword) 후보: 양방향 조건부 MLM + 자모 필터 (예: 타신→탓인, 조이→쪽이)
        # 안전장치: 원문 stem이 정확히 2자인 경우에만 시도하며, 후보 stem 길이도 2자로 고정,
        # joint MLM prob 하한과 원문 대비 prob 비율 하한으로 과교정을 차단한다.
        multi_token_enable: bool = True,
        multi_token_span_chars: int = 2,
        multi_token_k1: int = 10,
        multi_token_k2: int = 12,
        multi_token_max_candidates: int = 16,
        # 후보 생성 단계의 joint MLM prob 하한 (너무 드문 조합 컷)
        multi_token_min_joint_prob: float = 5e-4,
        # 멀티토큰 전용 후보 채택 조건:
        #  - KoGPT2 NLL 악화 금지 (multi_token_nll_min_improve 이상, 기본 0 = 동률 허용)
        #  - 후보 MLM joint prob ≥ multi_token_accept_min_prob (기본 1e-3)
        #  - 후보 prob / 원문 prob ≥ multi_token_accept_min_prob_ratio (기본 10배)
        # 이렇게 MLM 척도로 채택을 결정, NLL 은 안전장치로만 사용한다. KoGPT2 가
        # 타신↔탓인 같은 짧은 Korean 2-글자 교정의 PPL 차이를 사실상 구분 못하기 때문.
        multi_token_nll_min_improve: float = 0.01,
        multi_token_accept_min_prob: float = 1e-3,
        multi_token_accept_min_prob_ratio: float = 10.0,
        # 원문이 MLM 관점에서 정상(=prob 가 임계치 이상)이면 멀티토큰 교정 자체 차단.
        # 타신, 조이처럼 원문이 문맥상 매우 드문 경우에만 발화. STT 오류 교정 용도.
        multi_token_max_orig_prob: float = 1e-5,
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

        self.multi_token_enable = bool(multi_token_enable)
        self.multi_token_span_chars = int(multi_token_span_chars)
        self.multi_token_k1 = int(multi_token_k1)
        self.multi_token_k2 = int(multi_token_k2)
        self.multi_token_max_candidates = int(multi_token_max_candidates)
        self.multi_token_min_joint_prob = float(multi_token_min_joint_prob)
        self.multi_token_nll_min_improve = float(multi_token_nll_min_improve)
        self.multi_token_accept_min_prob = float(multi_token_accept_min_prob)
        self.multi_token_accept_min_prob_ratio = float(multi_token_accept_min_prob_ratio)
        self.multi_token_max_orig_prob = float(multi_token_max_orig_prob)

        self._nll_cache: dict[str, float] = {}
        logger.info(
            "KoGPT2Corrector 초기화 완료 (의료사전 %d개, 의료자모≤%d, RoBERTa자모≤%d, "
            "RoBERTa후보=%s, MLM=%s, 멀티토큰=%s)",
            len(self.medical_terms),
            max_jamo_distance,
            roberta_max_jamo_distance,
            (
                f"vocab전체≥{roberta_vocab_mlm_floor}"
                if roberta_full_vocab_jamo
                else "top-k만"
            ),
            "주입" if proposal_model else "없음",
            (
                f"k1={multi_token_k1}·k2={multi_token_k2}·자모≤{roberta_max_jamo_distance}"
                if multi_token_enable
                else "off"
            ),
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

    # ------------------------------------------------------------------
    # 멀티토큰(2-subword) 후보 생성: 양방향 조건부 MLM + 자모 필터
    # ------------------------------------------------------------------
    def _token_piece(self, tid: int) -> str:
        raw = self.proposal_tokenizer.convert_ids_to_tokens(int(tid)) or ""
        return raw[2:] if raw.startswith("##") else raw

    def _multi_token_candidates(
        self,
        text: str,
        start: int,
        end: int,
        original: str,
    ) -> list[tuple[str, float]]:
        """스팬 전체를 2-[MASK] 로 치환해 양방향 조건부 top-k 조합을 자모거리로 필터링.

        단일-[MASK] 경로는 서브워드 1 개짜리 후보만 만들 수 있어 `탓+##인`, `쪽+##이`
        같은 2-서브워드 교정을 구조적으로 놓친다. josa 를 따로 떼지 않고 원문 전체
        surface 길이(기본 2자)가 정확히 `multi_token_span_chars` 일 때만 시도하고,
        두 [MASK] 를 두고
          - LTR: pos0 top-k1 → 각 pos0 고정 후 pos1 조건부 top-k2
          - RTL: pos1 top-k1 → 각 pos1 고정 후 pos0 조건부 top-k2
        두 경로의 모든 조합 중 자모거리 ≤ roberta_max_jamo_distance 이고
        joint prob ≥ multi_token_min_joint_prob 인 것만 남겨 상위 N 개를 반환한다.
        반환 surface 에는 josa harmonize 를 하지 않는다 (모델이 조사까지 예측).
        """
        if self.proposal_model is None or self.proposal_tokenizer is None:
            return []
        if not self.multi_token_enable:
            return []
        if len(original) != self.multi_token_span_chars:
            return []

        tok = self.proposal_tokenizer
        mid = tok.mask_token_id
        cls = tok.cls_token_id
        sep = tok.sep_token_id
        if mid is None or cls is None or sep is None:
            return []

        prefix_ids = tok(text[:start], add_special_tokens=False)["input_ids"]
        suffix_ids = tok(text[end:], add_special_tokens=False)["input_ids"]
        pos0 = 1 + len(prefix_ids)
        pos1 = pos0 + 1

        def logits_with(mid_ids: list[int]) -> torch.Tensor:
            full = [cls] + prefix_ids + mid_ids + suffix_ids + [sep]
            inp = torch.tensor([full], device=self.device)
            with torch.no_grad():
                out = self.proposal_model(input_ids=inp)
            return out.logits[0]

        def topk_at(logits: torch.Tensor, pos: int, k: int) -> list[tuple[int, float]]:
            probs = F.softmax(logits[pos].float(), dim=-1)
            tp, ti = torch.topk(probs, k=min(k, int(probs.shape[0])))
            return [(int(i.item()), float(p_.item())) for p_, i in zip(tp, ti)]

        orig_jamo = to_jamo(original)
        jamo_limit = self.roberta_max_jamo_distance
        target_len = self.multi_token_span_chars
        jp_floor = self.multi_token_min_joint_prob

        cands: dict[str, float] = {}
        base_logits = logits_with([mid, mid])

        def consider(s0: str, s1: str, jp: float) -> None:
            if not s0 or not s1:
                return
            if not _KOREAN.fullmatch(s0) or not _KOREAN.fullmatch(s1):
                return
            surf = s0 + s1
            if len(surf) != target_len:
                return
            if surf == original:
                return
            if Levenshtein.distance(orig_jamo, to_jamo(surf)) > jamo_limit:
                return
            if jp < jp_floor:
                return
            if jp > cands.get(surf, 0.0):
                cands[surf] = jp

        # LTR: pos0 top-k1 → 각 pos0 고정 후 pos1 조건부 top-k2
        for tid0, p0 in topk_at(base_logits, pos0, self.multi_token_k1):
            s0 = self._token_piece(tid0)
            if not s0:
                continue
            cond_logits = logits_with([tid0, mid])
            for tid1, p1 in topk_at(cond_logits, pos1, self.multi_token_k2):
                consider(s0, self._token_piece(tid1), p0 * p1)

        # RTL: pos1 top-k1 → 각 pos1 고정 후 pos0 조건부 top-k2
        for tid1, p1 in topk_at(base_logits, pos1, self.multi_token_k1):
            s1 = self._token_piece(tid1)
            if not s1:
                continue
            cond_logits = logits_with([mid, tid1])
            for tid0, p0 in topk_at(cond_logits, pos0, self.multi_token_k2):
                consider(self._token_piece(tid0), s1, p0 * p1)

        if not cands:
            return []
        return sorted(cands.items(), key=lambda x: -x[1])[: self.multi_token_max_candidates]

    def _build_candidates(
        self,
        text: str,
        start: int,
        end: int,
        original: str,
        top_k: int,
    ) -> tuple[list[tuple[str, float]], torch.Tensor | None, set[str], set[str]]:
        """(후보 표면, 해당 위치 MLM 확률) 목록 + 마스크 MLM 분포(타이브레이크·원문 확률용)
        + 멀티토큰 경로에서만 생성된 surface 집합 (NLL 게이트에서 더 엄격 적용용)
        + RoBERTa full-vocab jamo 경로에서만 생성된 surface 집합 (MLM 타이브레이크 차단용).
        의료사전(_jamo_candidates) 출신 후보는 vocab_only 에 포함되지 않는다."""
        orig_stem, orig_josa = split_josa(original)
        orig_jamo = to_jamo(orig_stem)
        scored: list[tuple[str, float]] = []
        seen: set[str] = set()
        medical_surfaces: set[str] = set()

        mlm_probs = self._mlm_probs_at_mask(text, start, end)

        for term, _ in self._jamo_candidates(original):
            if term in seen:
                continue
            seen.add(term)
            restored = term + harmonize_josa(term, orig_josa)
            mp = self._surface_mlm_prob(restored, mlm_probs)
            scored.append((restored, mp))
            medical_surfaces.add(restored)

        rj_limit = self.roberta_max_jamo_distance
        vocab_only: set[str] = set()
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
                    if restored not in medical_surfaces:
                        vocab_only.add(restored)
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
                    if restored not in medical_surfaces:
                        vocab_only.add(restored)

        # 단일 [MASK] 경로가 만든 surface 집합 (멀티토큰-only 식별용)
        single_mask_surfaces = {s for s, _ in scored}

        # 멀티토큰(2-subword) 후보 — 단일 MASK 경로가 구조적으로 못 만드는 조합
        # (예: 타신→탓인 = 탓+##인, 조이→쪽이 = 쪽+##이) 커버.
        # 원문 전체 surface(josa 포함) 2자에 한해 전체를 2-[MASK] 로 치환한다.
        multi_only: set[str] = set()
        if self.multi_token_enable and self.proposal_model is not None:
            for surf, mp in self._multi_token_candidates(text, start, end, original):
                if surf == original:
                    continue
                scored.append((surf, mp))
                if surf not in single_mask_surfaces:
                    multi_only.add(surf)

        merged: dict[str, float] = {}
        for surf, mp in scored:
            merged[surf] = max(merged.get(surf, 0.0), mp)
        scored_out = sorted(merged.items(), key=lambda x: -x[1])[:top_k]
        final_surfaces = {s for s, _ in scored_out}
        multi_only &= final_surfaces
        vocab_only &= final_surfaces
        return scored_out, mlm_probs, multi_only, vocab_only

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

            candidates_scored, mlm_probs, multi_only, vocab_only = self._build_candidates(
                out, start, end, original, top_k=top_k
            )
            if not candidates_scored:
                continue

            orig_mlm_prob = self._surface_mlm_prob(original, mlm_probs)

            base_nll = self._sent_nll(out)
            best_nll = base_nll
            best_cand: str | None = None
            best_mlm_prob = -1.0

            # 단일경로 후보: 기존 NLL-기반 best 경쟁
            # 멀티토큰 전용 후보: 별도 트랙 (MLM prob 기반)
            mt_best: tuple[float, float, str] | None = None  # (mlm_p, nll, cand)

            for cand, mlm_p in candidates_scored:
                if cand == original:
                    continue
                if not speech_endings_compatible(original, cand):
                    continue
                trial = out[:start] + cand + out[end:]
                nll = self._sent_nll(trial)

                if cand in multi_only:
                    # 멀티토큰 트랙: 원문이 MLM 상 매우 낮아야 하고(정상 단어 보호),
                    # 후보는 절대/상대 prob 하한을 모두 통과해야 함. NLL 악화 금지.
                    if orig_mlm_prob > self.multi_token_max_orig_prob:
                        continue
                    imp = base_nll - nll
                    if imp < self.multi_token_nll_min_improve:
                        continue
                    if mlm_p < self.multi_token_accept_min_prob:
                        continue
                    if mlm_p < orig_mlm_prob * self.multi_token_accept_min_prob_ratio:
                        continue
                    if mt_best is None or mlm_p > mt_best[0]:
                        mt_best = (mlm_p, nll, cand)
                    continue

                if nll < best_nll or (
                    math.isclose(nll, best_nll, rel_tol=0.0, abs_tol=1e-6)
                    and mlm_p > best_mlm_prob
                ):
                    best_nll = nll
                    best_cand = cand
                    best_mlm_prob = mlm_p

            improve = base_nll - best_nll if best_cand is not None else float("-inf")
            improve_ratio = improve / base_nll if base_nll > 0 and best_cand is not None else float("-inf")

            passes_medical_nll_relax = False
            if best_cand is None or not (improve >= min_improve and improve_ratio >= min_improve_ratio):
                # 정식 NLL 채택 실패 시: 의료사전 표면 + MLM 하한 + NLL 악화 상한으로 재평가
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

            # 멀티토큰 트랙: 단일트랙이 확정한 best 가 없거나 NLL 게이트 미통과인 경우
            # mt_best (MLM 기반 조건 통과) 가 있으면 이를 채택한다. 단일트랙이 이미
            # 좋은 best 를 잡았더라도 멀티토큰이 의미적으로 더 맞을 수 있어서, NLL 이
            # 악화되지 않는 선에서 멀티토큰이 있으면 우선한다 (단일트랙은 잡힌 경우가
            # 곳은→가스는 같은 의료 과교정일 때가 많음).
            is_multi_only = False
            improve_precheck = base_nll - best_nll if best_cand is not None else float("-inf")
            ratio_precheck = (
                improve_precheck / base_nll if base_nll > 0 and best_cand is not None else float("-inf")
            )
            single_passes_nll = (
                best_cand is not None
                and improve_precheck >= min_improve
                and ratio_precheck >= min_improve_ratio
            )

            if mt_best is not None and not (single_passes_nll or passes_medical_nll_relax):
                # 단일트랙 실패 시 멀티토큰으로 대체
                best_mlm_prob, best_nll, best_cand = mt_best
                is_multi_only = True

            if best_cand is None:
                continue

            improve = base_nll - best_nll
            improve_ratio = improve / base_nll if base_nll > 0 else 0.0

            passes_nll = improve >= min_improve and improve_ratio >= min_improve_ratio
            # MLM 타이브레이크는 "의료사전 기반 후보" 용도. RoBERTa full-vocab jamo 경로로
            # 올라온 후보는 NLL 개선이 없으면 채택하지 않는다 (예: 없는→있는, 곳은→것은
            # 같은 의미 반전/수정 불요 케이스 차단).
            is_vocab_only = best_cand in vocab_only
            passes_mlm_tie = (
                not is_multi_only
                and not is_vocab_only
                and math.isclose(improve, 0.0, rel_tol=0.0, abs_tol=1e-5)
                and best_mlm_prob > orig_mlm_prob + 1e-15
                and best_mlm_prob >= _MLM_TIE_MIN_PROB
            )

            if not is_multi_only and not passes_nll and not passes_mlm_tie and not passes_medical_nll_relax:
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
                "multi_token": is_multi_only,
            })
            log_extra = ""
            if is_multi_only:
                log_extra = ", 멀티토큰"
            elif passes_mlm_tie and not passes_nll:
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
