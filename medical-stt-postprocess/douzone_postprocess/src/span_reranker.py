"""
Span Reranker — 2~3 어절 span 단위 멀티MASK 교정

단어 단위 교정(kobert_context, kogpt2_ppl)이 못 잡는
토큰 경계 오류 및 구(phrase) 단위 오인식을 처리합니다.

동작:
1. 슬라이딩 윈도우로 2~3 어절 span 추출
2. span을 토큰 수만큼 [MASK] 연속으로 교체
3. 각 MASK 위치별 RoBERTa top-k 후보(토큰 id)
4. 조합별 span 문자열 = tokenizer.decode(해당 id들) → 원문에 삽입 → KoGPT2 NLL
5. 개선 시 채택 (한 pass에 최대 1건)
"""

from __future__ import annotations

import logging
import math
import re
from itertools import islice, product

import Levenshtein
import torch
import torch.nn.functional as F

from src.korean_text_utils import speech_endings_compatible

logger = logging.getLogger(__name__)

_KOREAN = re.compile(r"[가-힣]+")


class SpanReranker:
    def __init__(
        self,
        roberta_model,
        roberta_tokenizer,
        kogpt2_model,
        kogpt2_tokenizer,
        device: str = "cpu",
        medical_terms: set[str] | None = None,
        span_words: int = 2,
        per_mask_top_k: int = 5,
        max_combinations: int = 25,
        min_improve: float = 0.1,
        min_improve_ratio: float = 0.015,
        min_span_chars: int = 2,
        max_span_char_edit: int = 2,
    ):
        self.roberta_model = roberta_model
        self.roberta_tokenizer = roberta_tokenizer
        self.kogpt2_model = kogpt2_model
        self.kogpt2_tokenizer = kogpt2_tokenizer
        self.device = device
        self.medical_terms = medical_terms or set()
        self.span_words = span_words
        self.per_mask_top_k = per_mask_top_k
        self.max_combinations = max_combinations
        self.min_improve = min_improve
        self.min_improve_ratio = min_improve_ratio
        self.min_span_chars = min_span_chars
        self.max_span_char_edit = max_span_char_edit

        self._nll_cache: dict[str, float] = {}

        logger.info(
            "SpanReranker 초기화 완료 (span_words=%d, per_mask_top_k=%d, max_combinations=%d)",
            span_words,
            per_mask_top_k,
            max_combinations,
        )

    def _sent_nll(self, text: str) -> float:
        if text in self._nll_cache:
            return self._nll_cache[text]
        enc = self.kogpt2_tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        with torch.no_grad():
            out = self.kogpt2_model(
                input_ids=input_ids, attention_mask=attn, labels=input_ids
            )
        nll = float(out.loss.item())
        self._nll_cache[text] = nll
        return nll

    def _multi_mask_topk_scored(
        self,
        text: str,
        span_start: int,
        span_end: int,
    ) -> list[list[tuple[int, float]]] | None:
        """각 MASK 위치별 (토큰 id, softmax 확률) top-k. UNK 후보는 제외."""
        tok = self.roberta_tokenizer
        mask_token = tok.mask_token
        mask_id = tok.mask_token_id
        if not mask_token or mask_id is None:
            return None

        span_text = text[span_start:span_end]
        span_token_ids = tok.encode(span_text, add_special_tokens=False)
        n_masks = len(span_token_ids)
        if n_masks == 0 or n_masks > 6:
            return None

        mask_str = " ".join([mask_token] * n_masks)
        masked_text = text[:span_start] + mask_str + text[span_end:]

        enc = tok(masked_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)

        mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0].tolist()
        if len(mask_positions) != n_masks:
            return None

        with torch.no_grad():
            out = self.roberta_model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0]

        unk_id = tok.unk_token_id
        per_mask: list[list[tuple[int, float]]] = []
        for mi, mpos in enumerate(mask_positions):
            probs = F.softmax(logits[int(mpos)].float(), dim=-1)
            k = min(self.per_mask_top_k * 5, int(probs.shape[0]))
            vals, idx = torch.topk(probs, k=k)
            row: list[tuple[int, float]] = []
            seen: set[int] = set()
            for i in range(k):
                if len(row) >= self.per_mask_top_k:
                    break
                tid = int(idx[i].item())
                if unk_id is not None and tid == unk_id:
                    continue
                if tid in seen:
                    continue
                seen.add(tid)
                row.append((tid, float(vals[i].item())))
            # 원문 span의 해당 서브워드는 반드시 후보에 (UNK 제거로 1위가 빠지는 경우 대비)
            oid = int(span_token_ids[mi])
            if oid not in seen:
                p0 = float(probs[oid].item()) if 0 <= oid < probs.shape[0] else 1e-12
                row.insert(0, (oid, p0))
                seen.add(oid)
            row = row[: self.per_mask_top_k]
            if not row:
                return None
            per_mask.append(row)

        return per_mask

    def _word_spans(self, text: str) -> list[tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]

    def _korean_only_span(self, span_text: str) -> bool:
        return bool(_KOREAN.search(span_text))

    def correct_text(self, text: str) -> tuple[str, list[dict]]:
        self._nll_cache.clear()
        out = text
        changes: list[dict] = []

        word_spans = self._word_spans(out)
        if len(word_spans) < self.span_words:
            return out, changes

        window_groups: list[tuple[int, int, str]] = []
        for i in range(len(word_spans) - self.span_words + 1):
            group = word_spans[i : i + self.span_words]
            span_start = group[0][0]
            span_end = group[-1][1]
            span_text = out[span_start:span_end]
            if len(span_text) < self.min_span_chars:
                continue
            if not self._korean_only_span(span_text):
                continue
            window_groups.append((span_start, span_end, span_text))

        tok = self.roberta_tokenizer
        base_nll = self._sent_nll(out)

        # 윈도우별로 (NLL개선 × √MLM조합확률)이 큰 조합을 고른 뒤, 전역 최고 점수 1건 적용
        best_global: tuple[float, int, int, str, str, float, float] | None = None

        for span_start, span_end, span_text in window_groups:
            per_mask_scored = self._multi_mask_topk_scored(out, span_start, span_end)
            if not per_mask_scored:
                continue

            raw_combos = list(product(*per_mask_scored))
            scored_order = sorted(
                raw_combos,
                key=lambda rows: -math.prod(p for _, p in rows),
            )
            combo_iter = islice(scored_order, self.max_combinations)

            best_combo_score = -1.0
            local_best_cand: str | None = None
            local_best_nll = base_nll

            for combo in combo_iter:
                ids_list = [int(t) for t, _ in combo]
                mlm_joint = math.prod(p for _, p in combo)
                cand_span = tok.convert_tokens_to_string(
                    tok.convert_ids_to_tokens(ids_list)
                )
                if not cand_span or cand_span.strip() == span_text.strip():
                    continue
                if not speech_endings_compatible(span_text, cand_span):
                    continue
                if Levenshtein.distance(span_text, cand_span) > self.max_span_char_edit:
                    continue

                trial = out[:span_start] + cand_span + out[span_end:]
                nll = self._sent_nll(trial)
                improve = base_nll - nll
                improve_ratio = improve / base_nll if base_nll > 0 else 0.0
                if improve < self.min_improve or improve_ratio < self.min_improve_ratio:
                    continue

                combo_score = improve * math.sqrt(mlm_joint)
                if combo_score > best_combo_score:
                    best_combo_score = combo_score
                    local_best_cand = cand_span
                    local_best_nll = nll

            if local_best_cand is None:
                continue

            improve = base_nll - local_best_nll
            if best_global is None or best_combo_score > best_global[0]:
                best_global = (
                    best_combo_score,
                    span_start,
                    span_end,
                    span_text,
                    local_best_cand,
                    local_best_nll,
                    improve,
                )

        if best_global is not None:
            _, span_start, span_end, span_text, best_cand_span, best_nll, improve = best_global
            improve_ratio = improve / base_nll if base_nll > 0 else 0.0
            out = out[:span_start] + best_cand_span + out[span_end:]
            changes.append(
                {
                    "type": "span_rerank",
                    "start": span_start,
                    "end": span_end,
                    "original": span_text,
                    "corrected": best_cand_span,
                    "nll_before": round(base_nll, 6),
                    "nll_after": round(best_nll, 6),
                    "improve": round(improve, 6),
                    "improve_ratio": round(improve_ratio, 6),
                }
            )
            logger.info(
                "[SpanReranker] '%s' → '%s' (NLL개선 %.4f)",
                span_text,
                best_cand_span,
                improve,
            )

        changes.reverse()
        return out, changes
