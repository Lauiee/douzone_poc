"""KoGPT2 기반 문맥(PPL) 보정."""

from __future__ import annotations

import logging
import re
import itertools

import Levenshtein
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from src.korean_text_utils import speech_endings_compatible

logger = logging.getLogger(__name__)

_KOREAN = re.compile(r"[가-힣]+")


class KoGPT2Corrector:
    def __init__(
        self,
        model_name: str = "skt/kogpt2-base-v2",
        proposal_model_name: str = "madatnlp/km-bert",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.proposal_tokenizer = AutoTokenizer.from_pretrained(proposal_model_name)
        self.proposal_model = AutoModelForMaskedLM.from_pretrained(proposal_model_name)
        self.proposal_model.eval()
        self.proposal_model.to(self.device)
        self._nll_cache: dict[str, float] = {}

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

    def _candidate_words(self, text: str, start: int, end: int, top_k: int) -> list[str]:
        mask_tok = self.proposal_tokenizer.mask_token
        mask_id = self.proposal_tokenizer.mask_token_id
        if mask_tok is None or mask_id is None:
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
        vals, idx = torch.topk(probs, k=k)
        words: list[str] = []
        seen = set()
        for i in range(k):
            tid = int(idx[i].item())
            tok = self.proposal_tokenizer.convert_ids_to_tokens(tid) or ""
            tok = tok[2:] if tok.startswith("##") else tok
            if not tok or not _KOREAN.fullmatch(tok):
                continue
            if len(tok) < 2:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            words.append(tok)
        return words

    def _insertion_candidates(self, text: str, pos: int, top_k: int) -> list[str]:
        """문맥상 pos 앞에 들어갈 단어 후보."""
        mask_tok = self.proposal_tokenizer.mask_token
        mask_id = self.proposal_tokenizer.mask_token_id
        if mask_tok is None or mask_id is None:
            return []
        masked = text[:pos] + mask_tok + text[pos:]
        enc = self.proposal_tokenizer(masked, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        pos_idx = (input_ids == mask_id).nonzero(as_tuple=True)
        if len(pos_idx[0]) != 1:
            return []
        midx = int(pos_idx[1][0])
        with torch.no_grad():
            out = self.proposal_model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0, midx]
        probs = F.softmax(logits.float(), dim=-1)
        k = min(top_k, int(probs.shape[0]))
        _, idx = torch.topk(probs, k=k)
        words: list[str] = []
        seen = set()
        for i in range(k):
            tid = int(idx[i].item())
            tok = self.proposal_tokenizer.convert_ids_to_tokens(tid) or ""
            tok = tok[2:] if tok.startswith("##") else tok
            if not tok or not _KOREAN.fullmatch(tok):
                continue
            if len(tok) < 2:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            words.append(tok)
        return words

    def _apply_word_replacements(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        replacement_map: dict[int, str],
    ) -> str:
        out = text
        for idx in sorted(replacement_map.keys(), reverse=True):
            start, end, _ = spans[idx]
            out = out[:start] + replacement_map[idx] + out[end:]
        return out

    def correct_text(
        self,
        text: str,
        top_k: int = 40,
        max_word_edit_distance: int = 1,
        min_improve: float = 0.06,
        min_span_chars: int = 2,
        span_words: int = 2,
        per_word_top_k: int = 6,
        max_combinations: int = 8,
        min_improve_ratio: float = 0.03,
    ) -> tuple[str, list[dict]]:
        self._nll_cache.clear()
        spans = self._word_spans(text)
        if not spans:
            return text, []
        spans.sort(key=lambda x: x[0])

        out = text
        changes: list[dict] = []
        i = 0
        while True:
            spans = self._word_spans(out)
            if i >= len(spans):
                break
            if len(spans[i][2]) < min_span_chars:
                i += 1
                continue

            left = max(0, i - (span_words // 2))
            right = min(len(spans), left + span_words)
            left = max(0, right - span_words)
            window_indices = list(range(left, right))
            if not window_indices:
                i += 1
                continue

            base_nll = self._sent_nll(out)
            per_word_options: list[list[str]] = []
            for wi in window_indices:
                s, e, w = spans[wi]
                opts = [w]
                if len(w) >= min_span_chars:
                    cands = self._candidate_words(out, s, e, top_k=top_k)
                    for cand in cands[:per_word_top_k]:
                        if cand == w:
                            continue
                        if abs(len(cand) - len(w)) > 1:
                            continue
                        if Levenshtein.distance(w, cand) > max_word_edit_distance:
                            continue
                        if not speech_endings_compatible(w, cand):
                            continue
                        opts.append(cand)
                    # 삽입 후보는 window 중심 토큰에만 허용(조합 폭발 방지)
                    center_wi = window_indices[len(window_indices) // 2]
                    if wi == center_wi:
                        ins = self._insertion_candidates(out, s, top_k=top_k)
                        for ins_word in ins[:2]:
                            if len(ins_word) < 2:
                                continue
                            opts.append(ins_word + w)
                # 중복 제거
                uniq = []
                seen = set()
                for o in opts:
                    if o not in seen:
                        seen.add(o)
                        uniq.append(o)
                per_word_options.append(uniq)

            all_product = itertools.product(*per_word_options)
            best_nll = base_nll
            best_repl: dict[int, str] = {}
            checked = 0
            for combo in all_product:
                if checked >= max_combinations:
                    break
                checked += 1
                repl = {}
                changed = False
                for j, wi in enumerate(window_indices):
                    original = spans[wi][2]
                    cand = combo[j]
                    if cand != original:
                        repl[wi] = cand
                        changed = True
                if not changed:
                    continue
                trial = self._apply_word_replacements(out, spans, repl)
                nll = self._sent_nll(trial)
                if nll < best_nll:
                    best_nll = nll
                    best_repl = repl

            improve = base_nll - best_nll
            improve_ratio = (improve / base_nll) if base_nll > 0 else 0.0
            if best_repl and improve >= min_improve and improve_ratio >= min_improve_ratio:
                before = out
                out = self._apply_word_replacements(out, spans, best_repl)
                new_spans = self._word_spans(out)
                # 변경 내역 기록
                for wi, cand in sorted(best_repl.items()):
                    os, oe, ow = spans[wi]
                    # 변경 후 동일 인덱스 기준으로 범위 기록 (근사)
                    ns, ne, _ = new_spans[min(wi, len(new_spans) - 1)]
                    changes.append(
                        {
                            "type": "kogpt2_ppl_span",
                            "start": ns,
                            "end": ne,
                            "original": ow,
                            "corrected": cand,
                            "nll_before": round(base_nll, 6),
                            "nll_after": round(best_nll, 6),
                            "improve": round(improve, 6),
                            "improve_ratio": round(improve_ratio, 6),
                            "window_before": before[spans[left][0]:spans[right - 1][1]],
                        }
                    )
                # 변경한 인덱스 근처부터 재탐색
                i = max(0, left - 1)
            else:
                i += 1

        return out, changes

