"""
KM-BERT 교정 모듈 — 인코더 정적 임베딩 + 의료 사전 후보만 사용 (MLM 아님).

KU-RIAS 쿠리아스 가중치(MLM 헤드 포함)는 mlm_corrector.py 와 scripts/mlm_smoke.py 로 사용.
"""

import re
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from jamo import h2j, j2hcj
import Levenshtein

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "madatnlp/km-bert"

COMMON_SUFFIXES = (
    "입니다", "습니다", "합니다", "됩니다",
    "에요", "이요", "어요", "아요", "해요", "세요",
    "해서", "에서", "부터", "까지", "처럼", "만큼",
    "으로", "는데", "인데", "런데", "은데",
    "하고", "이고", "되고", "라고", "다고",
    "으면", "하면", "이면", "지만", "더니",
    "면서", "거든", "는지", "건지", "든지",
    "네요", "군요", "지요", "죠",
    "시고", "보고", "리고", "이며", "하며",
    "겠습니다", "겠어요", "더라고요", "더라구요",
)

SKIP_WORDS = {
    "오늘", "어제", "내일", "지금", "여기", "거기", "저기", "이번", "다음",
    "그래서", "그런데", "그리고", "하지만", "그러나", "또는", "혹은",
    "어디", "어떻게", "언제", "왜", "어디가", "이것", "저것", "그것",
    "왼쪽", "오른쪽", "위쪽", "아래쪽", "앞쪽", "뒤쪽",
    "많이", "조금", "아주", "매우", "너무", "좀", "더", "훨씬",
    "계단", "침대", "병원", "의자", "방", "집", "손을",
    "정도", "잠깐", "한번", "다시", "안정", "문제", "쪽에",
    "계속", "오세요", "보세요", "체한", "개를", "즉",
    "재보고", "해보고", "들어보고", "알아보고",
    "통증", "바로", "십분", "이십분", "십", "이십",
    "발리뼈", "발리뼈가", "꾀안이", "약풍이", "금식", "금식입니다",
    "터지면", "증인",
    "파요", "반동성", "반동통",
    "있으세요", "나오나요", "되나요", "하시나요",
    "골반", "골반이", "골절",
}

MEDICAL_COMPOUND_SUFFIXES = ("약", "제", "기", "용", "법", "술", "증", "량")

_KOREAN_RE = re.compile(r'[가-힣]')

PARTICLES_AND_ENDINGS = sorted([
    "이", "가", "을", "를", "은", "는", "에", "의", "와", "과",
    "로", "으로", "도", "만", "랑", "이랑", "까지", "부터", "마저",
    "조차", "밖에", "보다", "처럼", "같이", "만큼",
    "에서", "에게", "한테", "께서",
    "면", "면서", "고", "다", "다가", "지만", "거나",
    "어서", "아서", "니까", "서",
    "는데", "인데", "은데",
    "기도", "기는", "기에", "기가",
    "어요", "아요", "해요", "세요",
    "셨어요", "었어요", "았어요",
    "셨", "었", "았",
    "습니다", "입니다", "합니다",
    "하고", "이고", "되고",
], key=len, reverse=True)


def _to_jamo(text: str) -> str:
    try:
        return j2hcj(h2j(text))
    except Exception:
        return text


def _strip_suffix(word: str) -> tuple[str, str]:
    for suffix in sorted(COMMON_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)], suffix
    return word, ""


class BertCorrector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        confidence_threshold: float = 0.45,
        medical_terms: set[str] | None = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.medical_terms = medical_terms or set()

        logger.info(f"KM-BERT 모델 로딩 중: {model_name} (device: {self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.emb_weight = self.model.embeddings.word_embeddings.weight.detach().to(self.device)
        self.vocab = self.tokenizer.get_vocab()

        self._build_candidate_pool()
        logger.info("KM-BERT 모델 로딩 완료")

    def _build_candidate_pool(self):
        self.candidate_pool: dict[str, str] = {}

        for token in self.vocab:
            if token.startswith("##"):
                continue
            cleaned = token.strip()
            if len(cleaned) < 2 or not _KOREAN_RE.search(cleaned):
                continue
            self.candidate_pool[cleaned] = _to_jamo(cleaned)

        for term in self.medical_terms:
            if term not in self.candidate_pool:
                self.candidate_pool[term] = _to_jamo(term)

        self.pool_list: list[str] = []
        emb_list: list[torch.Tensor] = []

        for word in self.candidate_pool:
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                continue
            ids_tensor = torch.tensor(token_ids, device=self.device)
            emb = self.emb_weight[ids_tensor].mean(dim=0)
            self.pool_list.append(word)
            emb_list.append(emb)

        self.pool_matrix = F.normalize(torch.stack(emb_list), dim=-1)
        self.pool_jamos = [self.candidate_pool[w] for w in self.pool_list]

        logger.info(
            f"후보 풀 구축 완료: {len(self.pool_list)}개 단어 "
            f"(vocab 단어 + 사전 {len(self.medical_terms)}개)"
        )

    def _is_known_word(self, word: str) -> bool:
        tokens = self.tokenizer.tokenize(word)
        return len(tokens) == 1

    def _is_base_known(self, base: str) -> bool:
        if len(base) < 1:
            return False
        if self._is_known_word(base):
            return True
        return base in self.candidate_pool or base in SKIP_WORDS or base in self.medical_terms

    def _is_known_base_or_compound(self, word: str) -> bool:
        if self._is_base_known(word):
            return True

        if word in PARTICLES_AND_ENDINGS:
            return True

        for particle in PARTICLES_AND_ENDINGS:
            if word.endswith(particle) and len(word) > len(particle):
                base = word[:-len(particle)]
                if self._is_base_known(base):
                    return True
                if self._is_medical_compound(base):
                    return True

        for suffix in COMMON_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                base = word[:-len(suffix)]
                if self._is_base_known(base):
                    return True
                if self._is_medical_compound(base):
                    return True

        bases_to_check = self._multi_level_strip(word, max_depth=3)
        for base in bases_to_check:
            if self._is_base_known(base):
                return True
            if self._is_medical_compound(base):
                return True

        if self._is_medical_compound(word):
            return True

        return False

    def _is_medical_compound(self, word: str) -> bool:
        for term in self.medical_terms:
            if word.startswith(term) and len(word) > len(term):
                remainder = word[len(term):]
                if remainder in MEDICAL_COMPOUND_SUFFIXES or self._is_base_known(remainder):
                    return True
        return False

    def _multi_level_strip(self, word: str, max_depth: int = 3) -> list[str]:
        results: list[str] = []
        queue = [word]
        seen = {word}

        for _ in range(max_depth):
            next_queue: list[str] = []
            for w in queue:
                for particle in PARTICLES_AND_ENDINGS:
                    if w.endswith(particle) and len(w) > len(particle):
                        base = w[:-len(particle)]
                        if len(base) >= 1 and base not in seen:
                            seen.add(base)
                            results.append(base)
                            next_queue.append(base)
            queue = next_queue
            if not queue:
                break

        return results

    def _is_suspicious(self, word: str) -> bool:
        if word in SKIP_WORDS or word in self.medical_terms:
            return False

        stem, sfx = _strip_suffix(word)
        check = stem if sfx else word

        if check in self.medical_terms or check in SKIP_WORDS:
            return False

        if self._is_known_base_or_compound(word):
            return False

        if sfx and self._is_known_base_or_compound(stem):
            return False

        return True

    def _find_uncertain_words(
        self, text: str, jamo_changes: list[dict] | None = None,
    ) -> list[tuple[str, int, int]]:
        pattern = re.compile(r'[가-힣]+')
        candidates = []

        corrected_words = set()
        if jamo_changes:
            for c in jamo_changes:
                corrected_words.add(c.get("corrected", ""))

        for m in pattern.finditer(text):
            word = m.group()
            if len(word) < 2 or word in corrected_words:
                continue
            if not self._is_suspicious(word):
                continue
            candidates.append((word, m.start(), m.end()))

        return candidates

    def _get_static_embedding(self, word: str) -> torch.Tensor | None:
        token_ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            return None
        ids_tensor = torch.tensor(token_ids, device=self.device)
        return self.emb_weight[ids_tensor].mean(dim=0)

    def _find_candidates(
        self, word: str, max_jamo_distance: int = 4, top_k: int = 10,
    ) -> list[tuple[str, float, int, float]]:
        word_jamo = _to_jamo(word)
        word_emb = self._get_static_embedding(word)
        if word_emb is None:
            return []

        word_emb_norm = F.normalize(word_emb.unsqueeze(0), dim=-1)
        semantic_sims = torch.mm(word_emb_norm, self.pool_matrix.t())[0]

        results = []
        for i, (term, term_jamo) in enumerate(zip(self.pool_list, self.pool_jamos)):
            if term == word:
                continue

            if term not in self.medical_terms:
                continue

            if abs(len(term) - len(word)) > 2:
                continue

            jamo_dist = Levenshtein.distance(word_jamo, term_jamo)
            if jamo_dist == 0 or jamo_dist > max_jamo_distance:
                continue

            max_jamo_len = max(len(word_jamo), len(term_jamo))
            phonetic_sim = 1.0 - (jamo_dist / max_jamo_len) if max_jamo_len else 0.0
            semantic_sim = semantic_sims[i].item()

            score = 0.35 * max(semantic_sim, 0.0) + 0.65 * phonetic_sim
            results.append((term, score, jamo_dist, semantic_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def correct(
        self,
        text: str,
        jamo_changes: list[dict] | None = None,
        trace_out: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:
        changes = []
        uncertain = self._find_uncertain_words(text, jamo_changes)

        if trace_out is not None:
            trace_out.append(
                {
                    "kind": "bert_trace_header",
                    "model": self.model_name,
                    "method": "static_embedding_plus_jamo, candidates restricted to medical_dict terms only",
                    "confidence_threshold": self.confidence_threshold,
                    "uncertain_word_count": len(uncertain),
                }
            )

        if not uncertain:
            logger.info("[BERT] 교정 대상 단어 없음")
            return text, changes

        logger.info(f"[BERT] {len(uncertain)}개 의심 단어 발견")

        offset = 0
        result = text

        for word, start, end in uncertain:
            adj_start = start + offset
            adj_end = end + offset

            target_word = word
            suffix = ""
            stem, sfx = _strip_suffix(word)
            if sfx and len(stem) >= 2:
                target_word = stem
                suffix = sfx

            if not suffix:
                for particle in PARTICLES_AND_ENDINGS:
                    if word.endswith(particle) and len(word) > len(particle) and len(word) - len(particle) >= 2:
                        target_word = word[:-len(particle)]
                        suffix = particle
                        break

            candidates = self._find_candidates(target_word)

            trace_row: dict | None = None
            if trace_out is not None:
                trace_row = {
                    "kind": "uncertain_word",
                    "surface": word,
                    "char_span": [start, end],
                    "stem_for_matching": target_word,
                    "suffix_restored": suffix or "",
                    "candidates_top": [
                        {
                            "term": t,
                            "score": round(s, 4),
                            "jamo_distance": jd,
                            "semantic_similarity": round(sem, 4),
                        }
                        for t, s, jd, sem in candidates[:8]
                    ],
                }

            if not candidates:
                if trace_out is not None and trace_row is not None:
                    trace_row["decision"] = "no_candidates"
                    trace_row["reason"] = (
                        "의료 사전 용어 중 자모거리·길이 조건을 통과한 후보 없음"
                    )
                    trace_out.append(trace_row)
                continue

            best_term, best_score, best_dist, best_sem = candidates[0]

            if best_score >= self.confidence_threshold:
                corrected_word = best_term + suffix
                result = result[:adj_start] + corrected_word + result[adj_end:]
                offset += len(corrected_word) - len(word)

                changes.append({
                    "type": "bert_vocab_hybrid",
                    "original": word,
                    "corrected": corrected_word,
                    "confidence": round(best_score, 4),
                    "jamo_distance": best_dist,
                    "semantic_similarity": round(best_sem, 4),
                })
                if trace_out is not None and trace_row is not None:
                    trace_row["decision"] = "applied"
                    trace_row["chosen"] = corrected_word
                    trace_out.append(trace_row)
            else:
                if trace_out is not None and trace_row is not None:
                    trace_row["decision"] = "below_threshold"
                    trace_row["best_term"] = best_term
                    trace_row["best_score"] = round(best_score, 4)
                    trace_row["reason"] = (
                        f"최고 후보 '{best_term}' 점수 {best_score:.4f} < 임계값 {self.confidence_threshold}"
                    )
                    trace_out.append(trace_row)

        if changes:
            logger.info(f"[BERT] {len(changes)}건 교정 완료")
        else:
            logger.info("[BERT] 교정 건수 없음 (threshold 미달 또는 후보 없음)")

        return result, changes
