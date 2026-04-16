"""
자모 기반 교정 모듈
- jamo 라이브러리로 한글을 자모로 분해
- python-Levenshtein으로 편집거리 계산
- 의료 용어 사전 내 가장 가까운 단어로 교정
"""

import re
import logging
from pathlib import Path

from jamo import h2j, j2hcj
import Levenshtein

logger = logging.getLogger(__name__)

DEFAULT_DICT_PATH = Path(__file__).parent.parent / "data" / "medical_dict.txt"

COMMON_STOPWORDS = {
    "오늘", "어제", "내일", "지금", "여기", "거기", "저기",
    "그래서", "그런데", "그리고", "하지만", "그러나", "또는", "혹은",
    "이것", "저것", "그것", "어디", "어떻게", "언제", "왜",
    "하고", "해요", "해서", "하면", "인데", "있는데", "없는데",
    "다시", "아직", "이미", "벌써", "항상", "자주", "가끔",
    "있어요", "없어요", "해요", "봐요", "가요", "와요",
    "들어", "나가", "올라", "내려", "하시고", "드리고",
    "같은데", "같아요", "정도", "때문에", "대해서",
    "안정", "문제", "검사", "결과", "상태", "가능",
    "하시고", "해보고", "재보고", "해보시죠",
    "눌러보겠습니다", "누워보세요", "울리면서", "힘들어요",
    "아파서", "아파요", "뚜렷하네요",
    "불편하셔서", "오셨어요", "오르내리면", "두근거리면서",
    "괜찮더라고요", "먹다", "해요", "있는지",
    "걷기도", "저녁부터", "같더니", "들어오세요",
    "아프신가요", "좀", "때", "것", "수", "중",
    "있고", "없고", "하고", "보고", "되고",
    "통증", "바로", "십분", "이십분",
    "누를", "파요", "이번", "물도", "터지면",
    # 행정/접수 문맥 단어는 의료 용어로 오교정되기 쉬워 제외
    "수속", "수속하면", "입원", "입원수속"
}


def load_medical_dict(dict_path: str | Path | None = None) -> set[str]:
    path = Path(dict_path) if dict_path else DEFAULT_DICT_PATH
    terms = set()

    if not path.exists():
        logger.warning(f"의료 용어 사전 파일을 찾을 수 없습니다: {path}")
        return terms

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for term in line.split(","):
                term = term.strip()
                if term:
                    terms.add(term)

    logger.info(f"의료 용어 사전 로드 완료: {len(terms)}개 용어")
    return terms


def to_jamo(text: str) -> str:
    try:
        return j2hcj(h2j(text))
    except Exception:
        return text


def _extract_words_with_positions(text: str) -> list[tuple[str, int, int]]:
    pattern = re.compile(r'[가-힣]+')
    return [(m.group(), m.start(), m.end()) for m in pattern.finditer(text)]


COMMON_PARTICLES = sorted([
    "이", "가", "을", "를", "은", "는", "에", "의", "와", "과",
    "로", "으로", "도", "만", "랑", "이랑", "까지", "부터", "마저",
    "조차", "밖에", "보다", "에서", "에게", "한테",
], key=len, reverse=True)

COMMON_SUFFIXES = (
    "입니다", "습니다", "합니다",
    "에요", "이요", "어요", "아요", "해요", "세요", "해서",
    "에서", "부터", "까지", "처럼", "만큼", "으로", "는데",
    "인데", "런데", "은데", "하고", "이고", "되고",
    "으면", "하면", "이면", "니다",
    "시고", "보고", "리고", "이며", "하며",
    "면서", "지만", "더니", "거든", "는지", "건지",
    "네요", "군요", "지요", "는데요", "거든요",
)


def _has_common_suffix(word: str) -> bool:
    return any(word.endswith(s) for s in COMMON_SUFFIXES)


def _strip_suffix(word: str) -> tuple[str, str]:
    for suffix in sorted(COMMON_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)], suffix
    return word, ""


class JamoCorrector:
    def __init__(
        self,
        dict_path: str | Path | None = None,
        max_edit_distance: int = 2,
        min_term_length: int = 2,
    ):
        self.medical_terms = load_medical_dict(dict_path)
        self.max_edit_distance = max_edit_distance
        self.min_term_length = min_term_length

        self._jamo_cache: dict[str, str] = {}
        for term in self.medical_terms:
            self._jamo_cache[term] = to_jamo(term)

    def _should_skip(self, word: str) -> bool:
        if word in COMMON_STOPWORDS:
            return True
        if word in self.medical_terms:
            return True

        for particle in COMMON_PARTICLES:
            if word.endswith(particle) and len(word) > len(particle):
                base = word[:-len(particle)]
                if base in self.medical_terms or base in COMMON_STOPWORDS:
                    return True

        stem, suffix = _strip_suffix(word)
        if stem in COMMON_STOPWORDS or stem in self.medical_terms:
            return True
        if _has_common_suffix(word) and len(stem) < self.min_term_length:
            return True
        return False

    def _find_best_match(self, word: str) -> tuple[str | None, int]:
        if word in self.medical_terms:
            return word, 0

        if len(word) < self.min_term_length:
            return None, -1

        word_jamo = to_jamo(word)
        best_match = None
        best_distance = self.max_edit_distance + 1

        for term, term_jamo in self._jamo_cache.items():
            if abs(len(term) - len(word)) > self.max_edit_distance:
                continue

            len_ratio = min(len(term), len(word)) / max(len(term), len(word))
            if len_ratio < 0.5:
                continue

            distance = Levenshtein.distance(word_jamo, term_jamo)

            jamo_len = max(len(word_jamo), len(term_jamo))
            similarity = 1.0 - (distance / jamo_len) if jamo_len > 0 else 0.0

            min_sim = 0.58 if distance >= 2 else 0.48
            if distance <= self.max_edit_distance and distance < best_distance and similarity >= min_sim:
                best_distance = distance
                best_match = term

        if best_match is not None:
            return best_match, best_distance
        return None, -1

    def correct(self, text: str) -> tuple[str, list[dict]]:
        changes = []
        words_with_pos = _extract_words_with_positions(text)

        offset = 0
        result = text

        for word, start, end in words_with_pos:
            if len(word) < self.min_term_length:
                continue

            if self._should_skip(word):
                continue

            best_match, distance = self._find_best_match(word)

            if best_match and distance > 0:
                adj_start = start + offset
                adj_end = end + offset
                result = result[:adj_start] + best_match + result[adj_end:]
                offset += len(best_match) - len(word)

                changes.append({
                    "type": "jamo_correction",
                    "original": word,
                    "corrected": best_match,
                    "edit_distance": distance,
                })
                continue

            stem, suffix = _strip_suffix(word)
            if suffix and len(stem) >= self.min_term_length:
                stem_match, stem_dist = self._find_best_match(stem)
                if stem_match and stem_dist > 0:
                    corrected_word = stem_match + suffix
                    adj_start = start + offset
                    adj_end = end + offset
                    result = result[:adj_start] + corrected_word + result[adj_end:]
                    offset += len(corrected_word) - len(word)

                    changes.append({
                        "type": "jamo_correction",
                        "original": word,
                        "corrected": corrected_word,
                        "edit_distance": stem_dist,
                        "stem_correction": f"{stem} → {stem_match}",
                    })

        if changes:
            logger.info(f"[Jamo] {len(changes)}건 교정 완료")
            for c in changes:
                logger.debug(
                    f"  '{c['original']}' → '{c['corrected']}' "
                    f"(편집거리: {c['edit_distance']})"
                )

        return result, changes
