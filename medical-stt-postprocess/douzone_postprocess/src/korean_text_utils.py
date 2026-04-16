"""Korean surface utilities shared across correction modules."""

from __future__ import annotations

# Mecab 없이 용언 활용 어미만 보고 STT 오인식이 아닌 문법형으로 간주 (과교정 방지).
# 긴 어미를 먼저 매칭하도록 정렬해 둔다.
VERB_FORM_ENDINGS: frozenset[str] = frozenset(
    {
        "면",
        "고",
        "서",
        "며",
        "니",
        "지",
        "도",
        "만",
        "으면",
        "아서",
        "어서",
        "아도",
        "어도",
        "시고",
        "시면",
        "시며",
        "시어",
        "셔서",
        "하면",
        "하고",
        "하며",
        "해서",
        "해",
        "나요",
        "나면",
        "으나",
        "게",
    }
)
_VERB_FORM_ENDINGS_SORTED: tuple[str, ...] = tuple(
    sorted(VERB_FORM_ENDINGS, key=len, reverse=True)
)


def looks_like_verb_conjugation(surface: str) -> bool:
    """용언 활용형으로 보이면 True — STT 교정 스팬에서 제외."""
    for ending in _VERB_FORM_ENDINGS_SORTED:
        if len(surface) > len(ending) and surface.endswith(ending):
            return True
    return False


JOSA_LIST = sorted(
    [
        "으로",
        "에서",
        "에게",
        "한테",
        "까지",
        "부터",
        "이랑",
        "을",
        "를",
        "이",
        "가",
        "은",
        "는",
        "에",
        "의",
        "와",
        "과",
        "도",
        "만",
        "로",
        "랑",
    ],
    key=len,
    reverse=True,
)


def split_josa(surface: str) -> tuple[str, str]:
    for josa in JOSA_LIST:
        if surface.endswith(josa) and len(surface) > len(josa):
            return surface[: -len(josa)], josa
    return surface, ""


def remove_josa(surface: str) -> str:
    stem, _ = split_josa(surface)
    return stem


def extract_josa(surface: str) -> str:
    _, josa = split_josa(surface)
    return josa


def _has_batchim(ch: str) -> bool:
    code = ord(ch)
    if code < 0xAC00 or code > 0xD7A3:
        return False
    return ((code - 0xAC00) % 28) != 0


def harmonize_josa(stem: str, original_josa: str) -> str:
    """Recompute common josa variants with/without batchim.

    Keeps unknown josa unchanged.
    """
    if not stem or not original_josa:
        return original_josa

    last = stem[-1]
    has_batchim = _has_batchim(last)

    if original_josa in ("을", "를"):
        return "을" if has_batchim else "를"
    if original_josa in ("이", "가"):
        return "이" if has_batchim else "가"
    if original_josa in ("은", "는"):
        return "은" if has_batchim else "는"
    if original_josa in ("과", "와"):
        return "과" if has_batchim else "와"
    if original_josa in ("으로", "로"):
        # 받침 없거나 ㄹ받침이면 "로"
        if not has_batchim:
            return "로"
        jong = (ord(last) - 0xAC00) % 28
        return "로" if jong == 8 else "으로"
    return original_josa


# [가-힣]+ 토큰 끝의 **문장 화법**에 가까운 접미사 (긴 것부터).
# '나요' 등은 '하나요'에 오탐하므로 넣지 않는다.
SPEECH_ENDING_SUFFIXES: tuple[str, ...] = (
    "습니다",
    "습니까",
    "입니까",
    "았습니다",
    "었습니다",
    "겠습니다",
    "을까요",
    "ㄹ까요",
    "는가요",
    "으세요",
    "세요",
    "었어요",
    "았어요",
    "겠어요",
    "어서",
    "아서",
    "해서",
    "네요",
    "시죠",
    "어요",
    "아요",
    "여요",
    "예요",
    "이에요",
    "에요",
    "해요",
    "지요",
    "죠",
    "요",
    "다",
    "까",
)


def extract_trailing_speech_ending(surface: str) -> str:
    """토큰 끝에서 가장 긴 화법 접미사 한 덩어리. 없으면 빈 문자열."""
    if not surface:
        return ""
    for suf in SPEECH_ENDING_SUFFIXES:
        if len(surface) > len(suf) and surface.endswith(suf):
            return suf
    return ""


def speech_endings_compatible(original: str, corrected: str) -> bool:
    """교정 전·후 종결/화법 꼬리가 같을 때만 True (다르면 EMR에서 과교정으로 간주)."""
    return extract_trailing_speech_ending(original) == extract_trailing_speech_ending(corrected)
