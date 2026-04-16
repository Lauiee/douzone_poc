"""
Rule-based 후처리 모듈
- 한글 숫자 → 아라비아 숫자 변환
- 의료 단위 정규화
"""

import re
import logging
import Levenshtein
from src.jamo_corrector import to_jamo
from src.korean_text_utils import harmonize_josa

logger = logging.getLogger(__name__)

KOREAN_DIGIT = {
    "영": 0, "공": 0,
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}

KOREAN_UNIT_MULTIPLIER = {
    "십": 10,
    "백": 100,
    "천": 1000,
    "만": 10000,
}

PURE_KOREAN_NUM = {
    "하나": 1, "한": 1,
    "두": 2, "둘": 2,
    "세": 3, "셋": 3,
    "네": 4, "넷": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
    "스물": 20, "스무": 20,
    "서른": 30,
    "마흔": 40,
    "쉰": 50,
    "예순": 60,
    "일흔": 70,
    "여든": 80,
    "아흔": 90,
}

MEDICAL_UNITS = {
    "씨씨": "cc",
    "밀리리터": "mL",
    "밀리그램": "mg",
    "마이크로그램": "μg",
    "그램": "g",
    "킬로그램": "kg",
    "센티미터": "cm",
    "밀리미터": "mm",
    "리터": "L",
    "데시리터": "dL",
    "퍼센트": "%",
    "밀리몰": "mmol",
}

SAFE_UNITS = {"시간", "개월", "캡슐", "밀리그램", "밀리리터",
              "센티미터", "밀리미터", "퍼센트", "씨씨", "리터"}

CONTEXT_UNITS = {"분", "년", "살", "번", "회", "개", "알", "정", "주", "일"}

KOREAN_PARTICLES = set("을를이가은는에서도의와과만부터까지로")

ABBREVIATION_MAP = {
    "HTN": "고혈압",
    "DM": "당뇨",
    "COPD": "만성폐쇄성폐질환",
    "MI": "심근경색",
    "AF": "심방세동",
    "CHF": "심부전",
    "CKD": "만성신장질환",
    "GERD": "역류성식도염",
    "DVT": "심부정맥혈전증",
    "PE": "폐색전증",
    "TB": "결핵",
    "URI": "상기도감염",
    "UTI": "요로감염",
    "CVA": "뇌졸중",
    "TIA": "일과성허혈발작",
    "ACS": "급성관상동맥증후군",
    "BPH": "전립선비대증",
    "RA": "류마티스관절염",
    "OA": "골관절염",
    "ICH": "뇌내출혈",
    "SAH": "지주막하출혈",
}

BODY_PART_TERMS = {
    "손", "발", "팔", "다리", "배", "복부", "허리", "등", "가슴", "목", "어깨", "무릎",
    "손목", "손가락", "발목", "발가락", "머리", "눈", "코", "입", "귀", "턱", "옆구리",
}

BODY_PART_STT_MAP = {
    "소": "손",
    "개": "배",
}


def _parse_sino_korean_number(text: str) -> int | None:
    """한자어 수사 문자열을 숫자로 변환한다."""
    if not text:
        return None

    result = 0
    current = 0
    i = 0
    matched_any = False

    while i < len(text):
        found = False
        chunk = text[i]

        if chunk in KOREAN_DIGIT:
            current = KOREAN_DIGIT[chunk]
            i += 1
            found = True
            matched_any = True
        elif chunk in KOREAN_UNIT_MULTIPLIER:
            multiplier = KOREAN_UNIT_MULTIPLIER[chunk]
            if current == 0:
                current = 1
            result += current * multiplier
            current = 0
            i += 1
            found = True
            matched_any = True

        if not found:
            return None

    result += current

    if not matched_any:
        return None
    return result


def _parse_pure_korean_number(text: str) -> int | None:
    """고유어 수사 문자열을 숫자로 변환한다."""
    if not text:
        return None

    result = 0
    remaining = text
    matched_any = False

    sorted_keys = sorted(PURE_KOREAN_NUM.keys(), key=len, reverse=True)

    while remaining:
        found = False
        for key in sorted_keys:
            if remaining.startswith(key):
                result += PURE_KOREAN_NUM[key]
                remaining = remaining[len(key):]
                found = True
                matched_any = True
                break
        if not found:
            return None

    return result if matched_any else None


def _is_word_boundary_before(text: str, pos: int) -> bool:
    if pos == 0:
        return True
    prev = text[pos - 1]
    return not prev.isalpha() or prev in ' \t\n' or prev in '.,?!;:()[]{}"\''


def _is_valid_unit_end(text: str, end: int) -> bool:
    if end >= len(text):
        return True
    next_char = text[end]
    if next_char in KOREAN_PARTICLES:
        return True
    if re.match(r'[\s.,?!;:()[\]{}"\'·]', next_char):
        return True
    if re.match(r'[가-힣]', next_char):
        return False
    return True


def apply_stt_phrase_fixes(text: str) -> tuple[str, list[dict]]:
    """레거시: 파이프라인에서 미사용. 테스트/스크립트 호환용."""
    return text, []


def fix_bodypart_context_phrases(text: str) -> tuple[str, list[dict]]:
    """의료 행위 문맥(__를/을 뗄 때)에서 신체부위 후보를 보수적으로 보정."""
    changes: list[dict] = []
    pattern = re.compile(r"(?P<noun>[가-힣]{1,3})(?P<josa>[을를])\s*뗄\s*때")

    def repl(m: re.Match) -> str:
        noun = m.group("noun")
        josa = m.group("josa")
        if noun in BODY_PART_TERMS:
            return m.group(0)

        chosen = BODY_PART_STT_MAP.get(noun)
        if chosen is None:
            # 보수적 fallback: 가장 가까운 신체부위가 명확할 때만.
            n_jamo = to_jamo(noun)
            best = None
            best_dist = 10**9
            tied = False
            for bp in BODY_PART_TERMS:
                d = Levenshtein.distance(n_jamo, to_jamo(bp))
                if d < best_dist:
                    best_dist = d
                    best = bp
                    tied = False
                elif d == best_dist:
                    tied = True
            if best is None or tied or best_dist > 2:
                return m.group(0)
            chosen = best

        chosen_josa = harmonize_josa(chosen, josa)
        corrected = f"{chosen}{chosen_josa} 뗄 때"
        changes.append(
            {
                "type": "bodypart_context_rule",
                "original": m.group(0),
                "corrected": corrected,
            }
        )
        return corrected

    return pattern.sub(repl, text), changes


def fix_stt_homophone_in_context(text: str) -> tuple[str, list[dict]]:
    """STT 유사음 오인식을 **좁은 문맥**에서만 고친다 (`medical_confusion` 같은 전역 고정 치환 없이)."""
    changes: list[dict] = []
    out = text

    # 시간 부사: "오늘 마침에 왔어요" 등
    _machim = re.compile(r"(오늘|어제|그제|내일)\s+마침에")

    def _repl_machim(m: re.Match) -> str:
        new = f"{m.group(1)} 아침에"
        changes.append(
            {
                "type": "stt_homophone_context",
                "original": m.group(0),
                "corrected": new,
            }
        )
        return new

    out = _machim.sub(_repl_machim, out)

    # 방향·위치: "어느 조이", "왼쪽 조이가" 등 → 쪽이
    _jjoi = re.compile(
        r"(어느|왼쪽|오른쪽|이|그|저)\s+조이(?=\s|[이가은는을를도만]|[.,?!…]|$)"
    )

    def _repl_jjoi(m: re.Match) -> str:
        new = f"{m.group(1)} 쪽이"
        changes.append(
            {
                "type": "stt_homophone_context",
                "original": m.group(0),
                "corrected": new,
            }
        )
        return new

    out = _jjoi.sub(_repl_jjoi, out)

    # 이름·호칭에 가까운 표면: 법률 "판결이 내려" 등은 제외
    _pan = re.compile(r"판결(?=이(?:라|고|래|네|었|는|은|랑|만|가|을|를))")

    def _repl_pan(m: re.Match) -> str:
        changes.append(
            {
                "type": "stt_homophone_context",
                "original": m.group(0),
                "corrected": "한결",
            }
        )
        return "한결"

    out = _pan.sub(_repl_pan, out)

    # 원인·탓 표현에서 STT가 "타신"으로 잡힌 경우
    _tasin = re.compile(
        r"(스트레스|과로|피로|수면|약|술|담배|유전|환경|식습관|과식|운동부족)\s+타신\s+(거|것)"
    )

    def _repl_tasin(m: re.Match) -> str:
        new = f"{m.group(1)} 탓인 {m.group(2)}"
        changes.append(
            {
                "type": "stt_homophone_context",
                "original": m.group(0),
                "corrected": new,
            }
        )
        return new

    out = _tasin.sub(_repl_tasin, out)

    _tasin2 = re.compile(r"(때문에|덕분에)\s+타신\s+(거|것)")

    def _repl_tasin2(m: re.Match) -> str:
        new = f"{m.group(1)} 탓인 {m.group(2)}"
        changes.append(
            {
                "type": "stt_homophone_context",
                "original": m.group(0),
                "corrected": new,
            }
        )
        return new

    out = _tasin2.sub(_repl_tasin2, out)

    return out, changes


def normalize_numbers(text: str) -> tuple[str, list[dict]]:
    changes = []

    _han_sip = re.compile(r"(?<![가-힣])(?:한|하나)\s+십\s*분")
    _han_isip = re.compile(r"(?<![가-힣])(?:한|하나)\s+이십\s*분")

    def _han_repl(m: re.Match, rep: str) -> str:
        changes.append({
            "type": "pure_plus_sino_minutes",
            "original": m.group(0),
            "corrected": rep,
        })
        return rep

    text = _han_sip.sub(lambda m: _han_repl(m, "10분"), text)
    text = _han_isip.sub(lambda m: _han_repl(m, "20분"), text)

    bp_pattern = re.compile(
        r'(?P<sys>[일이삼사오육칠팔구십백천]+)\s*에\s*(?P<dia>[일이삼사오육칠팔구십백천]+)'
    )

    def bp_replacer(m):
        sys_val = _parse_sino_korean_number(m.group("sys"))
        dia_val = _parse_sino_korean_number(m.group("dia"))
        if sys_val is not None and dia_val is not None:
            replacement = f"{sys_val}/{dia_val}"
            changes.append({
                "type": "blood_pressure",
                "original": m.group(0),
                "corrected": replacement,
            })
            return replacement
        return m.group(0)

    text = bp_pattern.sub(bp_replacer, text)

    sino_chars = r'[일이삼사오육칠팔구십백천만]'
    all_units = sorted(SAFE_UNITS | CONTEXT_UNITS, key=len, reverse=True)
    unit_list = '|'.join(all_units)

    multi_char_pattern = re.compile(
        rf'(?P<num>{sino_chars}{{2,}})\s*(?P<unit>{unit_list})'
    )

    def multi_char_replacer(m):
        num_val = _parse_sino_korean_number(m.group("num"))
        if num_val is not None and _is_valid_unit_end(text, m.end()):
            replacement = f"{num_val}{m.group('unit')}"
            changes.append({
                "type": "sino_number_unit",
                "original": m.group(0),
                "corrected": replacement,
            })
            return replacement
        return m.group(0)

    text = multi_char_pattern.sub(multi_char_replacer, text)

    single_char_pattern = re.compile(
        rf'(?P<num>{sino_chars})\s+(?P<unit>{unit_list})'
    )

    def _single_char_replacer_with_boundary(m):
        if not _is_word_boundary_before(text, m.start()):
            return m.group(0)
        if not _is_valid_unit_end(text, m.end()):
            return m.group(0)
        num_val = _parse_sino_korean_number(m.group("num"))
        if num_val is not None:
            replacement = f"{num_val}{m.group('unit')}"
            changes.append({
                "type": "sino_number_unit",
                "original": m.group(0),
                "corrected": replacement,
            })
            return replacement
        return m.group(0)

    text = single_char_pattern.sub(_single_char_replacer_with_boundary, text)

    pure_kr_keys = '|'.join(sorted(PURE_KOREAN_NUM.keys(), key=len, reverse=True))
    pure_unit_pattern = re.compile(
        rf'(?P<num>(?:{pure_kr_keys})(?:{pure_kr_keys})?)\s+(?P<unit>{unit_list})'
    )

    def pure_unit_replacer(m):
        if not _is_valid_unit_end(text, m.end()):
            return m.group(0)
        # "한 번"은 관용(한 번에, 한 번만 …)으로 자주 쓰여 숫자(1번)로 바꾸지 않는다.
        if m.group("num") == "한" and m.group("unit") == "번":
            return m.group(0)
        num_val = _parse_pure_korean_number(m.group("num"))
        if num_val is not None:
            replacement = f"{num_val}{m.group('unit')}"
            changes.append({
                "type": "pure_korean_number_unit",
                "original": m.group(0),
                "corrected": replacement,
            })
            return replacement
        return m.group(0)

    text = pure_unit_pattern.sub(pure_unit_replacer, text)

    return text, changes


def normalize_medical_units(text: str) -> tuple[str, list[dict]]:
    changes = []
    for korean, standard in sorted(MEDICAL_UNITS.items(), key=lambda x: len(x[0]), reverse=True):
        if korean in text:
            changes.append({
                "type": "medical_unit",
                "original": korean,
                "corrected": standard,
            })
            text = text.replace(korean, standard)
    return text, changes


def expand_abbreviations(text: str) -> tuple[str, list[dict]]:
    changes = []
    for abbr, full in ABBREVIATION_MAP.items():
        pattern = re.compile(rf'\b{re.escape(abbr)}\b', re.IGNORECASE)
        if pattern.search(text):
            changes.append({
                "type": "abbreviation",
                "original": abbr,
                "corrected": full,
            })
            text = pattern.sub(full, text)
    return text, changes


def apply_rule_based(text: str) -> tuple[str, list[dict]]:
    """숫자·단위 정규화, 신체 문맥 룰, STT 유사음(좁은 문맥)만 수행한다."""
    all_changes = []

    text, changes = normalize_numbers(text)
    all_changes.extend(changes)

    text, changes = normalize_medical_units(text)
    all_changes.extend(changes)

    text, changes = fix_bodypart_context_phrases(text)
    all_changes.extend(changes)

    text, changes = fix_stt_homophone_in_context(text)
    all_changes.extend(changes)

    if all_changes:
        logger.info(f"[Rule-based] {len(all_changes)}건 교정 완료")
        for c in all_changes:
            logger.debug(f"  [{c['type']}] '{c['original']}' → '{c['corrected']}'")

    return text, all_changes
