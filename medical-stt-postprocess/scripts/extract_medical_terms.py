#!/usr/bin/env python3
"""Training_medical.json에서 의료 엔티티를 추출해 사전에 병합한다."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TARGET_TYPES = {
    "TMM_DISEASE",
    "TMM_SYMPTOM",
    "TMM_DRUG",
    "TM_CELL_TISSUE_ORGAN",
    "TR_MED_PROCEDURE",
    "TR_MED_MEASURE",
    "AFW_MED_DEVICE",
}

ONLY_DIGITS_RE = re.compile(r"^\d+$")
ONLY_SPECIAL_RE = re.compile(r"^[^0-9A-Za-z가-힣]+$")
KOREAN_SPACE_RE = re.compile(r"^[가-힣 ]+$")
TRAILING_JOSA_SUFFIXES = (
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "과",
    "와",
    "로",
)
TOKEN_LEVEL_STOPWORDS = {
    "의",
    "에",
    "에서",
    "에게",
    "께",
    "한테",
    "보다",
    "처럼",
    "만",
    "도",
    "과",
    "와",
    "로",
    "으로",
    "를",
    "을",
    "가",
    "이",
    "은",
    "는",
}
DESCRIPTIVE_PREFIXES = (
    "가벼운",
    "가쁜",
    "가는",
    "가능성",
    "가볍게",
    "가려운",
    "조이는",
    "심한",
    "심하게",
    "심한",
    "약한",
    "강한",
    "큰",
    "작은",
    "심하지",
)
REJECT_LAST_TOKENS = {
    "결과",
    "도구",
    "프로그램",
    "방법",
    "방식",
    "느낌",
    "듯",
    "이상",
}
REJECT_CONTAINS_TOKENS = {
    "결리고",
    "아프다",
    "아프고",
    "화한",
    "동반한",
    "의한",
    "위한",
    "이용한",
    "나타나는",
    "보이는",
    "되는",
    "된다",
    "된",
    "하는",
    "하며",
    "하고",
}
REJECT_SUBSTRINGS = (
    "방법관절",
    "방법넙다리",
    "방법무릎",
    "방법빗장",
    "방법위팔",
    "방법정강",
    "갑상선기능항진증갑상선기능항진증",
    "관상동맥 우회술관상동맥우회술",
)
MAX_TERM_CHARS = 18


def normalize_entity(entity: str, korean_only: bool) -> str | None:
    entity = " ".join(entity.split()).strip()
    if len(entity.replace(" ", "")) < 2:
        return None
    if len(entity.replace(" ", "")) > MAX_TERM_CHARS:
        return None
    if ONLY_DIGITS_RE.fullmatch(entity):
        return None
    if ONLY_SPECIAL_RE.fullmatch(entity):
        return None
    if korean_only and not KOREAN_SPACE_RE.fullmatch(entity):
        return None
    if entity.startswith(DESCRIPTIVE_PREFIXES):
        return None
    if any(fragment in entity for fragment in REJECT_SUBSTRINGS):
        return None

    tokens = entity.split()
    if len(tokens) > 4:
        return None
    if any(token in TOKEN_LEVEL_STOPWORDS for token in tokens):
        return None
    if tokens[-1] in REJECT_LAST_TOKENS:
        return None
    if any(token in REJECT_CONTAINS_TOKENS for token in tokens):
        return None
    if len(tokens) > 1 and entity.endswith(TRAILING_JOSA_SUFFIXES):
        return None

    return entity


def collect_entities(node: Any, out: set[str], *, korean_only: bool) -> None:
    if isinstance(node, dict):
        entity = node.get("entity")
        entity_type = node.get("type")
        if isinstance(entity, str) and entity_type in TARGET_TYPES:
            normalized = normalize_entity(entity, korean_only=korean_only)
            if normalized:
                out.add(normalized)
        for value in node.values():
            collect_entities(value, out, korean_only=korean_only)
        return

    if isinstance(node, list):
        for item in node:
            collect_entities(item, out, korean_only=korean_only)


def extract_terms_from_lines(lines: list[str]) -> set[str]:
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_existing_terms(dict_path: Path) -> tuple[list[str], set[str]]:
    lines = dict_path.read_text(encoding="utf-8").splitlines()
    return lines, extract_terms_from_lines(lines)


def strip_generated_section(lines: list[str], marker: str) -> list[str]:
    if marker not in lines:
        return lines
    idx = lines.index(marker)
    while idx > 0 and lines[idx - 1] == "":
        idx -= 1
    return lines[:idx]


def merge_terms(dict_path: Path, new_terms: list[str], marker: str) -> None:
    lines, _ = load_existing_terms(dict_path)
    base_lines = strip_generated_section(lines, marker)
    while base_lines and base_lines[-1] == "":
        base_lines.pop()

    output = list(base_lines)
    if output:
        output.append("")
    output.append(marker)
    output.extend(new_terms)
    output.append("")
    dict_path.write_text("\n".join(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="의료 NER 엔티티를 사전에 병합")
    parser.add_argument(
        "--input",
        default="training_data/medical_word_data/Training_medical.json",
        help="입력 JSON 경로",
    )
    parser.add_argument(
        "--dict",
        default="data/medical_dict.txt",
        help="대상 사전 경로",
    )
    parser.add_argument(
        "--marker",
        default="# NER 추출 추가",
        help="자동 생성 섹션 마커",
    )
    parser.add_argument(
        "--allow-non-korean",
        action="store_true",
        help="한글/공백 외 문자를 포함한 엔티티도 허용",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    dict_path = Path(args.dict)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_terms: set[str] = set()
    collect_entities(data, extracted_terms, korean_only=not args.allow_non_korean)

    dict_lines, _ = load_existing_terms(dict_path)
    base_lines = strip_generated_section(dict_lines, args.marker)
    existing_terms = extract_terms_from_lines(base_lines)
    new_terms = sorted(extracted_terms - existing_terms)
    merge_terms(dict_path, new_terms, args.marker)

    print(f"extracted_terms={len(extracted_terms)}")
    print(f"existing_terms={len(existing_terms)}")
    print(f"new_terms={len(new_terms)}")


if __name__ == "__main__":
    main()
