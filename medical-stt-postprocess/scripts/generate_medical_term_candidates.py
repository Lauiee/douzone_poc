#!/usr/bin/env python3
"""Training_medical.json에서 고정밀 의료용어 후보를 추출한다."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TARGET_TYPES = {
    "TMM_DISEASE",
    "TMM_SYMPTOM",
    "TMM_DRUG",
    "TM_CELL_TISSUE_ORGAN",
    "TR_MED_PROCEDURE",
    "TR_MED_MEASURE",
    "AFW_MED_DEVICE",
}

KOREAN_SPACE_RE = re.compile(r"^[가-힣 ]+$")
STOP_TOKENS = {
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
VERBISH_TOKENS = {
    "의한",
    "위한",
    "이용한",
    "동반한",
    "하는",
    "하고",
    "하며",
    "되는",
    "된다",
    "된",
    "보이는",
    "나타나는",
}
DESCRIPTIVE_PREFIXES = {
    "심한",
    "가벼운",
    "가려운",
    "가쁜",
    "지속적인",
    "반복적인",
}


def normalize_entity(text: str) -> str:
    return " ".join(text.split()).strip()


def is_independent_term(entity: str, *, max_chars: int, max_tokens: int) -> bool:
    compact_len = len(entity.replace(" ", ""))
    if compact_len < 2 or compact_len > max_chars:
        return False
    if not KOREAN_SPACE_RE.fullmatch(entity):
        return False

    tokens = entity.split()
    if not tokens or len(tokens) > max_tokens:
        return False
    if any(token in STOP_TOKENS for token in tokens):
        return False
    if any(token in VERBISH_TOKENS for token in tokens):
        return False
    if len(tokens) > 1 and tokens[0] in DESCRIPTIVE_PREFIXES:
        return False
    return True


def extract_candidates(
    input_path: Path,
    *,
    min_freq: int,
    min_freq_spaced: int,
    max_chars: int,
    max_tokens: int,
    max_spaced_tokens: int,
) -> tuple[list[tuple[str, int]], Counter[str], int, int]:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    term_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    target_mentions = 0
    exact_span_mentions = 0

    for item in payload["data"]:
        text = item.get("text", "")
        for ne in item.get("NE", []):
            ne_type = ne.get("type")
            if ne_type not in TARGET_TYPES:
                continue
            target_mentions += 1

            entity = normalize_entity(str(ne.get("entity", "")))
            begin = ne.get("begin")
            end = ne.get("end")
            if not isinstance(begin, int) or not isinstance(end, int):
                continue

            if text[begin : end + 1] != entity:
                continue
            exact_span_mentions += 1

            if not is_independent_term(entity, max_chars=max_chars, max_tokens=max_tokens):
                continue

            term_counter[entity] += 1
            type_counter[ne_type] += 1

    candidates = sorted(
        [
            (term, count)
            for term, count in term_counter.items()
            if (
                count >= min_freq
                and (
                    " " not in term
                    or (
                        count >= min_freq_spaced
                        and len(term.split()) <= max_spaced_tokens
                    )
                )
            )
        ],
        key=lambda x: (-x[1], x[0]),
    )
    return candidates, type_counter, target_mentions, exact_span_mentions


def write_txt(path: Path, candidates: list[tuple[str, int]]) -> None:
    path.write_text("\n".join(term for term, _ in candidates) + "\n", encoding="utf-8")


def write_md(
    path: Path,
    candidates: list[tuple[str, int]],
    type_counter: Counter[str],
    *,
    min_freq: int,
    min_freq_spaced: int,
    max_spaced_tokens: int,
    target_mentions: int,
    exact_span_mentions: int,
) -> None:
    lines = [
        "# 의료용어 후보 추출 결과",
        "",
        "## 기준",
        "",
        f"- 대상 타입: {', '.join(sorted(TARGET_TYPES))}",
        "- `text[begin:end+1] == entity` 인 경우만 채택",
        "- 한글/공백만 허용",
        "- 독립 용어 형태만 허용",
        f"- 최소 등장 빈도: {min_freq}",
        f"- 공백 포함 후보 최소 등장 빈도: {min_freq_spaced}",
        f"- 공백 포함 후보 최대 토큰 수: {max_spaced_tokens}",
        "",
        "## 집계",
        "",
        f"- 대상 mention 수: {target_mentions}",
        f"- exact span 일치 mention 수: {exact_span_mentions}",
        f"- 최종 후보 수: {len(candidates)}",
        "",
        "## 타입별 mention 수",
        "",
    ]
    for ne_type, count in type_counter.most_common():
        lines.append(f"- `{ne_type}`: {count}")

    lines.extend(
        [
            "",
            "## 상위 후보 200개",
            "",
            "| 용어 | 빈도 |",
            "| --- | ---: |",
        ]
    )
    for term, count in candidates[:200]:
        lines.append(f"| {term} | {count} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="고정밀 의료용어 후보 추출")
    parser.add_argument(
        "--input",
        default="training_data/medical_word_data/Training_medical.json",
        help="입력 JSON 경로",
    )
    parser.add_argument(
        "--output-txt",
        default="strict_medical_term_candidates.txt",
        help="후보 txt 출력 경로",
    )
    parser.add_argument(
        "--output-md",
        default="strict_medical_term_candidates.md",
        help="후보 md 출력 경로",
    )
    parser.add_argument("--min-freq", type=int, default=5, help="최소 등장 빈도")
    parser.add_argument(
        "--min-freq-spaced",
        type=int,
        default=20,
        help="공백 포함 후보의 최소 등장 빈도",
    )
    parser.add_argument("--max-chars", type=int, default=12, help="최대 글자 수")
    parser.add_argument("--max-tokens", type=int, default=3, help="최대 토큰 수")
    parser.add_argument(
        "--max-spaced-tokens",
        type=int,
        default=2,
        help="공백 포함 후보의 최대 토큰 수",
    )
    args = parser.parse_args()

    candidates, type_counter, target_mentions, exact_span_mentions = extract_candidates(
        Path(args.input),
        min_freq=args.min_freq,
        min_freq_spaced=args.min_freq_spaced,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        max_spaced_tokens=args.max_spaced_tokens,
    )
    write_txt(Path(args.output_txt), candidates)
    write_md(
        Path(args.output_md),
        candidates,
        type_counter,
        min_freq=args.min_freq,
        min_freq_spaced=args.min_freq_spaced,
        max_spaced_tokens=args.max_spaced_tokens,
        target_mentions=target_mentions,
        exact_span_mentions=exact_span_mentions,
    )
    print(f"candidates={len(candidates)}")
    print(f"output_txt={args.output_txt}")
    print(f"output_md={args.output_md}")


if __name__ == "__main__":
    main()
