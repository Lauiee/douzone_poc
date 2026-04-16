#!/usr/bin/env python3
"""test_inputs 폴더의 txt 파일들을 전/후 비교 markdown으로 생성."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline import MedicalSTTPipeline  # noqa: E402


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _maybe_process_json_list(raw: str, pipeline: MedicalSTTPipeline) -> str | None:
    """JSON 배열 [{'text': '...'}, ...] 형태면 항목별 후처리 후 동일 형식으로 반환. 아니면 None."""
    s = raw.strip()
    if not (s.startswith("[") or s.startswith("{")):
        return None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data and all(
        isinstance(item, dict) and "text" in item for item in data
    ):
        out = []
        for item in data:
            row = dict(item)
            row["text"] = pipeline.process_text(str(row["text"])).corrected
            out.append(row)
        return json.dumps(out, ensure_ascii=False, indent=2)
    if isinstance(data, dict) and "text" in data:
        row = dict(data)
        row["text"] = pipeline.process_text(str(row["text"])).corrected
        return json.dumps(row, ensure_ascii=False, indent=2)
    return None


def _split_two_paragraphs(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def build_markdown(sections: list[tuple[str, str, str]]) -> str:
    out: list[str] = ["# 전체 텍스트 전후 비교", ""]
    for title, original, corrected in sections:
        o1, o2 = _split_two_paragraphs(original)
        c1, c2 = _split_two_paragraphs(corrected)
        out.extend(
            [
                f"## {title}",
                "",
                "### 전체 원문",
                "",
                o1,
                "",
                o2,
                "",
                "### 전체 후처리",
                "",
                c1,
                "",
                c2,
                "",
            ]
        )
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=_ROOT / "test_inputs",
        help="원문 txt 파일 폴더 (기본: test_inputs)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "full_before_after_comparison.md",
        help="출력 markdown 경로",
    )
    ap.add_argument(
        "--no-kogpt2",
        action="store_true",
        help="KoGPT2 PPL 단계 끄기 (빠른 회귀 확인용; 기본은 켬)",
    )
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"입력 txt 파일이 없습니다: {args.input_dir}")

    pipeline = MedicalSTTPipeline(enable_kogpt2=not args.no_kogpt2)
    sections: list[tuple[str, str, str]] = []

    for idx, path in enumerate(files, 1):
        original = _read_text(path)
        corrected = _maybe_process_json_list(original, pipeline)
        if corrected is None:
            corrected = pipeline.process_text(original).corrected
        title = path.stem.upper() if path.stem else f"CASE{idx}"
        sections.append((title, original, corrected))

    args.output.write_text(build_markdown(sections), encoding="utf-8")
    print(f"작성 완료: {args.output}")
    print(f"입력 파일 수: {len(files)}")


if __name__ == "__main__":
    main()
