#!/usr/bin/env python3
"""result_type1.json / result_type3.json 읽어 전·후 비교 리포트 생성 (MD + 나란히 HTML)."""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _collect_changes(stages: dict) -> list[str]:
    lines: list[str] = []
    order = [
        "rule_based",
        "medical_confusion",
        "kogpt2_ppl",
        "kobert_context",
    ]
    for name in order:
        if name not in stages:
            continue
        st = stages[name]
        if st.get("skipped"):
            continue
        ch = st.get("changes") or []
        if not ch:
            continue
        for c in ch:
            if isinstance(c, dict):
                o = c.get("original", "")
                corr = c.get("corrected", "")
                if o or corr:
                    lines.append(f"{name}: `{o}` → `{corr}`")
            else:
                lines.append(f"{name}: {c}")
    return lines


def build_markdown(
    type1_path: Path,
    type3_path: Path,
    t1: list,
    t3: list,
) -> str:
    out: list[str] = []
    out.append("# 케이스 테스트 전·후 비교\n")
    out.append(f"- 생성(UTC): {datetime.now(timezone.utc).isoformat()}")
    out.append(f"- 소스: `{type1_path.name}`, `{type3_path.name}`")
    out.append("")
    out.append("한눈에 비교는 **`case_test_report.html`** 을 브라우저로 여세요.\n")

    def blocks(title: str, items: list[dict]) -> None:
        out.append(f"## {title}\n")
        for i, item in enumerate(items, 1):
            orig = item.get("original", "")
            corr = item.get("corrected", "")
            stages = item.get("stages") or {}
            out.append(f"### {title} — 샘플 {i}\n")
            flag = "✅ 동일" if orig == corr else "⚠️ 변경"
            out.append(f"**상태:** {flag}\n")
            out.append("**원문**")
            out.append("```")
            out.append(orig)
            out.append("```\n")
            out.append("**교정**")
            out.append("```")
            out.append(corr)
            out.append("```\n")
            ch = _collect_changes(stages)
            if ch:
                out.append("**치환 요약**")
                for line in ch[:100]:
                    out.append(f"- {line}")
                out.append("")
            out.append("---\n")

    blocks("Type 1", t1)
    blocks("Type 3", t3)
    return "\n".join(out)


def build_html(
    type1_path: Path,
    type3_path: Path,
    t1: list,
    t3: list,
) -> str:
    css = """
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 1rem 1.5rem; max-width: 1600px; }
    h1 { font-size: 1.25rem; }
    h2 { margin-top: 2rem; font-size: 1.1rem; border-bottom: 1px solid #ccc; }
    h3 { font-size: 1rem; margin-top: 1.5rem; }
    .meta { color: #555; font-size: 0.9rem; margin-bottom: 1rem; }
    .pair {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 8px 0 16px;
      align-items: start;
    }
    @media (max-width: 900px) { .pair { grid-template-columns: 1fr; } }
    .col { border: 1px solid #ddd; border-radius: 6px; padding: 10px; background: #fafafa; }
    .col h4 { margin: 0 0 8px; font-size: 0.85rem; color: #333; }
    .col pre {
      margin: 0; white-space: pre-wrap; word-break: break-word;
      font-size: 0.88rem; line-height: 1.45; max-height: 420px; overflow: auto;
    }
    .orig { border-left: 4px solid #c62828; }
    .corr { border-left: 4px solid #2e7d32; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .same { background: #e8f5e9; color: #1b5e20; }
    .diff { background: #fff3e0; color: #e65100; }
    ul.changes { font-size: 0.85rem; margin: 8px 0 0; padding-left: 1.2rem; }
    """

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>")
    parts.append(f"<title>케이스 테스트 전·후</title><style>{css}</style></head><body>")
    parts.append("<h1>케이스 테스트 전·후 비교</h1>")
    parts.append(
        f"<p class='meta'>생성(UTC) {html.escape(datetime.now(timezone.utc).isoformat())} · "
        f"{html.escape(type1_path.name)}, {html.escape(type3_path.name)}</p>"
    )

    def section(title: str, items: list[dict]) -> None:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        for i, item in enumerate(items, 1):
            orig = item.get("original", "")
            corr = item.get("corrected", "")
            stages = item.get("stages") or {}
            same = orig == corr
            badge = "<span class='badge same'>동일</span>" if same else "<span class='badge diff'>변경</span>"
            parts.append(f"<h3>{html.escape(title)} — 샘플 {i} {badge}</h3>")
            parts.append("<div class='pair'>")
            parts.append("<div class='col orig'><h4>원문</h4><pre>")
            parts.append(html.escape(orig))
            parts.append("</pre></div>")
            parts.append("<div class='col corr'><h4>교정</h4><pre>")
            parts.append(html.escape(corr))
            parts.append("</pre></div></div>")
            ch = _collect_changes(stages)
            if ch:
                parts.append("<ul class='changes'>")
                for line in ch[:120]:
                    parts.append(f"<li>{html.escape(line)}</li>")
                parts.append("</ul>")

    section("Type 1", t1)
    section("Type 3", t3)
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type1", type=Path, default=_ROOT / "result_type1.json")
    ap.add_argument("--type3", type=Path, default=_ROOT / "result_type3.json")
    ap.add_argument("--md", type=Path, default=_ROOT / "case_test_report.md")
    ap.add_argument("--html", type=Path, default=_ROOT / "case_test_report.html")
    args = ap.parse_args()

    t1 = json.loads(args.type1.read_text(encoding="utf-8"))
    t3 = json.loads(args.type3.read_text(encoding="utf-8"))

    args.md.write_text(
        build_markdown(args.type1, args.type3, t1, t3),
        encoding="utf-8",
    )
    args.html.write_text(
        build_html(args.type1, args.type3, t1, t3),
        encoding="utf-8",
    )
    print(f"작성: {args.md}")
    print(f"작성: {args.html}")


if __name__ == "__main__":
    main()
