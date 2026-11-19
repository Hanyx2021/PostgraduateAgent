"""生成 OpenAlex 人工核对清单。

只面向自动补全没搞定的导师（歧义 / 查无此人）。每位列出候选作者 ID、
作品数、机构，并标出"精确同名"和"清华"，方便人工拍板。

核对后写入 ``data/processed/openalex_authors.json``（姓名 → 作者 ID），
再用 ``--author-map`` 重跑补全。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from advisor_match.config import settings
from advisor_match.ingest.advisors import load_advisors
from advisor_match.ingest.openalex import AuthorCandidate, OpenAlexClient, normalize_name
from advisor_match.ingest.papers import name_variants


@dataclass
class ReviewEntry:
    name: str
    romanized: str
    status: str
    candidates: list[AuthorCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "romanized": self.romanized,
            "status": self.status,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


def collect_candidates(name: str, client: OpenAlexClient) -> list[AuthorCandidate]:
    variants = name_variants(name)
    variants_norm = {normalize_name(item) for item in variants}
    seen: dict[str, AuthorCandidate] = {}
    for query in [*variants, name]:
        try:
            for candidate in client.search_authors(query):
                seen.setdefault(candidate.author_id, candidate)
        except RuntimeError:
            continue
    return sorted(
        seen.values(),
        key=lambda item: (
            not (normalize_name(item.display_name) in variants_norm and item.is_tsinghua),
            not item.is_tsinghua,
            -item.works_count,
        ),
    )


def build_review(
    names: list[tuple[str, str]], client: OpenAlexClient
) -> list[ReviewEntry]:
    entries: list[ReviewEntry] = []
    for name, status in names:
        entries.append(
            ReviewEntry(
                name=name,
                romanized=name_variants(name)[0],
                status=status,
                candidates=collect_candidates(name, client),
            )
        )
    return entries


def render_markdown(entries: list[ReviewEntry], *, generated_at: date) -> str:
    lines = [
        "# OpenAlex 人工核对清单",
        "",
        f"生成时间：{generated_at.isoformat()}　共 {len(entries)} 位"
        f"（{sum(1 for e in entries if e.status == 'ambiguous')} 位歧义，"
        f"{sum(1 for e in entries if e.status == 'none')} 位未找到）",
        "",
        "核对后把选定的作者 ID 写入 `data/processed/openalex_authors.json`，例如：",
        "",
        "```json",
        '{ "甘霖": "A5044791875" }',
        "```",
        "",
        "然后重跑：",
        "",
        "```bash",
        "python -m advisor_match.ingest.openalex \\",
        "  --advisors data/processed/advisors.jsonl \\",
        "  --author-map data/processed/openalex_authors.json",
        "```",
        "",
        "判断依据：机构为清华、方向与导师研究领域一致、作品数量合理。",
        "不要仅凭名字相同就选。",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(f"## {index}. {entry.name}（{entry.romanized}）— {entry.status}")
        lines.append("")
        if not entry.candidates:
            lines.append("无候选。")
            lines.append("")
            continue
        lines.append("| 作者 ID | 显示名 | 作品数 | 引用 | 机构 | 精确同名 | 清华 |")
        lines.append("| --- | --- | ---: | ---: | --- | :---: | :---: |")
        variants_norm = {normalize_name(item) for item in name_variants(entry.name)}
        for candidate in entry.candidates[:10]:
            exact = "是" if normalize_name(candidate.display_name) in variants_norm else ""
            tsinghua = "是" if candidate.is_tsinghua else ""
            # 清华排在最前，避免被截断后看不出机构
            ordered = sorted(
                candidate.institutions, key=lambda item: "tsinghua" not in item.lower()
            )
            institutions = "；".join(ordered[:3]) or "—"
            lines.append(
                f"| {candidate.author_id} | {candidate.display_name} | "
                f"{candidate.works_count} | {candidate.cited_by_count} | "
                f"{institutions} | {exact} | {tsinghua} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 OpenAlex 人工核对清单")
    parser.add_argument("--advisors", type=Path, default=Path("data/processed/advisors.jsonl"))
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/openalex_report.json")
    )
    parser.add_argument("--out-md", type=Path, default=Path("data/processed/openalex_review.md"))
    parser.add_argument("--out-json", type=Path, default=Path("data/processed/openalex_review.json"))
    args = parser.parse_args(argv)

    if not settings.openalex_key:
        parser.error("缺少 AM_OPENALEX_KEY，请在本地 .env 中配置")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    targets = [(name, "ambiguous") for name in report["ambiguous"]]
    targets += [(name, "none") for name in report["none"]]

    client = OpenAlexClient(settings.openalex_key)
    entries = build_review(targets, client)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(entries, generated_at=date.today()), encoding="utf-8")
    args.out_json.write_text(
        json.dumps([entry.to_dict() for entry in entries], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"review entries: {len(entries)}")
    print(f"written: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ReviewEntry", "build_review", "collect_candidates", "render_markdown"]
