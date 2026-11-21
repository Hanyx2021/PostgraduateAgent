"""一键重建导师主表。

顺序：抓招生目录 → 抓教职工名录 → 首次构建（拿到详情页链接）→ 抓详情页 →
二次构建（补研究领域/简介）→ OpenAlex 补论文 → 落盘。

所有人工维护的本地文件（别名、人工补录、作者映射、人工题录）如果存在会自动读取，
不存在就跳过。抓取遵守 ``advisors.fetch_page`` 的节流与反爬规则。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from advisor_match.config import settings
from advisor_match.ingest.advisors import (
    ProfileInfo,
    fetch_page,
    parse_directory,
    save_advisors,
)
from advisor_match.ingest.build import (
    BuildReport,
    build_advisors,
    load_overrides,
    write_report,
)
from advisor_match.ingest.catalog import parse_catalog
from advisor_match.ingest.openalex import OpenAlexClient, attach_papers
from advisor_match.ingest.profiles import fetch_profiles, save_profiles
from advisor_match.schemas import Advisor, Paper

DEFAULT_DIRECTORY_URL = "https://www.cs.tsinghua.edu.cn/szzk/jzgml.htm"


@dataclass
class RebuildOptions:
    year: int
    catalog_url: str
    directory_url: str = DEFAULT_DIRECTORY_URL
    data_dir: Path = Path("data/processed")
    raw_dir: Path = Path("data/raw/rebuild")
    skip_openalex: bool = False


def build_from_html(
    catalog_html: str,
    directory_html: str,
    *,
    year: int,
    catalog_source: str,
    directory_url: str,
    profiles: dict[str, ProfileInfo] | None = None,
    aliases: dict[str, str] | None = None,
    overrides: dict | None = None,
    fetched_at: date | None = None,
) -> tuple[list[Advisor], BuildReport]:
    """纯函数版本，便于用本地 fixture 测试，不访问网络。"""
    catalog = parse_catalog(catalog_html, year=year)
    entries = parse_directory(directory_html, base_url=directory_url)
    return build_advisors(
        catalog,
        entries,
        profiles=profiles,
        aliases=aliases,
        overrides=overrides,
        fetched_at=fetched_at,
        catalog_source=catalog_source,
    )


def _load_json_if_exists(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def rebuild(options: RebuildOptions) -> dict:
    data_dir = options.data_dir
    raw_dir = options.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] fetch catalog")
    catalog_html = fetch_page(options.catalog_url, cache_dir=raw_dir / "cache")
    (raw_dir / "catalog.html").write_text(catalog_html, encoding="utf-8")

    print("[2/5] fetch directory")
    directory_html = fetch_page(options.directory_url, cache_dir=raw_dir / "cache")
    (raw_dir / "directory.html").write_text(directory_html, encoding="utf-8")

    aliases = _load_json_if_exists(data_dir / "aliases.json")
    overrides_path = data_dir / "overrides.json"
    overrides = load_overrides(overrides_path) if overrides_path.exists() else None

    print("[3/5] first build (profile links)")
    first_pass, _ = build_from_html(
        catalog_html,
        directory_html,
        year=options.year,
        catalog_source=options.catalog_url,
        directory_url=options.directory_url,
        aliases=aliases,
        overrides=overrides,
        fetched_at=date.today(),
    )

    print(f"[4/5] fetch profiles ({sum(1 for a in first_pass if a.profile_url)} links)")
    profiles, failures = fetch_profiles(first_pass, cache_dir=raw_dir.parent / "profiles")
    save_profiles(profiles, data_dir / "profiles.json")

    advisors, report = build_from_html(
        catalog_html,
        directory_html,
        year=options.year,
        catalog_source=options.catalog_url,
        directory_url=options.directory_url,
        profiles=profiles,
        aliases=aliases,
        overrides=overrides,
        fetched_at=date.today(),
    )
    write_report(report, data_dir / "build_report.json", year=options.year)

    summary = {
        "advisors": len(advisors),
        "profiles": len(profiles),
        "profile_failures": failures,
        "build": asdict(report),
    }

    print("[5/5] openalex")
    if options.skip_openalex or not settings.openalex_key:
        summary["openalex"] = "skipped"
        print("  skipped (no key or --skip-openalex)")
    else:
        author_map = _load_json_if_exists(data_dir / "openalex_authors.json")
        manual_raw = _load_json_if_exists(data_dir / "manual_papers.json") or {}
        manual = {
            name: [Paper(**item) for item in items] for name, items in manual_raw.items()
        }
        openalex_report = attach_papers(
            advisors,
            OpenAlexClient(settings.openalex_key),
            author_map=author_map,
            manual_papers=manual,
        )
        payload = asdict(openalex_report)
        (data_dir / "openalex_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        summary["openalex"] = {
            "attached": openalex_report.attached,
            "ambiguous": len(openalex_report.ambiguous),
            "none": len(openalex_report.none),
        }

    save_advisors(advisors, data_dir / "advisors.jsonl")
    (data_dir / "rebuild_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"done: {len(advisors)} advisors -> {data_dir / 'advisors.jsonl'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一键重建导师主表")
    parser.add_argument("--catalog-url", required=True)
    parser.add_argument("--directory-url", default=DEFAULT_DIRECTORY_URL)
    parser.add_argument("--year", type=int, default=2027)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/rebuild"))
    parser.add_argument("--skip-openalex", action="store_true")
    args = parser.parse_args(argv)

    rebuild(
        RebuildOptions(
            year=args.year,
            catalog_url=args.catalog_url,
            directory_url=args.directory_url,
            data_dir=args.data_dir,
            raw_dir=args.raw_dir,
            skip_openalex=args.skip_openalex,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RebuildOptions", "build_from_html", "rebuild"]
