"""导师主表构建：招生目录定人选，系名录补档案。

主表为什么以目录为准：2020 版名录既不完整（缺新入职与双聘教师），也不代表
招生资格。目录回答"今年谁在招"，名录回答"这个人的单位、职称、邮箱、主页"。
两者按姓名合并，合不上的如实列出，不猜。

用法（离线，输入为已保存的 HTML）::

    python -m advisor_match.ingest.build \\
        --catalog data/raw/catalog2027.html \\
        --directory data/raw/jzgml.html \\
        --year 2027 \\
        --source "https://yzbm.tsinghua.edu.cn/..." \\
        --out data/processed/advisors.jsonl \\
        --report data/processed/build_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from advisor_match.ingest.advisors import (
    DEPARTMENT_NAME,
    DirectoryEntry,
    ProfileInfo,
    load_advisors,
    parse_directory,
    save_advisors,
    to_advisor,
)
from advisor_match.ingest.catalog import Catalog, apply_catalog, parse_catalog
from advisor_match.ingest.profiles import load_profiles
from advisor_match.schemas import Advisor, SourceRef

#: 名录里的姓名噪声：全角/半角括号的兼职标注、姓名中的空格。
_NAME_NOISE = re.compile(r"[（(]\s*(校内兼职|兼职|退休)\s*[）)]")
_SPACES = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """姓名匹配键：去空格、去兼职标注。仅用于匹配，不改变展示名。"""
    return _SPACES.sub("", _NAME_NOISE.sub("", name))


@dataclass
class ManualOverride:
    """人工补录的一条档案。每项都必须附带出处，不允许"我记得是这样"。"""

    source: str
    email: str | None = None
    title: str | None = None
    unit: str | None = None
    profile_url: str | None = None
    homepage: str | None = None
    research_tags: list[str] = field(default_factory=list)
    research_summary: str | None = None


def load_overrides(path: Path) -> dict[str, ManualOverride]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: ManualOverride(**item)  # type: ignore[arg-type]
        for name, item in payload.items()
    }


def apply_overrides(
    advisors: list[Advisor], overrides: dict[str, ManualOverride]
) -> list[str]:
    """把人工补录写入 Advisor，返回没匹配上的键名（用于发现拼写错误）。"""
    by_name = {normalize_name(advisor.name): advisor for advisor in advisors}
    missing: list[str] = []

    for key, override in overrides.items():
        advisor = by_name.get(normalize_name(key))
        if advisor is None:
            missing.append(key)
            continue

        fields = (
            "email",
            "title",
            "unit",
            "profile_url",
            "homepage",
            "research_tags",
            "research_summary",
        )
        for name in fields:
            value = getattr(override, name)
            if not value:
                continue
            setattr(advisor, name, value)
            quote = "，".join(value) if isinstance(value, list) else str(value)
            advisor.sources.append(
                SourceRef(doc_id=override.source, locator="人工补录", quote=quote[:800])
            )
    return missing


@dataclass
class BuildReport:
    """构建结果的可审计摘要。字段全是计数或姓名，不含邮箱正文。"""

    total: int = 0
    with_directory: int = 0
    with_email: int = 0
    with_profile_url: int = 0
    with_homepage: int = 0
    with_research_tags: int = 0
    with_papers: int = 0
    with_admissions: int = 0
    #: 目录在列但名录查无此人——需人工补录档案。
    unmatched_catalog: list[str] = field(default_factory=list)
    #: 名录里有、目录里没有——可能有招生资格但不招，或已离开，不进入主表。
    directory_only: list[str] = field(default_factory=list)
    duplicate_directory_names: list[str] = field(default_factory=list)
    #: 人工补录文件里对不上主表的键名，通常是拼写错误。
    missing_overrides: list[str] = field(default_factory=list)

    def coverage(self) -> dict[str, str]:
        if self.total == 0:
            return {}
        return {
            "邮箱": f"{self.with_email}/{self.total}",
            "系内详情页": f"{self.with_profile_url}/{self.total}",
            "个人主页": f"{self.with_homepage}/{self.total}",
            "研究领域": f"{self.with_research_tags}/{self.total}",
            "近期论文": f"{self.with_papers}/{self.total}",
            "招生记录": f"{self.with_admissions}/{self.total}",
        }


def build_advisors(
    catalog: Catalog,
    entries: list[DirectoryEntry],
    *,
    profiles: dict[str, ProfileInfo] | None = None,
    aliases: dict[str, str] | None = None,
    overrides: dict[str, ManualOverride] | None = None,
    fetched_at: date | None = None,
    catalog_source: str,
) -> tuple[list[Advisor], BuildReport]:
    """按目录顺序生成导师主表，并用名录/详情页补全档案。

    ``aliases``：目录名 → 名录名，只在有明确证据时填写（外籍教师的中文名等）。
    ``overrides``：人工补录，覆盖名录缺失的字段，每条必须带出处。
    """
    profiles = profiles or {}
    aliases = aliases or {}
    overrides = overrides or {}
    fetched_at = fetched_at or date.today()

    by_name: dict[str, DirectoryEntry] = {}
    duplicates: list[str] = []
    for entry in entries:
        key = normalize_name(entry.name)
        if key in by_name:
            duplicates.append(entry.name)
        else:
            by_name[key] = entry

    report = BuildReport(duplicate_directory_names=duplicates)
    advisors: list[Advisor] = []

    for catalog_advisor in catalog.advisors():
        name = catalog_advisor.name
        entry = by_name.get(normalize_name(name))
        if entry is None and name in aliases:
            entry = by_name.get(normalize_name(aliases[name]))
        profile = profiles.get(name) or profiles.get(normalize_name(name))

        if entry is not None:
            advisor = to_advisor(entry, profile or ProfileInfo(name=name), fetched_at=fetched_at)
            # 展示名以招生目录为准，名录里的空格、兼职标注只用于匹配。
            advisor.name = name
            report.with_directory += 1
        else:
            advisor = Advisor(name=name, affiliation=DEPARTMENT_NAME, fetched_at=fetched_at)
            report.unmatched_catalog.append(name)
        if profile is not None:
            advisor.research_summary = profile.research_summary
        advisors.append(advisor)

    apply_catalog(advisors, catalog, source_url=catalog_source)
    report.missing_overrides = apply_overrides(advisors, overrides)

    catalog_keys = {normalize_name(advisor.name) for advisor in advisors}
    report.directory_only = sorted(
        {entry.name for entry in entries if normalize_name(entry.name) not in catalog_keys}
    )

    report.total = len(advisors)
    report.with_email = sum(1 for advisor in advisors if advisor.email)
    report.with_profile_url = sum(1 for advisor in advisors if advisor.profile_url)
    report.with_homepage = sum(1 for advisor in advisors if advisor.homepage)
    report.with_research_tags = sum(1 for advisor in advisors if advisor.research_tags)
    report.with_papers = sum(1 for advisor in advisors if advisor.recent_papers)
    report.with_admissions = sum(1 for advisor in advisors if advisor.admissions)

    return advisors, report


def write_report(report: BuildReport, path: Path, *, year: int) -> None:
    payload = asdict(report)
    payload["coverage"] = report.coverage()
    payload["year"] = year
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建导师主表（目录 × 名录）")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--source", required=True, help="招生目录 URL")
    parser.add_argument(
        "--directory-url",
        default="https://www.cs.tsinghua.edu.cn/szzk/jzgml.htm",
        help="教职工名录页面 URL，用于把详情页相对链接补全",
    )
    parser.add_argument("--profiles", type=Path, default=None, help="profiles.json，由 profiles.py 生成")
    parser.add_argument("--aliases", type=Path, default=None, help="别名映射 JSON：目录名 → 名录名")
    parser.add_argument("--overrides", type=Path, default=None, help="人工补录 JSON")
    parser.add_argument("--previous", type=Path, default=None, help="上一版 advisors.jsonl，保留其近五年论文")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    catalog = parse_catalog(args.catalog.read_text(encoding="utf-8"), year=args.year)
    entries = parse_directory(
        args.directory.read_text(encoding="utf-8"), base_url=args.directory_url
    )
    profiles = load_profiles(args.profiles) if args.profiles else None
    aliases = json.loads(args.aliases.read_text(encoding="utf-8")) if args.aliases else None
    overrides = load_overrides(args.overrides) if args.overrides else None
    advisors, report = build_advisors(
        catalog,
        entries,
        profiles=profiles,
        aliases=aliases,
        overrides=overrides,
        catalog_source=args.source,
    )

    if args.previous and args.previous.exists():
        previous = {advisor.name: advisor for advisor in load_advisors(args.previous)}
        for advisor in advisors:
            old = previous.get(advisor.name)
            if old is not None and old.recent_papers:
                advisor.recent_papers = old.recent_papers
        report.with_papers = sum(1 for advisor in advisors if advisor.recent_papers)

    save_advisors(advisors, args.out)
    if args.report:
        write_report(report, args.report, year=args.year)

    print(f"advisors: {report.total}")
    print(f"matched directory: {report.with_directory}")
    print(f"unmatched catalog: {len(report.unmatched_catalog)}")
    print(f"directory only: {len(report.directory_only)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
