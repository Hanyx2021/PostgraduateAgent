"""一键重建流水线的纯函数测试（使用本地 fixture，不访问网络）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from advisor_match.ingest.advisors import parse_profile
from advisor_match.ingest.rebuild import build_from_html

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_build_from_html_combines_catalog_directory_and_profile() -> None:
    profiles = {"张三": parse_profile(load("advisor_profile.html"))}
    advisors, report = build_from_html(
        load("admission_catalog.html"),
        load("advisor_directory.html"),
        year=2027,
        catalog_source="https://yzbm.example.edu.cn/catalog",
        directory_url="https://www.cs.example.edu.cn/szzk/jzgml.htm",
        profiles=profiles,
        fetched_at=date(2026, 11, 21),
    )

    assert [advisor.name for advisor in advisors] == ["张三", "李四", "王五", "赵六"]
    assert report.total == 4
    assert report.with_directory == 4
    assert all(advisor.admissions for advisor in advisors)
    zhang = advisors[0]
    assert zhang.email == "zhangsan@example.edu.cn"
    assert zhang.research_tags == ["自然语言处理", "多模态学习", "信息检索"]


def test_build_from_html_tolerates_missing_profiles() -> None:
    advisors, report = build_from_html(
        load("admission_catalog.html"),
        load("advisor_directory.html"),
        year=2027,
        catalog_source="https://yzbm.example.edu.cn/catalog",
        directory_url="https://www.cs.example.edu.cn/szzk/jzgml.htm",
    )
    assert len(advisors) == 4
    assert all(advisor.research_tags == [] for advisor in advisors)
    assert all(advisor.admissions for advisor in advisors)
