"""导师主表构建测试（合成 fixture）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from advisor_match.ingest.advisors import DirectoryEntry, parse_directory, parse_profile
from advisor_match.ingest.build import ManualOverride, build_advisors, write_report
from advisor_match.ingest.catalog import parse_catalog

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG_SOURCE = "https://yzbm.example.edu.cn/catalog?yxsdm=024"
DIRECTORY_URL = "https://www.cs.example.edu.cn/szzk/jzgml.htm"
FETCHED = date(2026, 11, 17)


@pytest.fixture()
def catalog():
    return parse_catalog((FIXTURES / "admission_catalog.html").read_text(encoding="utf-8"), year=2027)


@pytest.fixture()
def entries():
    return parse_directory((FIXTURES / "advisor_directory.html").read_text(encoding="utf-8"), base_url=DIRECTORY_URL)


@pytest.fixture()
def profile():
    return parse_profile((FIXTURES / "advisor_profile.html").read_text(encoding="utf-8"))


def test_catalog_is_the_primary_table(catalog, entries) -> None:
    advisors, report = build_advisors(
        catalog, entries, fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )

    assert [advisor.name for advisor in advisors] == ["张三", "李四", "王五", "赵六"]
    assert report.total == 4
    assert report.with_directory == 4
    assert report.unmatched_catalog == []
    assert report.directory_only == []


def test_directory_fields_are_filled(catalog, entries, profile) -> None:
    advisors, _ = build_advisors(
        catalog, entries, profiles={"张三": profile}, fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )
    zhang = advisors[0]

    assert zhang.unit == "计算机软件研究所"
    assert zhang.title == "教授"
    assert zhang.email == "zhangsan@example.edu.cn"
    assert zhang.profile_url == "https://www.cs.example.edu.cn/info/1101/0001.htm"
    assert zhang.homepage == "https://example-lab.edu.cn/zhangsan"
    assert zhang.research_tags == ["自然语言处理", "多模态学习", "信息检索"]
    assert zhang.admissions[0].program_code == "081200"
    assert zhang.fetched_at == FETCHED


def test_catalog_only_advisor_is_kept_with_empty_profile(catalog, entries) -> None:
    advisors, report = build_advisors(
        catalog, entries[:3], fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )

    zhao = next(advisor for advisor in advisors if advisor.name == "赵六")
    assert zhao.unit is None
    assert zhao.email is None
    assert zhao.research_tags == []
    assert zhao.admissions, "没有档案也必须保留招生记录"
    assert report.unmatched_catalog == ["赵六"]
    assert report.with_directory == 3


def test_directory_only_people_are_reported(catalog, entries) -> None:
    extra = DirectoryEntry(
        name="孙八",
        title="教授",
        email="sunba@example.edu.cn",
        unit="计算机软件研究所",
        rank_group="正高",
        profile_url="https://www.cs.example.edu.cn/info/1101/9999.htm",
    )
    advisors, report = build_advisors(
        catalog, [*entries, extra], fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )

    assert all(advisor.name != "孙八" for advisor in advisors)
    assert report.directory_only == ["孙八"]


def test_duplicate_directory_names_are_reported(catalog, entries) -> None:
    _, report = build_advisors(
        catalog, [*entries, entries[0]], fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )
    assert report.duplicate_directory_names == ["张三"]


def test_report_coverage_counts(catalog, entries, profile, tmp_path) -> None:
    advisors, report = build_advisors(
        catalog, entries, profiles={"张三": profile}, fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )
    coverage = report.coverage()
    assert coverage["邮箱"] == "4/4"
    assert coverage["研究领域"] == "1/4"

    path = tmp_path / "report.json"
    write_report(report, path, year=2027)
    assert '"coverage"' in path.read_text(encoding="utf-8")


def test_name_normalization_handles_spaces_and_part_time_suffix(catalog) -> None:
    entry = DirectoryEntry(
        name="张 三（校内兼职）",
        title="教授",
        email="zhangsan@example.edu.cn",
        unit="计算机软件研究所",
        rank_group="正高",
        profile_url="https://www.cs.example.edu.cn/info/1101/0001.htm",
    )
    advisors, report = build_advisors(
        catalog, [entry], fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )
    assert advisors[0].name == "张三", "展示名以目录为准"
    assert advisors[0].email == "zhangsan@example.edu.cn"
    assert report.with_directory == 1


def test_alias_maps_catalog_name_to_directory_entry(entries) -> None:
    """外籍/拼音名可通过别名匹配名录，展示名仍以目录为准。"""
    raw = (FIXTURES / "admission_catalog.html").read_text(encoding="utf-8")
    catalog = parse_catalog(raw.replace("赵六", "John Smith"), year=2027)

    advisors, report = build_advisors(
        catalog,
        entries,
        aliases={"John Smith": "赵六"},
        fetched_at=FETCHED,
        catalog_source=CATALOG_SOURCE,
    )
    john = advisors[3]
    assert john.name == "John Smith"
    assert john.email == "zhaoliu@example.edu.cn"
    assert report.with_directory == 4


def test_overrides_fill_missing_fields_with_source(catalog, entries) -> None:
    override = ManualOverride(
        source="https://example.edu.cn/faculty/li",
        email="lisi@lab.example.edu.cn",
        title="研究员",
        research_tags=["信息检索"],
    )
    advisors, report = build_advisors(
        catalog,
        entries,
        overrides={"李四": override},
        fetched_at=FETCHED,
        catalog_source=CATALOG_SOURCE,
    )
    li = advisors[1]
    assert li.email == "lisi@lab.example.edu.cn"
    assert li.title == "研究员"
    assert li.research_tags == ["信息检索"]
    assert any(source.doc_id == "https://example.edu.cn/faculty/li" for source in li.sources)
    assert report.missing_overrides == []


def test_unknown_override_key_is_reported(catalog, entries) -> None:
    override = ManualOverride(source="https://example.edu.cn/x", email="x@example.edu.cn")
    _, report = build_advisors(
        catalog,
        entries,
        overrides={"查无此人": override},
        fetched_at=FETCHED,
        catalog_source=CATALOG_SOURCE,
    )
    assert report.missing_overrides == ["查无此人"]


def test_build_never_stores_phone(catalog, entries) -> None:
    advisors, _ = build_advisors(
        catalog, entries, fetched_at=FETCHED, catalog_source=CATALOG_SOURCE
    )
    dumped = "".join(advisor.model_dump_json() for advisor in advisors)
    assert "62780001" not in dumped
    assert "62780003" not in dumped
