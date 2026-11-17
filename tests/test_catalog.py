"""招生目录解析测试（合成 fixture）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from advisor_match.ingest.catalog import apply_catalog, parse_catalog
from advisor_match.schemas import Advisor

FIXTURE = Path(__file__).parent / "fixtures" / "admission_catalog.html"
SOURCE = "https://yzbm.example.edu.cn/catalog?yxsdm=024"


@pytest.fixture()
def catalog():
    return parse_catalog(FIXTURE.read_text(encoding="utf-8"), year=2026)


def test_parses_directions(catalog) -> None:
    assert [direction.label for direction in catalog.directions] == [
        "01 高性能计算",
        "02 人工智能（强基）（仅推免）",
    ]
    assert catalog.directions[0].program_label == "081200 计算机科学与技术（学术学位）"
    assert catalog.directions[1].program_label == "140500 智能科学与技术（学术学位）"


def test_parses_advisors_and_notes(catalog) -> None:
    first = catalog.directions[0]
    assert [entry.name for entry in first.advisors] == ["张三", "李四", "王五"]
    assert first.advisors[1].note == "不招收直博生"
    assert first.advisors[0].note is None


def test_marks_only_recommendation(catalog) -> None:
    assert catalog.directions[1].only_recommendation is True


def test_excludes_group_entries(catalog) -> None:
    names = [entry.name for direction in catalog.directions for entry in direction.advisors]
    assert "导师组" not in names


def test_apply_catalog_sets_eligibility_not_quota(catalog) -> None:
    advisors = [
        Advisor(name="张三"),
        Advisor(name="李四"),
        Advisor(name="王五"),
        Advisor(name="赵六"),
    ]
    unmatched, matched = apply_catalog(advisors, catalog, source_url=SOURCE)

    assert unmatched == []
    assert matched == ["张三", "李四", "王五", "赵六"]

    zhang = advisors[0].admissions[0]
    assert (zhang.year, zhang.program_code, zhang.direction_code) == (2026, "081200", "01")
    assert zhang.direction == "高性能计算"
    assert zhang.program_type == "学术学位"
    assert zhang.source == SOURCE
    assert advisors[1].admissions[0].notes == ["不招收直博生"]
    assert advisors[3].admissions[0].only_recommendation is True
    # 资格不等于名额：recruiting 仍是未知。
    assert all(advisor.recruiting is None for advisor in advisors)


def test_apply_catalog_reports_unmatched(catalog) -> None:
    advisors = [Advisor(name="张三")]
    unmatched, _ = apply_catalog(advisors, catalog, source_url=SOURCE)
    assert sorted(unmatched) == ["李四", "王五", "赵六"]


def test_apply_catalog_is_idempotent(catalog) -> None:
    advisors = [Advisor(name="张三")]
    apply_catalog(advisors, catalog, source_url=SOURCE)
    apply_catalog(advisors, catalog, source_url=SOURCE)
    assert len(advisors[0].admissions) == 1
