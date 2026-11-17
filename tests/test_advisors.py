"""导师库解析测试（使用合成 fixture，不访问网络）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from advisor_match.ingest.advisors import (
    BlockedError,
    parse_directory,
    parse_profile,
    to_advisor,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "https://www.cs.example.edu.cn/szzk/jzgml.htm"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture()
def entries():
    return parse_directory(load("advisor_directory.html"), base_url=BASE_URL)


@pytest.fixture()
def profile():
    return parse_profile(load("advisor_profile.html"))


def test_directory_parses_all_entries(entries) -> None:
    assert [entry.name for entry in entries] == ["张三", "李四", "王五", "赵六"]
    assert entries[0].unit == "计算机软件研究所"
    assert entries[0].rank_group == "正高"
    assert entries[2].unit == "计算机网络研究所"
    assert entries[2].rank_group == "副高"
    assert entries[2].title == "副研究员"


def test_directory_normalizes_profile_links(entries) -> None:
    assert entries[0].profile_url == "https://www.cs.example.edu.cn/info/1101/0001.htm"


def test_directory_handles_missing_phone(entries) -> None:
    """李四的条目没有电话，邮箱仍要能解析出来。"""
    assert entries[1].email == "lisi@example.edu.cn"
    assert entries[1].title == "副教授"


def test_directory_drops_phone_numbers(entries) -> None:
    """电话在解析阶段就被丢弃，不进入任何中间结构。"""
    serialized = " ".join(repr(entry) for entry in entries)
    assert "62780001" not in serialized
    assert "62780003" not in serialized


def test_profile_extracts_research_areas(profile) -> None:
    assert profile.name == "张三"
    assert profile.research_areas == ["自然语言处理", "多模态学习", "信息检索"]


def test_profile_extracts_homepage(profile) -> None:
    assert profile.homepage == "https://example-lab.edu.cn/zhangsan"


def test_profile_extracts_publications(profile) -> None:
    assert len(profile.publications) == 2
    assert "ACL" in profile.publications[0]


def test_profile_never_keeps_phone(profile) -> None:
    assert "62780001" not in repr(profile)


def test_to_advisor_binds_research_areas_to_source(entries, profile) -> None:
    advisor = to_advisor(entries[0], profile, fetched_at=date(2026, 11, 17))

    assert advisor.unit == "计算机软件研究所"
    assert advisor.email == "zhangsan@example.edu.cn"
    assert advisor.research_tags == ["自然语言处理", "多模态学习", "信息检索"]
    assert advisor.research_summary == "长期从事自然语言处理与多模态学习研究。"
    assert advisor.recruiting is None
    assert advisor.fetched_at == date(2026, 11, 17)
    assert advisor.sources[0].doc_id == entries[0].profile_url
    assert "自然语言处理" in advisor.sources[0].quote
    assert "62780001" not in advisor.model_dump_json()


def test_to_advisor_without_profile_does_not_invent_tags(entries) -> None:
    empty = parse_profile("")
    advisor = to_advisor(entries[0], empty, fetched_at=date(2026, 11, 17))
    assert advisor.research_tags == []
    assert advisor.sources == []


def test_blocked_page_raises() -> None:
    from advisor_match.ingest.advisors import _looks_blocked

    assert _looks_blocked("<title>Making sure you're not a bot!</title>")
    assert not _looks_blocked("<html><title>教师名录</title></html>")


def test_fetch_raises_blocked_before_writing_cache(tmp_path, monkeypatch) -> None:
    """反爬页必须报错，且不落盘、不重试。"""
    import advisor_match.ingest.advisors as module

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"<html><title>Making sure you're not a bot!</title></html>"

    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(BlockedError):
        module.fetch_page("https://example.edu.cn/x.htm", cache_dir=tmp_path)
    assert list(tmp_path.glob("*.html")) == []
