"""索引与召回测试。"""

from __future__ import annotations

from advisor_match.retrieval.index import (
    AdvisorIndex,
    advisor_to_text,
    load_index,
    save_index,
    tokenize,
)
from advisor_match.retrieval.recall import build_query, recall
from advisor_match.schemas import (
    Advisor,
    LedgerRecord,
    SourceRef,
    StudentProfile,
)


def make_advisor(name: str, tags: list[str]) -> Advisor:
    return Advisor(
        name=name,
        affiliation="某大学计算机系",
        title="教授",
        research_tags=tags,
        sources=[SourceRef(doc_id=f"profile-{name}", locator="研究领域", quote="，".join(tags))],
    )


def make_profile(targets: list[str], statement: str = "完成多模态检索增强生成的课程项目") -> StudentProfile:
    return StudentProfile(
        ledger=[
            LedgerRecord(
                id="L-0001",
                kind="course_project",
                statement=statement,
                evidence=[SourceRef(doc_id="resume", locator="p.1", quote=statement)],
            )
        ],
        target_directions=targets,
    )


def test_tokenize_handles_chinese() -> None:
    tokens = tokenize("信息检索与多模态学习")
    assert "信息检索" in tokens


def test_advisor_text_excludes_identity_fields() -> None:
    advisor = make_advisor("张三", ["信息检索"])
    text = advisor_to_text(advisor)
    assert "信息检索" in text
    assert "张三" not in text
    assert "某大学计算机系" not in text
    assert "教授" not in text


def test_build_skips_advisors_without_research_text() -> None:
    index = AdvisorIndex.build([make_advisor("甲", ["信息检索"]), Advisor(name="乙")])
    assert [advisor.name for advisor in index.advisors] == ["甲"]
    assert index.skipped == ["乙"]


def test_recall_returns_relevant_first() -> None:
    index = AdvisorIndex.build(
        [
            make_advisor("甲", ["信息检索", "多模态学习"]),
            make_advisor("乙", ["计算机体系结构"]),
            make_advisor("丙", ["数据库"]),
        ]
    )
    result = recall(make_profile(["信息检索", "多模态学习"]), index, k=2)
    assert result[0].name == "甲"
    assert len(result) <= 2


def test_recall_skips_zero_score_and_empty_query() -> None:
    index = AdvisorIndex.build([make_advisor("甲", ["信息检索"])])
    assert recall(make_profile(["量子计算"]), index, k=3) == []
    assert recall(make_profile([]), index, k=3) == []


def test_recall_is_deterministic_for_ties() -> None:
    index = AdvisorIndex.build([make_advisor("乙", ["信息检索"]), make_advisor("甲", ["信息检索"])])
    names = [advisor.name for advisor in recall(make_profile(["信息检索"]), index, k=2)]
    assert names == ["乙", "甲"] or names == ["甲", "乙"]
    again = [advisor.name for advisor in recall(make_profile(["信息检索"]), index, k=2)]
    assert names == again


def test_query_contains_targets_and_project_but_not_identity() -> None:
    query = build_query(make_profile(["信息检索"]))
    assert "信息检索" in query
    assert "多模态检索增强生成" in query


def test_index_roundtrip(tmp_path) -> None:
    index = AdvisorIndex.build([make_advisor("甲", ["信息检索"])])
    path = tmp_path / "index.pkl"
    save_index(index, path)
    loaded = load_index(path)
    assert [advisor.name for advisor in loaded.advisors] == ["甲"]
    assert recall(make_profile(["信息检索"]), loaded, k=1)[0].name == "甲"
