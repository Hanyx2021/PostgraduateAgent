"""分档测试：确定性排序、保留探索位、未列出的理由。"""

from __future__ import annotations

from advisor_match.ranking.tiers import assemble, selected_count, sort_key
from advisor_match.schemas import Advisor, AdvisorMatch, MatchPoint, SourceRef


def point(index: int) -> MatchPoint:
    return MatchPoint(
        dimension="direction",
        statement=f"依据 {index}",
        student_ledger_refs=["L-0001"],
        advisor_refs=[SourceRef(doc_id="profile", locator="研究领域", quote="方向")],
    )


def match(name: str, points: int, tier: str = "adjacent") -> AdvisorMatch:
    return AdvisorMatch(
        advisor=Advisor(name=name),
        tier=tier,  # type: ignore[arg-type]
        match_points=[point(index) for index in range(points)],
    )


def test_sort_key_prefers_more_evidence_then_name() -> None:
    assert sort_key(match("甲", 2)) < sort_key(match("乙", 1))
    assert sort_key(match("乙", 1)) < sort_key(match("甲", 1))


def test_assemble_keeps_three_tiers_and_marks_exploration() -> None:
    matches = [
        match("甲", 3, "core"),
        match("乙", 2, "core"),
        match("丙", 1, "adjacent"),
        match("丁", 1, "adjacent"),
        match("戊", 1, "needs_support"),
        match("己", 1, "needs_support"),
    ]
    result = assemble(matches, keep_k=3, exploration_slots=2)

    assert {item.advisor.name for item in result.core} == {"甲", "乙"}
    assert {item.advisor.name for item in result.adjacent} == {"丙", "丁"}
    assert {item.advisor.name for item in result.needs_support} == {"己"}
    assert set(result.exploration) == {"丙", "己"}
    assert set(result.excluded) == {"戊"}


def test_assemble_reports_excluded_reason() -> None:
    matches = [match(name, 1) for name in ["甲", "乙", "丙", "丁", "戊"]]
    result = assemble(matches, keep_k=2, exploration_slots=1)
    assert selected_count(result) == 3
    assert len(result.excluded) == 2
    assert all("未进入前 2 位" in reason for reason in result.excluded.values())


def test_assemble_is_deterministic() -> None:
    matches = [match(name, 1) for name in ["丙", "甲", "乙"]]
    first = assemble(matches, keep_k=2, exploration_slots=1)
    second = assemble(list(reversed(matches)), keep_k=2, exploration_slots=1)
    assert [item.advisor.name for item in first.adjacent] == [
        item.advisor.name for item in second.adjacent
    ]
