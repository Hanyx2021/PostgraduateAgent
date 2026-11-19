"""证据化重排测试：匹配点必须双向有据。"""

from __future__ import annotations

from advisor_match.ranking.rerank import rerank_one, verify_grounding
from advisor_match.schemas import (
    Advisor,
    AdvisorMatch,
    LedgerRecord,
    MatchPoint,
    SourceRef,
    StudentProfile,
)


def make_advisor(tags: list[str], *, with_source: bool = True) -> Advisor:
    return Advisor(
        name="李老师",
        research_tags=tags,
        sources=(
            [SourceRef(doc_id="profile", locator="研究领域", quote="，".join(tags))]
            if with_source
            else []
        ),
    )


def make_profile(targets: list[str], statement: str) -> StudentProfile:
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


def test_two_matching_tags_produce_core_tier() -> None:
    profile = make_profile(["信息检索", "多模态学习"], "完成多模态检索增强生成的课程项目")
    match = rerank_one(profile, make_advisor(["信息检索", "多模态学习"]))
    assert match is not None
    assert match.tier == "core"
    assert len(match.match_points) == 2
    for point in match.match_points:
        assert point.student_ledger_refs == ["L-0001"]
        assert point.advisor_refs[0].quote in {"信息检索", "多模态学习"}


def test_single_covered_target_is_core() -> None:
    profile = make_profile(["信息检索"], "做过信息检索课程项目")
    match = rerank_one(profile, make_advisor(["信息检索"]))
    assert match is not None
    assert match.tier == "core"


def test_weak_overlap_is_needs_support() -> None:
    profile = make_profile(["量子计算"], "做过信息检索课程项目")
    match = rerank_one(profile, make_advisor(["信息检索"]))
    assert match is not None
    assert match.tier == "needs_support"


def test_advisor_without_source_is_not_ranked() -> None:
    profile = make_profile(["信息检索"], "做过信息检索课程项目")
    assert rerank_one(profile, make_advisor(["信息检索"], with_source=False)) is None


def test_unmatched_direction_becomes_gap() -> None:
    profile = make_profile(["信息检索", "多模态学习"], "做过信息检索课程项目")
    match = rerank_one(profile, make_advisor(["信息检索"]))
    assert match is not None
    assert any("多模态学习" in gap.statement for gap in match.gaps)


def test_verify_grounding_drops_fabricated_quote() -> None:
    match = AdvisorMatch(
        advisor=Advisor(name="李老师"),
        tier="core",
        match_points=[
            MatchPoint(
                dimension="direction",
                statement="编造的依据",
                student_ledger_refs=["L-0001"],
                advisor_refs=[SourceRef(doc_id="profile", locator="研究领域", quote="不存在的研究方向")],
            )
        ],
    )
    cleaned, rejected = verify_grounding(
        match,
        {"L-0001": LedgerRecord(
            id="L-0001",
            kind="other",
            statement="随便",
            evidence=[SourceRef(doc_id="resume", locator="p.1", quote="随便")],
        )},
        {"profile": "，".join(["信息检索"])},
    )
    assert cleaned.match_points == []
    assert rejected
