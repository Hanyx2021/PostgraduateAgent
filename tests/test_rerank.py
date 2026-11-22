"""证据化重排测试：匹配点必须双向有据。"""

from __future__ import annotations

import json

from advisor_match.ranking.rerank import rerank_one, verify_grounding
from advisor_match.schemas import (
    AdmissionEntry,
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


class FakeEmbedding:
    """把"信息检索"和"机器学习"映射到同一维，模拟语义相近。"""

    def encode(self, texts):
        import numpy as np

        rows = []
        for text in texts:
            vector = np.zeros(3, dtype=np.float32)
            if "检索" in text or "机器学习" in text:
                vector[0] = 1.0
            if "数据库" in text:
                vector[1] = 1.0
            rows.append(vector)
        return np.vstack(rows)


def test_semantic_overlap_can_create_evidence_point() -> None:
    profile = make_profile(["信息检索"], "做过信息检索课程项目")
    match = rerank_one(profile, make_advisor(["机器学习"]), embedding=FakeEmbedding())
    assert match is not None
    assert "语义上相近" in match.match_points[0].statement
    assert match.tier == "core"


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        return self.response


def llm_response(quote: str = "信息检索", statement: str = "模型给出的判断") -> str:
    return json.dumps(
        {
            "match_points": [
                {
                    "dimension": "direction",
                    "statement": statement,
                    "student_ledger_refs": ["L-0001"],
                    "source_doc_id": "profile",
                    "advisor_quote": quote,
                }
            ]
        },
        ensure_ascii=False,
    )


def make_advisor_with_admission(direction: str) -> Advisor:
    return Advisor(
        name="陈老师",
        admissions=[
            AdmissionEntry(
                year=2027,
                program_code="081200",
                program_name="计算机科学与技术",
                program_type="学术学位",
                direction_code="01",
                direction=direction,
                source="https://yzbm.example.edu.cn/catalog",
            )
        ],
    )


def test_admission_direction_can_supply_evidence_without_profile() -> None:
    profile = make_profile(["信息检索"], "完成信息检索方向的项目")
    match = rerank_one(profile, make_advisor_with_admission("信息检索"))
    assert match is not None
    assert match.tier == "core"
    point = match.match_points[0]
    assert point.advisor_refs[0].doc_id == "https://yzbm.example.edu.cn/catalog"
    assert "招生目录" in point.statement
    assert point.advisor_refs[0].quote == "信息检索"


def test_admission_does_not_duplicate_tag_evidence() -> None:
    advisor = make_advisor(["信息检索"])
    advisor.admissions = make_advisor_with_admission("信息检索").admissions
    profile = make_profile(["信息检索"], "完成信息检索方向的项目")
    match = rerank_one(profile, advisor)
    assert match is not None
    assert len(match.match_points) == 1


def test_llm_points_are_used_when_quote_is_grounded() -> None:
    profile = make_profile(["信息检索"], "做过信息检索课程项目")
    match = rerank_one(profile, make_advisor(["信息检索"]), llm=FakeLLM(llm_response()))
    assert match is not None
    assert match.match_points[0].statement == "模型给出的判断"
    assert match.match_points[0].advisor_refs[0].quote == "信息检索"


def test_llm_must_pick_single_source_item() -> None:
    advisor = make_advisor(["信息检索", "多模态学习"])
    profile = make_profile(["信息检索"], "做过信息检索课程项目")
    joined = llm_response(quote="信息检索，多模态学习")
    match = rerank_one(profile, advisor, llm=FakeLLM(joined))
    assert match is not None
    # 拼接的引文不算"单个条目"，模型判断被丢弃，退回规则版
    assert all("模型给出的判断" not in point.statement for point in match.match_points)


def test_llm_fabricated_quote_falls_back_to_rule_points() -> None:
    profile = make_profile(["信息检索"], "做过信息检索课程项目")
    match = rerank_one(
        profile, make_advisor(["信息检索"]), llm=FakeLLM(llm_response(quote="不存在的研究方向"))
    )
    assert match is not None
    assert all("模型给出的判断" not in point.statement for point in match.match_points)
    assert match.match_points


def test_invalid_llm_output_falls_back_to_rule_points() -> None:
    profile = make_profile(["信息检索"], "做过信息检索课程项目")
    match = rerank_one(profile, make_advisor(["信息检索"]), llm=FakeLLM("这不是 JSON"))
    assert match is not None
    assert match.match_points


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
