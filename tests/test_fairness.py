"""公平性测试：反事实一致性（幻灯片 5 的验收项）。

两层：
1. 结构层：画像里不存在敏感字段，且无法通过 extra 注入；
2. 行为层：只改变姓名/性别/生源地，推荐结果（含排序与各维度得分）必须完全一致。
   第 2 层依赖 ranking 实现，目前标记 skip，实现后必须补上。
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from advisor_match.guards.sensitive import (
    SENSITIVE_FIELD_NAMES,
    assert_no_sensitive_fields,
)
from advisor_match.schemas import StudentProfile


def test_profile_field_names_are_not_sensitive() -> None:
    forbidden = set(StudentProfile.model_fields) & SENSITIVE_FIELD_NAMES
    assert not forbidden, f"StudentProfile 出现了敏感字段：{forbidden}"


def test_profile_rejects_sensitive_extra() -> None:
    with pytest.raises(ValidationError):
        StudentProfile.model_validate(
            {
                "ledger": [
                    {
                        "id": "L-0001",
                        "kind": "skill",
                        "statement": "熟悉 PyTorch",
                        "evidence": [
                            {"doc_id": "resume", "locator": "p.1", "quote": "PyTorch"}
                        ],
                    }
                ],
                "gender": "女",
            }
        )


def test_guard_catches_sensitive_field_in_nested_model() -> None:
    class Inner(BaseModel):
        hometown: str

    class Outer(BaseModel):
        profile: Inner

    with pytest.raises(ValueError, match="hometown"):
        assert_no_sensitive_fields(Outer(profile=Inner(hometown="某省")))


def test_counterfactual_recommendation_is_identical() -> None:
    """同一份台账 + 仅身份文本不同的画像，推荐结果必须完全一致。

    身份信息不允许出现在字段里，这里通过自由文本（constraints）注入姓名、
    性别、生源地，验证它们不会进入查询词或重排判断。
    """
    from advisor_match.pipeline import run_matching
    from advisor_match.retrieval.index import AdvisorIndex
    from advisor_match.schemas import Advisor, LedgerRecord, SourceRef, StudentProfile

    advisors = [
        Advisor(
            name="甲",
            research_tags=["信息检索", "多模态学习"],
            sources=[SourceRef(doc_id="p-甲", locator="研究领域", quote="信息检索，多模态学习")],
        ),
        Advisor(
            name="乙",
            research_tags=["数据库"],
            sources=[SourceRef(doc_id="p-乙", locator="研究领域", quote="数据库")],
        ),
    ]
    index = AdvisorIndex.build(advisors)
    ledger = [
        LedgerRecord(
            id="L-0001",
            kind="course_project",
            statement="完成信息检索与多模态学习的课程项目",
            evidence=[SourceRef(doc_id="resume", locator="p.1", quote="信息检索与多模态学习")],
        )
    ]

    def profile(constraint: str) -> StudentProfile:
        return StudentProfile(
            ledger=ledger,
            target_directions=["信息检索"],
            constraints=[constraint],
        )

    first, _ = run_matching(profile("我叫张三，女，来自某省"), index, keep_k=2)
    second, _ = run_matching(profile("我叫李四，男，来自另一省"), index, keep_k=2)
    third, _ = run_matching(profile(""), index, keep_k=2)

    assert first.model_dump() == second.model_dump() == third.model_dump()
    assert first.core or first.adjacent
