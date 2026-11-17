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


@pytest.mark.skip(reason="等待 ranking 实现：构造仅姓名/性别/生源地不同的三份画像，断言 MatchResult 完全相同")
def test_counterfactual_recommendation_is_identical() -> None:
    """TODO: 用同一份台账 + 三份仅身份不同的画像跑 run_matching，比较 model_dump()。

    注意：由于 StudentProfile 不允许身份字段，这里的变化应当通过“在自由文本里
    写入姓名/籍贯”来模拟，以验证脱敏器真的把这些内容处理掉。
    """
