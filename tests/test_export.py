"""导出内容测试：结构与隐私边界。"""

from __future__ import annotations

import json
from datetime import datetime

from advisor_match.email.confirm import ConfirmationRecord
from advisor_match.product.export import confirmation_json, match_result_json
from advisor_match.schemas import Advisor, AdvisorMatch, MatchResult


def test_match_result_json_contains_tiers_and_no_score() -> None:
    result = MatchResult(
        core=[AdvisorMatch(advisor=Advisor(name="李老师"), tier="core")],
        excluded={"王老师": "证据不足"},
    )
    payload = json.loads(match_result_json(result))
    assert payload["core"][0]["advisor"]["name"] == "李老师"
    assert payload["core"][0]["tier"] == "core"
    assert payload["excluded"]["王老师"] == "证据不足"
    assert "score" not in payload["core"][0]


def test_confirmation_json_only_has_behavior_fields() -> None:
    record = ConfirmationRecord(
        advisor_name="李老师",
        confirmed_at=datetime(2026, 11, 22, 10, 0),
        sentence_count=7,
        refs_used=["L-0001"],
        student_edit_ratio=0.2,
    )
    payload = json.loads(confirmation_json(record))
    assert payload["advisor_name"] == "李老师"
    assert payload["refs_used"] == ["L-0001"]
    assert "text" not in payload and "subject" not in payload
