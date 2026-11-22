"""导出：匹配结果、确认留痕。

导出内容与界面看到的一致，不额外拼接学生的画像或邮箱列表；
匹配结果只含导师侧信息与理由，确认记录只含行为字段。
"""

from __future__ import annotations

import json
from dataclasses import asdict

from advisor_match.email.confirm import ConfirmationRecord
from advisor_match.schemas import MatchResult


def match_result_json(result: MatchResult) -> str:
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=1)


def confirmation_json(record: ConfirmationRecord) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, default=str, indent=1)


__all__ = ["confirmation_json", "match_result_json"]
