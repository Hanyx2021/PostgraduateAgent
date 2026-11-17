"""候选召回：从导师库取回 ``recall_k``（默认 30）位。

召回只看研究方向，不做"学生背景"匹配；招生状态未知的导师与明确招生的导师
一同进入候选（不因为"不知道"而排除）。
"""

from __future__ import annotations

from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.schemas import Advisor, StudentProfile


def build_query(profile: StudentProfile) -> str:
    """查询词由目标方向 + 兴趣 + 台账中的方法/项目主题拼接。

    不使用台账里的机构、获奖等级等与研究方向无关的字段。
    """
    parts = list(profile.target_directions) + list(profile.interests)
    for record in profile.ledger:
        if record.kind in {"paper", "course_project"}:
            parts.append(record.statement)
    return " ".join(parts)


def recall(
    profile: StudentProfile, index: AdvisorIndex, *, k: int | None = None
) -> list[Advisor]:
    raise NotImplementedError("由韩宇轩实现：BM25 与向量分数归一化后加权，按导师姓名去重")
