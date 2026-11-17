"""分档：核心契合 / 相邻方向 / 需要补充。

刻意不做"总分排序"：
- 档内顺序按确定性规则（证据条数、维度覆盖数、导师 ID），同样的输入永远同样的输出；
- 每档保留探索位，避免排在前面的永远是同一类背景；
- 输出必须三档并存，不能只给"核心契合"。
"""

from __future__ import annotations

from advisor_match.schemas import AdvisorMatch, MatchResult, Tier

DIMENSION_ORDER = ("direction", "method", "experience", "resource")


def sort_key(match: AdvisorMatch) -> tuple[int, int, str]:
    """档内排序键。不含任何学生侧属性，因此反事实测试天然成立。"""
    dimensions = {point.dimension for point in match.match_points} & set(DIMENSION_ORDER)
    return (
        -len(match.match_points),
        -len(dimensions),
        match.advisor.name,
    )


def assemble(
    matches: list[AdvisorMatch],
    *,
    keep_k: int | None = None,
    exploration_slots: int | None = None,
) -> MatchResult:
    """三档组装。

    探索位规则：按召回分数取"未进入主榜"的前 N 位，标注原因后放入相邻方向档，
    并在界面显示"探索位"标签，让学生知道这是系统主动加进来的。
    """
    raise NotImplementedError("由韩宇轩实现：按 tier 分组 + 探索位 + 生成 excluded 理由")


def tier_label(tier: Tier) -> str:
    return {
        "core": "核心契合",
        "adjacent": "相邻方向",
        "needs_support": "需要补充",
    }[tier]
