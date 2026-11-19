"""分档：核心契合 / 相邻方向 / 需要补充。

刻意不做"总分排序"：
- 档内顺序按确定性规则（证据条数、维度覆盖数、姓名），同样输入永远同样输出；
- 每档保留探索位，避免排在前面的永远是同一类背景；
- 输出三档并存，不能只给"核心契合"。
"""

from __future__ import annotations

from advisor_match.config import settings
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

    探索位规则：在主榜之外按排序取前 N 位，放进它们各自的档位，
    并在 ``MatchResult.exploration`` 中记录名字，界面据此显示"探索位"标签。
    """
    keep = settings.retrieval.keep_k if keep_k is None else keep_k
    slots = settings.exploration_slots if exploration_slots is None else exploration_slots

    ordered = sorted(matches, key=sort_key)
    main, rest = ordered[:keep], ordered[keep:]
    exploration = rest[:slots]
    selected = main + exploration

    result = MatchResult()
    for match in selected:
        getattr(result, match.tier).append(match)

    result.exploration = [match.advisor.name for match in exploration]
    for match in rest[slots:]:
        result.excluded[match.advisor.name] = (
            f"未进入前 {keep} 位（其中 {slots} 个为探索位），证据不足以单独列出"
        )
    return result


def tier_label(tier: Tier) -> str:
    return {
        "core": "核心契合",
        "adjacent": "相邻方向",
        "needs_support": "需要补充",
    }[tier]


def selected_count(result: MatchResult) -> int:
    return len(result.core) + len(result.adjacent) + len(result.needs_support)


__all__ = ["assemble", "selected_count", "sort_key", "tier_label"]
