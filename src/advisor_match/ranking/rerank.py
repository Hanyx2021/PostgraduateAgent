"""证据化重排：逐对（学生台账 × 导师论文/主页）抽取匹配点与差距。

这里的模型输出必须经过 ``verify_grounding``：
- ``student_ledger_refs`` 必须存在于台账；
- ``advisor_refs`` 的 quote 必须能在抓取的导师材料里找到；
否则该条判断被丢弃。这样"研究兴趣高度吻合"这类空话无法过关。
"""

from __future__ import annotations

from advisor_match.ledger.build import quote_is_grounded
from advisor_match.schemas import Advisor, AdvisorMatch, LedgerRecord, MatchPoint


def rerank_one(
    profile_ledger: dict[str, LedgerRecord],
    advisor: Advisor,
    *,
    advisor_source_text: dict[str, str],
) -> AdvisorMatch | None:
    """对一位导师产出匹配结果；证据不足时返回 None（宁可少给，不可编造）。"""
    raise NotImplementedError("由韩宇轩实现：调用 LLM + verify_grounding 后组装 AdvisorMatch")


def verify_grounding(
    match: AdvisorMatch,
    ledger: dict[str, LedgerRecord],
    advisor_source_text: dict[str, str],
) -> tuple[AdvisorMatch, list[str]]:
    """剔除无依据的匹配点与差距，返回（清理后的结果, 剔除原因）。"""
    kept_points: list[MatchPoint] = []
    rejected: list[str] = []

    for point in match.match_points:
        missing_refs = [ref for ref in point.student_ledger_refs if ref not in ledger]
        if missing_refs:
            rejected.append(f"台账中不存在 {missing_refs}")
            continue
        bad_quotes = [
            ref for ref in point.advisor_refs
            if ref.doc_id not in advisor_source_text
            or not quote_is_grounded(ref.quote, advisor_source_text[ref.doc_id])
        ]
        if bad_quotes:
            rejected.append(f"导师材料中未找到引文：{[r.quote[:30] for r in bad_quotes]}")
            continue
        kept_points.append(point)

    match.match_points = kept_points
    if not kept_points:
        return match, rejected + ["没有任何一条匹配点通过证据校验"]
    return match, rejected
