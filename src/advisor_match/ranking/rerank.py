"""证据化重排：逐对（学生台账 × 导师材料）抽取匹配点与差距。

当前实现是确定性的规则版：用字符二元组重合找方向与成果的交集，
每条判断都绑定学生台账 ID 和导师原文引文。规则版的价值是可复现、可测试；
后续接入模型时，只需替换 ``_rank_points`` 的生成部分，``verify_grounding``
仍然负责兜底剔除无依据判断。
"""

from __future__ import annotations

import re

from advisor_match.ledger.build import quote_is_grounded
from advisor_match.retrieval.recall import build_query
from advisor_match.schemas import (
    Advisor,
    AdvisorMatch,
    Gap,
    LedgerRecord,
    MatchPoint,
    SourceRef,
    StudentProfile,
)

PUNCTUATION = re.compile(r"[\s，。；：、,.;:!?（）()]")


def _bigrams(text: str) -> set[str]:
    cleaned = PUNCTUATION.sub("", text)
    return {cleaned[index : index + 2] for index in range(len(cleaned) - 1)}


def _overlap(left: str, right: str) -> int:
    return len(_bigrams(left) & _bigrams(right))


def _student_text(profile: StudentProfile) -> str:
    parts = [*profile.target_directions, *profile.interests, build_query(profile)]
    return " ".join(part for part in parts if part)


def _supporting_record(tag: str, ledger: dict[str, LedgerRecord]) -> LedgerRecord | None:
    """找一条能支撑该方向的台账记录；找不到就不写这条判断。"""
    best: LedgerRecord | None = None
    best_score = 0
    for record in ledger.values():
        score = _overlap(tag, record.statement)
        if score > best_score:
            best, best_score = record, score
    return best if best_score >= 1 else None


def _advisor_sources(advisor: Advisor) -> dict[str, str]:
    return {ref.doc_id: ref.quote for ref in advisor.sources}


def rerank_one(
    profile: StudentProfile,
    advisor: Advisor,
    *,
    advisor_source_text: dict[str, str] | None = None,
) -> AdvisorMatch | None:
    """对一位导师产出匹配结果；证据不足时返回 None（宁可少给，不可编造）。"""
    sources = advisor_source_text if advisor_source_text is not None else _advisor_sources(advisor)
    if not sources:
        return None

    ledger = {record.id: record for record in profile.ledger}
    student_text = _student_text(profile)
    doc_id = next(iter(sources))

    points: list[MatchPoint] = []
    for tag in advisor.research_tags:
        overlap = _overlap(tag, student_text)
        if overlap < 1:
            continue
        record = _supporting_record(tag, ledger)
        if record is None:
            continue
        statement = (
            f"你的成果「{record.statement[:40]}」与{advisor.name}老师的研究方向「{tag}」相关"
        )
        points.append(
            MatchPoint(
                dimension="direction",
                statement=statement[:300],
                student_ledger_refs=[record.id],
                advisor_refs=[SourceRef(doc_id=doc_id, locator="研究领域", quote=tag)],
            )
        )
        if len(points) >= 3:
            break

    if not points:
        return None

    # 档位按"目标方向覆盖"判断，而不是数依据条数：
    # 全部目标方向都能在导师方向里找到对应 → 核心；部分覆盖 → 相邻；只有字面弱重合 → 需要补充。
    covered = [
        direction
        for direction in profile.target_directions
        if any(_overlap(direction, tag) >= 1 for tag in advisor.research_tags)
    ]
    if profile.target_directions and len(covered) == len(profile.target_directions):
        tier = "core"
    elif covered:
        tier = "adjacent"
    else:
        tier = "needs_support"

    gaps: list[Gap] = []
    for direction in profile.target_directions:
        if any(_overlap(direction, tag) >= 1 for tag in advisor.research_tags):
            continue
        gaps.append(Gap(statement=f"你的目标方向「{direction}」在该课题组的研究领域中体现较少"))
        if len(gaps) >= 2:
            break

    match = AdvisorMatch(
        advisor=advisor,
        tier=tier,
        match_points=points,
        gaps=gaps,
    )
    match, _ = verify_grounding(match, ledger, sources)
    if not match.match_points:
        return None
    return match


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


__all__ = ["rerank_one", "verify_grounding"]
