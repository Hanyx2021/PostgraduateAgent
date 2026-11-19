"""证据化重排：逐对（学生台账 × 导师材料）抽取匹配点与差距。

判定有两条通道，都必须是可核验的：
1. 字面重合（字符二元组），用于"信息检索 vs 信息检索"这类直接对应；
2. 语义相近（本地 BGE 向量余弦 ≥ 阈值），用于"人工智能 vs 机器学习"这类
   词汇差异。语义通道的措辞必须写成"语义上相近"，不能假装是字面一致。

每条判断仍绑定学生台账 ID 和导师原文引文；``verify_grounding`` 继续兜底。
"""

from __future__ import annotations

import re

import numpy as np

from advisor_match.config import settings
from advisor_match.ledger.build import quote_is_grounded
from advisor_match.retrieval.recall import build_query
from advisor_match.retrieval.vector import EmbeddingProvider
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


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.dot(left, right))


def _supporting_record(
    tag: str,
    ledger: dict[str, LedgerRecord],
    *,
    tag_vector: np.ndarray | None = None,
    record_vectors: dict[str, np.ndarray] | None = None,
) -> tuple[LedgerRecord | None, bool]:
    """找一条能支撑该方向的台账记录，返回（记录, 是否靠语义匹配）。"""
    best: LedgerRecord | None = None
    best_score = (0, 0.0)
    threshold = settings.retrieval.rerank_min_similarity
    for record in ledger.values():
        lexical = _overlap(tag, record.statement)
        semantic = _cosine(tag_vector, (record_vectors or {}).get(record.id))
        score = (lexical, semantic)
        if score > best_score:
            best, best_score = record, score
    if best is None:
        return None, False
    if best_score[0] >= 1:
        return best, False
    if best_score[1] >= threshold:
        return best, True
    return None, False


def _advisor_sources(advisor: Advisor) -> dict[str, str]:
    return {ref.doc_id: ref.quote for ref in advisor.sources}


def rerank_one(
    profile: StudentProfile,
    advisor: Advisor,
    *,
    advisor_source_text: dict[str, str] | None = None,
    embedding: EmbeddingProvider | None = None,
) -> AdvisorMatch | None:
    """对一位导师产出匹配结果；证据不足时返回 None（宁可少给，不可编造）。"""
    sources = advisor_source_text if advisor_source_text is not None else _advisor_sources(advisor)
    if not sources:
        return None

    ledger = {record.id: record for record in profile.ledger}
    student_text = _student_text(profile)
    doc_id = next(iter(sources))
    threshold = settings.retrieval.rerank_min_similarity

    tag_vectors: list[np.ndarray] | None = None
    direction_vectors: list[np.ndarray] | None = None
    student_vector: np.ndarray | None = None
    record_vectors: dict[str, np.ndarray] = {}
    if embedding is not None:
        student_vector = embedding.encode([student_text])[0]
        if advisor.research_tags:
            tag_vectors = list(embedding.encode(advisor.research_tags))
        if profile.target_directions:
            direction_vectors = list(embedding.encode(profile.target_directions))
        if ledger:
            statement_vectors = embedding.encode([record.statement for record in ledger.values()])
            record_vectors = dict(zip(ledger.keys(), statement_vectors))

    points: list[MatchPoint] = []
    for position, tag in enumerate(advisor.research_tags):
        lexical = _overlap(tag, student_text)
        tag_vector = tag_vectors[position] if tag_vectors is not None else None
        semantic = _cosine(tag_vector, student_vector)
        if lexical < 1 and semantic < threshold:
            continue

        record, used_semantic = _supporting_record(
            tag, ledger, tag_vector=tag_vector, record_vectors=record_vectors
        )
        if record is None:
            continue

        relation = "在语义上相近" if used_semantic else "相关"
        statement = (
            f"你的成果「{record.statement[:40]}」与{advisor.name}老师的研究方向"
            f"「{tag}」{relation}"
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

    # 档位按"目标方向覆盖"判断：全部覆盖 → 核心；部分覆盖 → 相邻；只有弱重合 → 需要补充。
    covered_flags = [
        any(
            _overlap(direction, tag) >= 1
            or (
                direction_vectors is not None
                and tag_vectors is not None
                and _cosine(direction_vectors[index], tag_vectors[tag_index]) >= threshold
            )
            for tag_index, tag in enumerate(advisor.research_tags)
        )
        for index, direction in enumerate(profile.target_directions)
    ]
    if profile.target_directions and all(covered_flags):
        tier = "core"
    elif any(covered_flags):
        tier = "adjacent"
    else:
        tier = "needs_support"

    gaps: list[Gap] = []
    for index, direction in enumerate(profile.target_directions):
        if covered_flags[index]:
            continue
        gaps.append(Gap(statement=f"你的目标方向「{direction}」在该课题组的研究领域中体现较少"))
        if len(gaps) >= 2:
            break

    match = AdvisorMatch(advisor=advisor, tier=tier, match_points=points, gaps=gaps)
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
