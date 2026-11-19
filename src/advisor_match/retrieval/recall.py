"""候选召回：BM25 + 向量混合，取回 ``recall_k``（默认 30）位。

召回只看研究方向，不做"学生背景"匹配；招生状态未知的导师与明确招生的导师
一同进入候选（不因为"不知道"而排除）。

融合方式：两路分数各自 min-max 归一化后加权（权重见 ``config.settings``）。
候选集合是"词面有交集"与"向量相似度过阈值"的并集——纯向量是为了接住
"人工智能 vs 机器学习"这类词汇差异，阈值防止无关导师混进来。
"""

from __future__ import annotations

import numpy as np

from advisor_match.config import settings
from advisor_match.retrieval.index import AdvisorIndex, tokenize
from advisor_match.schemas import Advisor, StudentProfile


def build_query(profile: StudentProfile) -> str:
    """查询词由目标方向 + 兴趣 + 台账中的论文/项目主题拼接。

    不使用台账里的机构、获奖等级等与研究方向无关的字段。
    """
    parts: list[str] = list(profile.target_directions) + list(profile.interests)
    for record in profile.ledger:
        if record.kind in {"paper", "course_project"}:
            parts.append(record.statement)
    return " ".join(part for part in parts if part)


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def recall(
    profile: StudentProfile,
    index: AdvisorIndex,
    *,
    k: int | None = None,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
) -> list[Advisor]:
    """返回前 k 位，同分按姓名排序，保证结果可复现。"""
    limit = settings.retrieval.recall_k if k is None else k
    weight_bm25 = settings.retrieval.bm25_weight if bm25_weight is None else bm25_weight
    weight_vector = settings.retrieval.vector_weight if vector_weight is None else vector_weight

    query = build_query(profile)
    if not query.strip() or index.bm25 is None:
        return []

    bm25_scores = np.asarray(index.scores(query), dtype=float)
    query_tokens = set(tokenize(query))

    vector_scores = None
    if weight_vector > 0 and index.vectors is not None:
        vector_scores = index.vector_scores(query)

    if vector_scores is not None:
        combined = weight_bm25 * _minmax(bm25_scores) + weight_vector * _minmax(vector_scores)
        candidates = [
            position
            for position, (tokens, similarity) in enumerate(zip(index.tokens, vector_scores))
            if (query_tokens & set(tokens))
            or similarity >= settings.retrieval.vector_min_similarity
        ]
    else:
        # 没有向量模型时退回纯 BM25：用词面交集过滤，分数只负责排序。
        combined = _minmax(bm25_scores)
        candidates = [
            position
            for position, tokens in enumerate(index.tokens)
            if query_tokens & set(tokens)
        ]

    ranked = sorted(candidates, key=lambda position: (-combined[position], index.advisors[position].name))
    return [index.advisors[position] for position in ranked][:limit]


def query_terms(profile: StudentProfile) -> list[str]:
    """查询词的 token 形式，供重排阶段判断方向重合。"""
    return tokenize(build_query(profile))


__all__ = ["build_query", "query_terms", "recall"]
