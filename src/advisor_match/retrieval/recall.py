"""候选召回：从导师库取回 ``recall_k``（默认 30）位。

召回只看研究方向，不做"学生背景"匹配；招生状态未知的导师与明确招生的导师
一同进入候选（不因为"不知道"而排除）。分数为 0 的导师不返回——检索不到就是
检索不到，不用随机候选凑数。
"""

from __future__ import annotations

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


def recall(
    profile: StudentProfile, index: AdvisorIndex, *, k: int | None = None
) -> list[Advisor]:
    """按 BM25 分数返回前 k 位，同分按姓名排序，保证结果可复现。"""
    limit = settings.retrieval.recall_k if k is None else k
    query = build_query(profile)
    if not query.strip() or index.bm25 is None:
        return []

    # 小语料下 BM25 的 IDF 可能为负，不能用"分数 > 0"判断相关性；
    # 用查询词与文档词的实际交集过滤，分数只负责排序。
    query_tokens = set(tokenize(query))
    rows: list[tuple[float, Advisor]] = []
    for advisor, doc_tokens, score in zip(index.advisors, index.tokens, index.scores(query)):
        if query_tokens & set(doc_tokens):
            rows.append((score, advisor))

    ranked = sorted(rows, key=lambda item: (-item[0], item[1].name))
    return [advisor for _, advisor in ranked][:limit]


def query_terms(profile: StudentProfile) -> list[str]:
    """查询词的 token 形式，供重排阶段判断方向重合。"""
    return tokenize(build_query(profile))


__all__ = ["build_query", "query_terms", "recall"]
