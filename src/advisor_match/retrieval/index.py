"""混合检索索引：BM25（当前）+ 向量（预留）。

索引文本只由研究方向、研究简介与论文标题/摘要组成，**不包含姓名、职称、单位**，
这样检索阶段就不会让与研究无关的身份信息影响结果。

没有研究信息的导师不进入索引（检索不到总比按空文档乱配好），
``build`` 会返回被跳过的姓名，供数据质量统计。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from advisor_match.schemas import Advisor


def tokenize(text: str) -> list[str]:
    """中文用 jieba 分词，英文/数字保持原样，过滤空白。"""
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


@dataclass
class AdvisorIndex:
    advisors: list[Advisor]
    docs: list[str]
    tokens: list[list[str]]
    skipped: list[str]
    bm25: BM25Okapi | None = None

    @classmethod
    def build(cls, advisors: list[Advisor]) -> "AdvisorIndex":
        kept: list[Advisor] = []
        docs: list[str] = []
        skipped: list[str] = []
        for advisor in advisors:
            text = advisor_to_text(advisor)
            if not text.strip():
                skipped.append(advisor.name)
                continue
            kept.append(advisor)
            docs.append(text)
        tokens = [tokenize(doc) for doc in docs]
        tokens = [token if token else [""] for token in tokens]
        bm25 = BM25Okapi(tokens) if tokens else None
        return cls(advisors=kept, docs=docs, tokens=tokens, skipped=skipped, bm25=bm25)

    def scores(self, query: str) -> list[float]:
        if self.bm25 is None:
            return []
        return list(self.bm25.get_scores(tokenize(query)))


def advisor_to_text(advisor: Advisor) -> str:
    """导师的可检索文本。刻意不拼接 name / affiliation / title。"""
    parts: list[str] = list(advisor.research_tags)
    if advisor.research_summary:
        parts.append(advisor.research_summary)
    for paper in advisor.recent_papers:
        parts.append(paper.title)
        if paper.abstract:
            parts.append(paper.abstract)
    return "\n".join(part for part in parts if part)


def save_index(index: AdvisorIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "advisors": [advisor.model_dump(mode="json") for advisor in index.advisors],
        "docs": index.docs,
        "tokens": index.tokens,
        "skipped": index.skipped,
    }
    path.write_bytes(pickle.dumps(payload))


def load_index(path: Path) -> AdvisorIndex:
    payload = pickle.loads(path.read_bytes())
    advisors = [Advisor.model_validate(item) for item in payload["advisors"]]
    bm25 = BM25Okapi(payload["tokens"]) if payload["tokens"] else None
    return AdvisorIndex(
        advisors=advisors,
        docs=payload["docs"],
        tokens=payload["tokens"],
        skipped=payload["skipped"],
        bm25=bm25,
    )


__all__ = ["AdvisorIndex", "advisor_to_text", "load_index", "save_index", "tokenize"]
