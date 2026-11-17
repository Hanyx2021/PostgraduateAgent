"""混合检索索引：BM25 + 向量。

建索引的文本由导师的 research_tags + 近五年论文标题/摘要组成，
不包含姓名、院系、职称等与研究方向无关的字段——检索阶段就不该让这些信息起作用。
"""

from __future__ import annotations

from dataclasses import dataclass

from advisor_match.schemas import Advisor


@dataclass
class AdvisorIndex:
    advisors: list[Advisor]
    #: 与 ``advisors`` 同序的检索文本，便于调试"为什么召回了他"。
    docs: list[str]

    @classmethod
    def build(cls, advisors: list[Advisor]) -> "AdvisorIndex":
        docs = [advisor_to_text(advisor) for advisor in advisors]
        return cls(advisors=advisors, docs=docs)


def advisor_to_text(advisor: Advisor) -> str:
    """导师的可检索文本。刻意不拼接 name / affiliation。"""
    parts: list[str] = []
    parts.extend(advisor.research_tags)
    for paper in advisor.recent_papers:
        parts.append(paper.title)
        if paper.abstract:
            parts.append(paper.abstract)
    return "\n".join(parts)


def save_index(index: AdvisorIndex, path: str) -> None:
    """TODO: 中期前用 joblib 存 BM25，向量用 numpy 矩阵 + faiss 可选。"""
    raise NotImplementedError("由韩宇轩实现")


def load_index(path: str) -> AdvisorIndex:
    raise NotImplementedError("由韩宇轩实现")
