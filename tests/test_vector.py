"""向量召回测试：用假 embedding 验证融合逻辑，不下载模型。"""

from __future__ import annotations

import numpy as np

from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.retrieval.recall import recall
from advisor_match.schemas import Advisor, LedgerRecord, SourceRef, StudentProfile

DIM = 4
KEYWORDS = {
    "信息检索": 0,
    "检索": 0,
    "机器学习": 0,  # 与"信息检索"在语义空间里相近
    "多模态": 1,
    "数据库": 2,
}


class FakeEmbedding:
    """按关键词桶生成 one-hot 向量，模拟"语义相近但字面不同"。"""

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(DIM, dtype=np.float32)
            for keyword, index in KEYWORDS.items():
                if keyword in text:
                    vector[index] += 1.0
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.vstack(rows)


def make_advisor(name: str, tags: list[str]) -> Advisor:
    return Advisor(
        name=name,
        research_tags=tags,
        sources=[SourceRef(doc_id=f"p-{name}", locator="研究领域", quote="，".join(tags))],
    )


def make_profile(target: str) -> StudentProfile:
    return StudentProfile(
        ledger=[
            LedgerRecord(
                id="L-0001",
                kind="course_project",
                statement=f"完成{target}方向的课程项目",
                evidence=[SourceRef(doc_id="resume", locator="p.1", quote=target)],
            )
        ],
        target_directions=[target],
    )


def test_vector_scores_available_only_with_embedding() -> None:
    advisors = [make_advisor("甲", ["机器学习"])]
    assert AdvisorIndex.build(advisors).vector_scores("信息检索") is None

    index = AdvisorIndex.build(advisors, embedding=FakeEmbedding())
    scores = index.vector_scores("信息检索")
    assert scores is not None and scores[0] > 0.9


def test_bm25_misses_semantic_match_but_vector_catches_it() -> None:
    advisors = [make_advisor("甲", ["机器学习"]), make_advisor("乙", ["数据库"])]
    profile = make_profile("信息检索")

    bm25_only = recall(
        profile,
        AdvisorIndex.build(advisors),
        k=5,
        bm25_weight=1.0,
        vector_weight=0.0,
    )
    assert bm25_only == [], "字面无交集时 BM25 召回应为空"

    hybrid_index = AdvisorIndex.build(advisors, embedding=FakeEmbedding())
    hybrid = recall(profile, hybrid_index, k=5, bm25_weight=0.5, vector_weight=0.5)
    assert [advisor.name for advisor in hybrid] == ["甲"]

    vector_only = recall(profile, hybrid_index, k=5, bm25_weight=0.0, vector_weight=1.0)
    assert [advisor.name for advisor in vector_only] == ["甲"]


def test_vector_threshold_filters_unrelated() -> None:
    advisors = [make_advisor("乙", ["数据库"])]
    index = AdvisorIndex.build(advisors, embedding=FakeEmbedding())
    result = recall(make_profile("信息检索"), index, k=5, bm25_weight=0.0, vector_weight=1.0)
    assert result == []


def test_index_vectors_roundtrip(tmp_path) -> None:
    from advisor_match.retrieval.index import load_index, save_index

    advisors = [make_advisor("甲", ["机器学习"])]
    index = AdvisorIndex.build(advisors, embedding=FakeEmbedding())
    path = tmp_path / "index.pkl"
    save_index(index, path)

    loaded = load_index(path, embedding=FakeEmbedding())
    assert loaded.vectors is not None
    assert loaded.vector_scores("信息检索")[0] > 0.9
