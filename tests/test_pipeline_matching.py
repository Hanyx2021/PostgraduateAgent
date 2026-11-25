"""端到端匹配测试：召回 → 重排 → 分档。"""

from __future__ import annotations

import threading

from advisor_match.pipeline import run_matching
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.schemas import Advisor, LedgerRecord, SourceRef, StudentProfile


def make_advisor(name: str, tags: list[str]) -> Advisor:
    return Advisor(
        name=name,
        research_tags=tags,
        sources=[SourceRef(doc_id=f"p-{name}", locator="研究领域", quote="，".join(tags))],
    )


def make_profile() -> StudentProfile:
    return StudentProfile(
        ledger=[
            LedgerRecord(
                id="L-0001",
                kind="course_project",
                statement="完成信息检索与多模态学习的课程项目",
                evidence=[SourceRef(doc_id="resume", locator="p.1", quote="信息检索与多模态学习")],
            )
        ],
        target_directions=["信息检索", "多模态学习"],
    )


def test_run_matching_returns_tiers_and_logs() -> None:
    advisors = [
        make_advisor("甲", ["信息检索", "多模态学习"]),
        make_advisor("乙", ["数据库"]),
        make_advisor("丙", ["信息检索"]),
        Advisor(name="丁"),  # 无研究方向，不应进入索引
    ]
    index = AdvisorIndex.build(advisors)

    result, logs = run_matching(make_profile(), index, keep_k=2)

    assert [log.stage for log in logs] == ["recall", "rerank", "tiers"]
    assert result.core and result.core[0].advisor.name == "甲"
    assert all(match.advisor.name != "乙" for match in result.core + result.adjacent)


def test_run_matching_respects_sensitive_field_guard() -> None:
    index = AdvisorIndex.build([make_advisor("甲", ["信息检索"])])
    profile = make_profile()
    profile.ledger[0].notes = "某中学"  # notes 是允许的自由文本，不应误伤
    result, _ = run_matching(profile, index, keep_k=1)
    assert result is not None


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: set[str] = set()
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        with self._lock:
            self.calls += 1
            self.prompts.add(user)
        return "不是 JSON"


def test_llm_rerank_only_calls_top_candidates() -> None:
    advisors = [make_advisor(f"甲{index}", ["信息检索"]) for index in range(5)]
    index = AdvisorIndex.build(advisors)
    llm = CountingLLM()

    run_matching(make_profile(), index, keep_k=5, llm=llm, llm_rerank_top=2)
    # 解析失败会重试，所以按"不同 prompt 数"判断调用了几个候选
    assert len(llm.prompts) == 2


def test_recall_only_candidates_are_kept_separately() -> None:
    advisors = [make_advisor("甲", ["信息检索"])]
    index = AdvisorIndex.build(advisors)
    profile = StudentProfile(
        ledger=[
            LedgerRecord(
                id="L-0001",
                kind="other",
                statement="完成英语课程项目",
                evidence=[SourceRef(doc_id="r", locator="p", quote="完成英语课程项目")],
            )
        ],
        target_directions=["信息检索"],
    )
    result, _ = run_matching(profile, index, keep_k=3)
    assert result.core == [] and result.adjacent == []
    assert [match.advisor.name for match in result.recall_only] == ["甲"]


def test_processing_is_reproducible() -> None:
    index = AdvisorIndex.build([make_advisor("甲", ["信息检索"]), make_advisor("乙", ["信息检索"])])
    first, _ = run_matching(make_profile(), index, keep_k=2)
    second, _ = run_matching(make_profile(), index, keep_k=2)
    assert first.model_dump() == second.model_dump()
