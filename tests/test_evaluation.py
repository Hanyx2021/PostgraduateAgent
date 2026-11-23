"""评估集与指标测试。"""

from __future__ import annotations

from advisor_match.evaluation import (
    build_catalog_profiles,
    evaluate,
    evaluate_final_rankings,
    ndcg_at_k,
    recall_at_k,
    select_subset,
)
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.schemas import AdmissionEntry, Advisor, SourceRef


class FakeEmbedding:
    def encode(self, texts):
        import numpy as np

        return np.vstack([np.ones(2, dtype=np.float32) for _ in texts])


def make_advisor(name: str, tags: list[str], direction: str) -> Advisor:
    return Advisor(
        name=name,
        research_tags=tags,
        sources=[SourceRef(doc_id=f"p-{name}", locator="研究领域", quote="，".join(tags))],
        admissions=[
            AdmissionEntry(
                year=2027,
                program_code="081200",
                program_name="计算机科学与技术",
                program_type="学术学位",
                direction_code="01" if direction == tags[0] else "02",
                direction=direction,
                source="https://example.edu.cn/catalog",
            )
        ],
    )


def test_recall_at_k() -> None:
    ranked = ["甲", "乙", "丙"]
    assert recall_at_k(ranked, ["乙"], k=2) == 1.0
    assert recall_at_k(ranked, ["甲", "丙"], k=1) == 0.5
    assert recall_at_k([], ["甲"], k=10) == 0.0


def test_ndcg_at_k_prefers_relevant_first() -> None:
    assert ndcg_at_k(["甲", "乙"], ["甲"], k=2) == 1.0
    assert ndcg_at_k(["乙", "甲"], ["甲"], k=2) < 1.0
    assert ndcg_at_k([], ["甲"], k=10) == 0.0


def test_build_profiles_from_catalog_directions() -> None:
    advisors = [
        make_advisor("甲", ["信息检索"], "信息检索"),
        make_advisor("乙", ["信息检索"], "信息检索"),
        make_advisor("丙", ["数据库"], "数据库"),
    ]
    profiles = build_catalog_profiles(advisors, n=30)
    assert len(profiles) == 2
    profile = next(item for item in profiles if item.label.endswith("信息检索"))
    assert set(profile.ground_truth) == {"甲", "乙"}
    assert profile.profile.target_directions == ["信息检索"]


def test_evaluate_beats_or_matches_baselines_on_controlled_data() -> None:
    advisors = [
        make_advisor("乙", ["数据库"], "数据库"),
        make_advisor("甲", ["信息检索"], "信息检索"),
    ]
    index = AdvisorIndex.build(advisors)
    profiles = build_catalog_profiles(advisors, n=10, min_relevant=1)
    report = evaluate(profiles, index, k=1)

    by_name = {system.name: system for system in report.systems}
    assert by_name["pipeline"].recall == 1.0
    # 主表顺序把"乙"排在前面，目录基线对"信息检索"画像命中为 0
    assert by_name["catalog"].recall < by_name["pipeline"].recall
    assert all(0.0 <= system.ndcg <= 1.0 for system in report.systems)


def test_report_has_program_breakdown_and_weight_grid() -> None:
    advisors = [make_advisor("甲", ["信息检索"], "信息检索")]
    index = AdvisorIndex.build(advisors, embedding=FakeEmbedding())
    profiles = build_catalog_profiles(advisors, n=10, min_relevant=1)
    report = evaluate(profiles, index, k=1)

    assert "081200" in report.by_program
    assert len(report.grid) == 5
    assert {row["bm25_weight"] for row in report.grid} == {0.0, 0.3, 0.5, 0.7, 1.0}


class FastLLM:
    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        return "不是 JSON"


def test_final_ranking_comparison_falls_back_to_rule() -> None:
    advisors = [make_advisor("甲", ["信息检索"], "信息检索")]
    index = AdvisorIndex.build(advisors)
    profiles = build_catalog_profiles(advisors, n=10, min_relevant=1)
    subset = select_subset(profiles, 1)

    comparison = evaluate_final_rankings(
        subset, index, llm=FastLLM(), k=1, llm_rerank_top=1
    )
    assert comparison["profiles"] == 1
    assert comparison["rule"] == comparison["llm"] == comparison["hybrid"]
    assert comparison["per_profile"][0]["label"].endswith("信息检索")


def test_report_contains_per_profile_details() -> None:
    advisors = [make_advisor("甲", ["信息检索"], "信息检索")]
    index = AdvisorIndex.build(advisors)
    profiles = build_catalog_profiles(advisors, n=10, min_relevant=1)
    report = evaluate(profiles, index, k=10)
    row = report.per_profile[0]
    assert row["indexed_ground_truth"] == 1
    assert "pipeline_recall" in row and "catalog_recall" in row and "random_recall" in row
    assert report.notes
