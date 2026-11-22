"""离线评估：合成画像 + Recall@10 / nDCG@10。

真值来源：2027 招生目录的「专业 + 方向 → 在列导师」。它与召回使用的
研究标签来自不同字段，避免"用同一份数据考自己"。

合成方式是机械的：每个方向生成一份画像，目标方向 = 目录方向，
台账陈述 = "完成{方向}方向的课程项目与研究"。这不是人类标注，
报告里必须注明；它的用途是回归测试与横向比较，不替代双盲标注。

基线：
- ``catalog``：按导师主表顺序取前 k 位（"只按名单排"的朴素做法）；
- ``random``：固定随机种子打乱（下界参考）。
通用大模型直接推荐的基线需要模型可用后补上。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from advisor_match.config import settings
from advisor_match.ingest.advisors import load_advisors
from advisor_match.llm.client import get_client
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.retrieval.recall import recall
from advisor_match.retrieval.vector import load_embedding
from advisor_match.pipeline import run_matching
from advisor_match.schemas import Advisor, LedgerRecord, SourceRef, StudentProfile


@dataclass
class EvalProfile:
    id: str
    label: str
    profile: StudentProfile
    ground_truth: list[str]


def build_catalog_profiles(
    advisors: list[Advisor],
    *,
    n: int = 30,
    min_relevant: int = 1,
    max_relevant: int = 20,
) -> list[EvalProfile]:
    """按招生目录方向生成画像。方向导师数落在 [min, max] 才入选。"""
    groups: dict[tuple[str, str, str], list[Advisor]] = defaultdict(list)
    for advisor in advisors:
        for admission in advisor.admissions:
            key = (admission.program_code, admission.direction_code, admission.direction)
            groups[key].append(advisor)

    eligible = [
        (len(group), key, group)
        for key, group in groups.items()
        if min_relevant <= len(group) <= max_relevant
    ]
    # 先按导师数降序（样本更稳定），再按方向名排序保证可复现。
    eligible.sort(key=lambda item: (-item[0], item[1][2], item[1][1]))
    selected = eligible if n is None else eligible[:n]

    profiles: list[EvalProfile] = []
    for index, (_, key, group) in enumerate(selected, start=1):
        program_code, direction_code, direction = key
        statement = f"完成{direction}方向的课程项目与研究"
        ledger = [
            LedgerRecord(
                id="L-0001",
                kind="course_project",
                statement=statement,
                evidence=[SourceRef(doc_id="synthetic", locator="评估集", quote=statement)],
            )
        ]
        profiles.append(
            EvalProfile(
                id=f"E-{index:03d}",
                label=f"{program_code}/{direction_code} {direction}",
                profile=StudentProfile(ledger=ledger, target_directions=[direction]),
                ground_truth=[advisor.name for advisor in group],
            )
        )
    return profiles


def recall_at_k(ranked: list[str], relevant: list[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: list[str], k: int = 10) -> float:
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, name in enumerate(ranked[:k])
        if name in relevant_set
    )
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(len(relevant_set), k)))
    return dcg / ideal if ideal else 0.0


def _catalog_ranking(index: AdvisorIndex) -> list[str]:
    return [advisor.name for advisor in index.advisors]


def _random_ranking(index: AdvisorIndex, seed: int) -> list[str]:
    names = [advisor.name for advisor in index.advisors]
    random.Random(seed).shuffle(names)
    return names


@dataclass
class SystemScore:
    name: str
    recall: float = 0.0
    ndcg: float = 0.0


@dataclass
class EvalReport:
    k: int
    profile_count: int
    systems: list[SystemScore] = field(default_factory=list)
    per_profile: list[dict] = field(default_factory=list)
    #: 按专业代码聚合的混合召回指标。
    by_program: dict[str, dict] = field(default_factory=dict)
    #: BM25/向量权重网格，仅在启用向量时计算。
    grid: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def evaluate(
    profiles: list[EvalProfile], index: AdvisorIndex, *, k: int = 10
) -> EvalReport:
    modes = [
        ("bm25", 1.0, 0.0),
        ("pipeline", settings.retrieval.bm25_weight, settings.retrieval.vector_weight),
    ]
    if index.vectors is not None:
        modes.append(("vector", 0.0, 1.0))

    mode_rows: dict[str, list[tuple[float, float]]] = {name: [] for name, _, _ in modes}
    catalog_rows: list[tuple[float, float]] = []
    random_rows: list[tuple[float, float]] = []
    per_profile: list[dict] = []
    catalog_ranking = _catalog_ranking(index)
    indexed_names = {advisor.name for advisor in index.advisors}

    for profile in profiles:
        metrics: dict = {
            "id": profile.id,
            "label": profile.label,
            "ground_truth": len(profile.ground_truth),
            "indexed_ground_truth": len(set(profile.ground_truth) & indexed_names),
        }
        for name, bm25_weight, vector_weight in modes:
            ranked = [
                advisor.name
                for advisor in recall(
                    profile.profile,
                    index,
                    k=len(index.advisors),
                    bm25_weight=bm25_weight,
                    vector_weight=vector_weight,
                )
            ]
            metrics[f"{name}_recall"] = recall_at_k(ranked, profile.ground_truth, k)
            metrics[f"{name}_ndcg"] = ndcg_at_k(ranked, profile.ground_truth, k)
            mode_rows[name].append((metrics[f"{name}_recall"], metrics[f"{name}_ndcg"]))

        random_ranked = _random_ranking(index, seed=profile.id)
        metrics["catalog_recall"] = recall_at_k(catalog_ranking, profile.ground_truth, k)
        metrics["catalog_ndcg"] = ndcg_at_k(catalog_ranking, profile.ground_truth, k)
        metrics["random_recall"] = recall_at_k(random_ranked, profile.ground_truth, k)
        metrics["random_ndcg"] = ndcg_at_k(random_ranked, profile.ground_truth, k)
        per_profile.append(metrics)
        catalog_rows.append((metrics["catalog_recall"], metrics["catalog_ndcg"]))
        random_rows.append((metrics["random_recall"], metrics["random_ndcg"]))

    def average(rows: list[tuple[float, float]]) -> tuple[float, float]:
        if not rows:
            return 0.0, 0.0
        return (
            sum(row[0] for row in rows) / len(rows),
            sum(row[1] for row in rows) / len(rows),
        )

    report = EvalReport(k=k, profile_count=len(profiles), per_profile=per_profile)
    for name, rows in [
        *((name, mode_rows[name]) for name, _, _ in modes),
        ("catalog", catalog_rows),
        ("random", random_rows),
    ]:
        recall_score, ndcg_score = average(rows)
        report.systems.append(SystemScore(name=name, recall=recall_score, ndcg=ndcg_score))
    # 分专业看：避免平均值掩盖某个专业方向的系统性偏差。
    programs: dict[str, list[dict]] = defaultdict(list)
    for row in per_profile:
        program = row["label"].split("/")[0] if "/" in row["label"] else "unknown"
        programs[program].append(row)
    for program, rows in sorted(programs.items()):
        report.by_program[program] = {
            "profiles": len(rows),
            "recall": sum(row["pipeline_recall"] for row in rows) / len(rows),
            "ndcg": sum(row["pipeline_ndcg"] for row in rows) / len(rows),
        }

    # 权重网格：验证默认 0.5/0.5 是否合理，而不是拍脑袋。
    if index.vectors is not None:
        for weight in (0.0, 0.3, 0.5, 0.7, 1.0):
            rows: list[tuple[float, float]] = []
            for profile in profiles:
                ranked = [
                    advisor.name
                    for advisor in recall(
                        profile.profile,
                        index,
                        k=len(index.advisors),
                        bm25_weight=weight,
                        vector_weight=1.0 - weight,
                    )
                ]
                rows.append(
                    (
                        recall_at_k(ranked, profile.ground_truth, k),
                        ndcg_at_k(ranked, profile.ground_truth, k),
                    )
                )
            recall_score, ndcg_score = average(rows)
            report.grid.append(
                {
                    "bm25_weight": weight,
                    "vector_weight": round(1.0 - weight, 1),
                    "recall": recall_score,
                    "ndcg": ndcg_score,
                }
            )

    report.notes.append("真值来自 2027 招生目录方向，画像由模板生成，非人工双盲标注")
    report.notes.append("通用大模型直接推荐基线待模型可用后补充")
    return report


def _final_order(result) -> list[str]:
    """三档按 core → adjacent → needs_support 展开成最终推荐顺序。"""
    return [
        match.advisor.name
        for tier in ("core", "adjacent", "needs_support")
        for match in getattr(result, tier)
    ]


def select_subset(profiles: list[EvalProfile], n: int) -> list[EvalProfile]:
    """按专业轮询抽样，避免对比只覆盖某一个专业。"""
    groups: dict[str, list[EvalProfile]] = defaultdict(list)
    for profile in profiles:
        groups[profile.label.split("/")[0]].append(profile)
    lists = [groups[key] for key in sorted(groups)]
    selected: list[EvalProfile] = []
    index = 0
    while len(selected) < min(n, len(profiles)):
        added = False
        for items in lists:
            if index < len(items) and len(selected) < n:
                selected.append(items[index])
                added = True
        if not added:
            break
        index += 1
    return selected


def evaluate_final_rankings(
    profiles: list[EvalProfile],
    index: AdvisorIndex,
    *,
    llm,
    k: int = 10,
    llm_rerank_top: int = 8,
) -> dict:
    """规则版 vs 模型版：比较最终推荐列表的 Recall@k / nDCG@k。"""
    rows: list[dict] = []
    rule_scores: list[tuple[float, float]] = []
    llm_scores: list[tuple[float, float]] = []

    for profile in profiles:
        rule_result, _ = run_matching(profile.profile, index, llm=None)
        llm_result, _ = run_matching(
            profile.profile,
            index,
            llm=llm,
            llm_rerank_top=llm_rerank_top,
        )
        rule_rank = _final_order(rule_result)
        llm_rank = _final_order(llm_result)
        row = {
            "label": profile.label,
            "ground_truth": len(profile.ground_truth),
            "rule_recall": recall_at_k(rule_rank, profile.ground_truth, k),
            "rule_ndcg": ndcg_at_k(rule_rank, profile.ground_truth, k),
            "llm_recall": recall_at_k(llm_rank, profile.ground_truth, k),
            "llm_ndcg": ndcg_at_k(llm_rank, profile.ground_truth, k),
        }
        rows.append(row)
        rule_scores.append((row["rule_recall"], row["rule_ndcg"]))
        llm_scores.append((row["llm_recall"], row["llm_ndcg"]))

    def average(scores: list[tuple[float, float]]) -> tuple[float, float]:
        if not scores:
            return 0.0, 0.0
        return (
            sum(item[0] for item in scores) / len(scores),
            sum(item[1] for item in scores) / len(scores),
        )

    rule_recall, rule_ndcg = average(rule_scores)
    llm_recall, llm_ndcg = average(llm_scores)
    return {
        "profiles": len(profiles),
        "rule": {"recall": rule_recall, "ndcg": rule_ndcg},
        "llm": {"recall": llm_recall, "ndcg": llm_ndcg},
        "per_profile": rows,
        "notes": "最终推荐列表（三档展开），真值仍为 2027 目录方向",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行合成画像评估")
    parser.add_argument(
        "--advisors", type=Path, default=Path("data/processed/advisors.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/eval_report.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=30, help="画像数量")
    parser.add_argument("--no-vector", action="store_true", help="只评估 BM25")
    parser.add_argument("--rerank-compare", action="store_true", help="规则版 vs 模型版重排对比")
    parser.add_argument("--subset", type=int, default=8, help="对比用的画像数量")
    parser.add_argument("--llm-rerank-top", type=int, default=8)
    args = parser.parse_args(argv)

    advisors = load_advisors(args.advisors)
    embedding = None if args.no_vector else load_embedding()
    index = AdvisorIndex.build(advisors, embedding=embedding)
    print(f"vector: {'on' if embedding else 'off'}")
    profiles = build_catalog_profiles(advisors, n=args.n)
    report = evaluate(profiles, index, k=args.k)

    payload = {
        "k": report.k,
        "profile_count": report.profile_count,
        "systems": [asdict(system) for system in report.systems],
        "by_program": report.by_program,
        "grid": report.grid,
        "notes": report.notes,
        "per_profile": report.per_profile,
    }

    comparison = None
    if args.rerank_compare:
        if not settings.llm_api_key:
            print("rerank comparison skipped: 缺少 AM_LLM_API_KEY")
        else:
            subset = select_subset(profiles, args.subset)
            comparison = evaluate_final_rankings(
                subset,
                index,
                llm=get_client(),
                k=args.k,
                llm_rerank_top=args.llm_rerank_top,
            )
            payload["rerank_comparison"] = comparison
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"profiles: {report.profile_count}  k={report.k}")
    for system in report.systems:
        print(f"{system.name:9s} Recall@{report.k}={system.recall:.3f}  nDCG@{report.k}={system.ndcg:.3f}")

    if report.by_program:
        print("\nby program:")
        for program, metrics in report.by_program.items():
            print(
                f"  {program}: n={metrics['profiles']}  "
                f"R={metrics['recall']:.3f}  N={metrics['ndcg']:.3f}"
            )

    if report.grid:
        print("\nweight grid (bm25 / vector):")
        for row in report.grid:
            print(
                f"  {row['bm25_weight']:.1f}/{row['vector_weight']:.1f}: "
                f"R={row['recall']:.3f}  N={row['ndcg']:.3f}"
            )

    if comparison is not None:
        print(f"\nrerank comparison (n={comparison['profiles']}):")
        print(
            f"  rule: R={comparison['rule']['recall']:.3f}  "
            f"N={comparison['rule']['ndcg']:.3f}"
        )
        print(
            f"  llm:  R={comparison['llm']['recall']:.3f}  "
            f"N={comparison['llm']['ndcg']:.3f}"
        )

    worst = sorted(report.per_profile, key=lambda row: row["pipeline_ndcg"])[:5]
    print("\nworst 5 profiles:")
    for row in worst:
        print(
            f"  {row['label']}: R={row['pipeline_recall']:.2f} N={row['pipeline_ndcg']:.2f} "
            f"(GT={row['ground_truth']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
