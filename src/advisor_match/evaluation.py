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

from advisor_match.ingest.advisors import load_advisors
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.retrieval.recall import recall
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
    notes: list[str] = field(default_factory=list)


def evaluate(
    profiles: list[EvalProfile], index: AdvisorIndex, *, k: int = 10
) -> EvalReport:
    pipeline_rows: list[tuple[float, float]] = []
    catalog_rows: list[tuple[float, float]] = []
    random_rows: list[tuple[float, float]] = []
    per_profile: list[dict] = []
    catalog_ranking = _catalog_ranking(index)

    for profile in profiles:
        ranked = [advisor.name for advisor in recall(profile.profile, index, k=len(index.advisors))]
        random_ranked = _random_ranking(index, seed=profile.id)
        indexed_names = {advisor.name for advisor in index.advisors}
        metrics = {
            "id": profile.id,
            "label": profile.label,
            "ground_truth": len(profile.ground_truth),
            "indexed_ground_truth": len(set(profile.ground_truth) & indexed_names),
            "pipeline_recall": recall_at_k(ranked, profile.ground_truth, k),
            "pipeline_ndcg": ndcg_at_k(ranked, profile.ground_truth, k),
            "catalog_recall": recall_at_k(catalog_ranking, profile.ground_truth, k),
            "catalog_ndcg": ndcg_at_k(catalog_ranking, profile.ground_truth, k),
            "random_recall": recall_at_k(random_ranked, profile.ground_truth, k),
            "random_ndcg": ndcg_at_k(random_ranked, profile.ground_truth, k),
        }
        per_profile.append(metrics)
        pipeline_rows.append((metrics["pipeline_recall"], metrics["pipeline_ndcg"]))
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
    for name, rows in (
        ("pipeline", pipeline_rows),
        ("catalog", catalog_rows),
        ("random", random_rows),
    ):
        recall_score, ndcg_score = average(rows)
        report.systems.append(SystemScore(name=name, recall=recall_score, ndcg=ndcg_score))
    report.notes.append("真值来自 2027 招生目录方向，画像由模板生成，非人工双盲标注")
    report.notes.append("通用大模型直接推荐基线待模型可用后补充")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行合成画像评估")
    parser.add_argument(
        "--advisors", type=Path, default=Path("data/processed/advisors.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/eval_report.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=30, help="画像数量")
    args = parser.parse_args(argv)

    advisors = load_advisors(args.advisors)
    index = AdvisorIndex.build(advisors)
    profiles = build_catalog_profiles(advisors, n=args.n)
    report = evaluate(profiles, index, k=args.k)

    payload = {
        "k": report.k,
        "profile_count": report.profile_count,
        "systems": [asdict(system) for system in report.systems],
        "notes": report.notes,
        "per_profile": report.per_profile,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"profiles: {report.profile_count}  k={report.k}")
    for system in report.systems:
        print(f"{system.name:9s} Recall@{report.k}={system.recall:.3f}  nDCG@{report.k}={system.ndcg:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
