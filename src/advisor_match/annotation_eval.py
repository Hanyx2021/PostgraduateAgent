"""用人工标注表评估各个召回/重排策略。

输入：annotation_sheet.xlsx（人工标签）+ annotation_key.json（候选→导师）。
输出：每种策略的 Recall@10、nDCG@10、覆盖率，以及随机导师里的漏召统计。

注意：候选池是按召回结果抽样的，所以 recall 排序在这份评估里天然占优；
比较重排策略时应主要看它们彼此之间的差异。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from openpyxl import load_workbook

from advisor_match.evaluation import EvalProfile, _final_order, build_catalog_profiles, select_subset
from advisor_match.ingest.advisors import load_advisors
from advisor_match.llm.client import get_client
from advisor_match.pipeline import run_matching
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.retrieval.recall import recall
from advisor_match.retrieval.vector import load_embedding


def load_labels(sheet_path: Path) -> dict[str, dict[str, int]]:
    sheet = load_workbook(sheet_path, data_only=True).active
    labels: dict[str, dict[str, int]] = {}
    for row in range(2, sheet.max_row + 1):
        profile_id = sheet.cell(row, 1).value
        advisor = sheet.cell(row, 5).value
        label = sheet.cell(row, 12).value
        if profile_id is None or advisor is None or label is None:
            continue
        labels.setdefault(str(profile_id), {})[str(advisor)] = int(label)
    return labels


def ranking_metrics(ranked: list[str], labels: dict[str, int], k: int = 10) -> dict:
    relevant = {name for name, label in labels.items() if label >= 1}
    filtered = [name for name in ranked if name in labels]
    hits = sum(1 for name in filtered[:k] if name in relevant)

    dcg = sum(
        (2 ** labels[name] - 1) / math.log2(position + 2)
        for position, name in enumerate(filtered[:k])
        if labels.get(name, 0) > 0
    )
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum((2**label - 1) / math.log2(position + 2) for position, label in enumerate(ideal))

    return {
        "recall": hits / len(relevant) if relevant else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "coverage": (
            sum(1 for name in ranked if name in relevant) / len(relevant) if relevant else 0.0
        ),
        "list_length": len(ranked),
    }


def evaluate(
    profiles: list[EvalProfile],
    index: AdvisorIndex,
    labels: dict[str, dict[str, int]],
    *,
    llm=None,
    k: int = 10,
) -> dict:
    systems = {
        "recall": [],
        "rule": [],
        "llm": [],
        "hybrid": [],
    }

    for number, profile in enumerate(profiles, start=1):
        profile_id = f"P-{number:02d}"
        profile_labels = labels.get(profile_id, {})
        orders = {
            "recall": [advisor.name for advisor in recall(profile.profile, index, k=30)],
            "rule": _final_order(run_matching(profile.profile, index, llm=None)[0]),
        }
        if llm is not None:
            orders["llm"] = _final_order(
                run_matching(profile.profile, index, llm=llm, rerank_mode="llm")[0]
            )
            orders["hybrid"] = _final_order(
                run_matching(profile.profile, index, llm=llm, rerank_mode="hybrid")[0]
            )
        for name, ranked in orders.items():
            systems[name].append(ranking_metrics(ranked, profile_labels, k))

    summary = {}
    for name, rows in systems.items():
        if not rows:
            continue
        summary[name] = {
            key: sum(row[key] for row in rows) / len(rows)
            for key in ("recall", "ndcg", "coverage", "list_length")
        }
    return {"profiles": len(profiles), "systems": summary}


def extras_summary(
    key_path: Path, labels: dict[str, dict[str, int]]
) -> dict:
    """随机加入的导师里，有多少是人工判定相关的（衡量召回漏掉多少人）。"""
    key = json.loads(key_path.read_text(encoding="utf-8"))
    total = 0
    relevant = 0
    for profile_id, items in key.items():
        for meta in items.values():
            if not meta.get("is_extra"):
                continue
            total += 1
            label = labels.get(profile_id, {}).get(meta["advisor"], 0)
            if label >= 1:
                relevant += 1
    return {"extras": total, "relevant_extras": relevant}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用人工标注评估各策略")
    parser.add_argument("--advisors", type=Path, default=Path("data/processed/advisors.jsonl"))
    parser.add_argument("--sheet", type=Path, default=Path("data/processed/annotation_sheet.xlsx"))
    parser.add_argument("--key", type=Path, default=Path("data/processed/annotation_key.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/annotation_eval_report.json"))
    parser.add_argument("--profiles", type=int, default=10)
    args = parser.parse_args(argv)

    labels = load_labels(args.sheet)
    advisors = load_advisors(args.advisors)
    index = AdvisorIndex.build(advisors, embedding=load_embedding())
    profiles = select_subset(build_catalog_profiles(advisors, n=30), args.profiles)

    report = evaluate(profiles, index, labels, llm=get_client())
    report["extras"] = extras_summary(args.key, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for name, metrics in report["systems"].items():
        print(
            f"{name:7s} R@10={metrics['recall']:.3f}  nDCG@10={metrics['ndcg']:.3f}  "
            f"coverage={metrics['coverage']:.3f}  list={metrics['list_length']:.1f}"
        )
    extras = report["extras"]
    print(f"extras: {extras['relevant_extras']}/{extras['extras']} relevant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
