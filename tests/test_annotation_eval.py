"""人工标注评估指标测试。"""

from __future__ import annotations

import json
from pathlib import Path

from advisor_match.annotation_eval import extras_summary, ranking_metrics


def test_perfect_ranking_scores_one() -> None:
    labels = {"甲": 2, "乙": 1, "丙": 0}
    metrics = ranking_metrics(["甲", "乙", "丙"], labels)
    assert metrics["recall"] == 1.0
    assert metrics["coverage"] == 1.0
    assert metrics["ndcg"] == 1.0


def test_missing_relevant_reduces_coverage() -> None:
    labels = {"甲": 2, "乙": 1, "丙": 0}
    metrics = ranking_metrics(["丙", "甲"], labels)
    assert metrics["coverage"] == 0.5
    assert 0 < metrics["recall"] < 1


def test_extras_summary_counts_relevant(tmp_path: Path) -> None:
    key = {
        "P-01": {
            "P-01-C01": {"advisor": "甲", "is_extra": True},
            "P-01-C02": {"advisor": "乙", "is_extra": True},
            "P-01-C03": {"advisor": "丙", "is_extra": False},
        }
    }
    path = tmp_path / "key.json"
    path.write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    summary = extras_summary(path, {"P-01": {"甲": 2, "乙": 0, "丙": 2}})
    assert summary == {"extras": 2, "relevant_extras": 1}
