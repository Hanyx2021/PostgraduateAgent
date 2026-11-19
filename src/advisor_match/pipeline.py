"""端到端流水线（对齐幻灯片 6 的五步）。

    材料结构化 → 候选召回 → 证据化重排 → 分档呈现 → 邮件与核验
                                                    ↑ 学生在这里做决定

第 4、5 步的"学生决定"体现在接口上：``run_matching`` 只产出 ``MatchResult``，
``run_email`` 只产出 ``EmailDraft``（status=draft），两者都没有副作用。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from advisor_match.guards.sensitive import assert_no_sensitive_fields
from advisor_match.ingest.advisors import load_advisors
from advisor_match.ranking.rerank import rerank_one
from advisor_match.ranking.tiers import assemble, selected_count, tier_label
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.retrieval.recall import recall
from advisor_match.schemas import (
    AdvisorContact,
    AdvisorMatch,
    EmailDraft,
    LedgerRecord,
    MatchResult,
    SourceRef,
    StudentProfile,
)


@dataclass
class StageLog:
    stage: str
    input_count: int
    output_count: int
    note: str = ""


def run_matching(
    profile: StudentProfile,
    index: AdvisorIndex,
    *,
    advisor_source_text: dict[str, dict[str, str]] | None = None,
    keep_k: int | None = None,
) -> tuple[MatchResult, list[StageLog]]:
    """执行第 2–4 步。入口处强制检查画像不含敏感字段。

    ``advisor_source_text`` 可选，键为导师姓名；缺省时使用导师自带的 sources。
    """
    assert_no_sensitive_fields(profile)

    candidates = recall(profile, index)
    logs = [StageLog("recall", len(index.advisors), len(candidates))]

    matches: list[AdvisorMatch] = []
    for advisor in candidates:
        sources = advisor_source_text.get(advisor.name) if advisor_source_text else None
        match = rerank_one(profile, advisor, advisor_source_text=sources)
        if match is not None:
            matches.append(match)
    logs.append(StageLog("rerank", len(candidates), len(matches)))

    result = assemble(matches, keep_k=keep_k)
    logs.append(StageLog("tiers", len(matches), selected_count(result)))
    return result, logs


def run_email(
    match: AdvisorMatch,
    contact: AdvisorContact,
    ledger: dict[str, LedgerRecord],
    *,
    student_points: list[str] | None = None,
) -> EmailDraft:
    """执行第 5 步的前半段：生成 + 校验草稿。不确认、不发送。"""
    raise NotImplementedError("由陈英豪在 feat/email-agent 实现")


def _demo_ledger(statements: list[str]) -> dict[str, LedgerRecord]:
    ledger: dict[str, LedgerRecord] = {}
    for index, statement in enumerate(statements, start=1):
        record_id = f"L-{index:04d}"
        ledger[record_id] = LedgerRecord(
            id=record_id,
            kind="other",
            statement=statement,
            evidence=[SourceRef(doc_id="demo", locator="命令行输入", quote=statement)],
        )
    return ledger


def _print_result(result: MatchResult, logs: list[StageLog]) -> None:
    for log in logs:
        print(f"[{log.stage}] {log.input_count} -> {log.output_count}")
    print()

    for tier in ("core", "adjacent", "needs_support"):
        matches = getattr(result, tier)
        if not matches:
            continue
        print(f"{tier_label(tier)}（{len(matches)} 位）")
        for match in matches:
            flag = " [探索位]" if match.advisor.name in result.exploration else ""
            print(f"  - {match.advisor.name}{flag}：{len(match.match_points)} 条依据")
            for point in match.match_points:
                print(f"      · {point.statement}")
            for gap in match.gaps:
                print(f"      ? {gap.statement}")
        print()

    if result.excluded:
        print("未列入：")
        for name, reason in result.excluded.items():
            print(f"  - {name}：{reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行召回 → 重排 → 分档")
    parser.add_argument(
        "--advisors",
        type=Path,
        default=Path("data/processed/advisors.jsonl"),
        help="导师主表 jsonl",
    )
    parser.add_argument("--target", default="", help="目标方向，逗号分隔")
    parser.add_argument("--interest", default="", help="研究兴趣，逗号分隔")
    parser.add_argument(
        "--statement",
        action="append",
        default=[],
        help="演示用学生成果陈述，可重复；正式使用应传入台账文件",
    )
    parser.add_argument("--top", type=int, default=None, help="主榜人数，默认取配置值")
    args = parser.parse_args(argv)

    if not args.statement:
        parser.error("至少提供一条 --statement，否则画像没有可核验材料")

    advisors = load_advisors(args.advisors)
    index = AdvisorIndex.build(advisors)
    profile = StudentProfile(
        ledger=list(_demo_ledger(args.statement).values()),
        target_directions=[item.strip() for item in args.target.split(",") if item.strip()],
        interests=[item.strip() for item in args.interest.split(",") if item.strip()],
    )

    result, logs = run_matching(profile, index, keep_k=args.top)
    print(f"索引导师 {len(index.advisors)} 位，跳过无研究方向 {len(index.skipped)} 位")
    _print_result(result, logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_matching", "run_email", "StageLog", "main"]
