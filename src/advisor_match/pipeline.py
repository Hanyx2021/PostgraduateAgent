"""端到端流水线（对齐幻灯片 6 的五步）。

    材料结构化 → 候选召回 → 证据化重排 → 分档呈现 → 邮件与核验
                                                    ↑ 学生在这里做决定

第 4、5 步的"学生决定"体现在接口上：``run_matching`` 只产出 ``MatchResult``，
``run_email`` 只产出 ``EmailDraft``（status=draft），两者都没有副作用。
"""

from __future__ import annotations

from dataclasses import dataclass

from advisor_match.guards.sensitive import assert_no_sensitive_fields
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.schemas import (
    AdvisorContact,
    AdvisorMatch,
    EmailDraft,
    LedgerRecord,
    MatchResult,
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
) -> tuple[MatchResult, list[StageLog]]:
    """执行第 2–4 步。入口处强制检查画像不含敏感字段。"""
    assert_no_sensitive_fields(profile)
    raise NotImplementedError("串起 recall → rerank → assemble，并返回各阶段日志")


def run_email(
    match: AdvisorMatch,
    contact: AdvisorContact,
    ledger: dict[str, LedgerRecord],
    *,
    student_points: list[str] | None = None,
) -> EmailDraft:
    """执行第 5 步的前半段：生成 + 校验草稿。不确认、不发送。"""
    raise NotImplementedError("串起 build_draft → validate_draft → check_recipient")


__all__ = ["run_matching", "run_email", "StageLog"]
