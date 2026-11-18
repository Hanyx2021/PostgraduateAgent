"""端到端流水线（对齐幻灯片 6 的五步）。

    材料结构化 → 候选召回 → 证据化重排 → 分档呈现 → 邮件与核验
                                                    ↑ 学生在这里做决定

第 4、5 步的"学生决定"体现在接口上：``run_matching`` 只产出 ``MatchResult``，
``run_email`` 只产出 ``EmailDraft``（status=draft），两者都没有副作用。
"""

from __future__ import annotations

from dataclasses import dataclass

from advisor_match.email.draft import build_draft
from advisor_match.email.validator import validate_draft
from advisor_match.guards.recipient import RecipientIssue, check_recipient
from advisor_match.guards.sensitive import assert_no_sensitive_fields
from advisor_match.llm.client import LLMClient
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.schemas import (
    AdvisorContact,
    AdvisorMatch,
    EmailDraft,
    LedgerRecord,
    MatchResult,
    StudentProfile,
    ValidationReport,
)


@dataclass
class StageLog:
    stage: str
    input_count: int
    output_count: int
    note: str = ""


@dataclass
class EmailResult:
    """草稿 + 校验报告。没有"已发送"字段，也不应该有。"""

    draft: EmailDraft
    report: ValidationReport


class RecipientMismatch(ValueError):
    """收件人核对未通过。必须硬阻塞，不能降级为警告。"""

    def __init__(self, issues: list[RecipientIssue]) -> None:
        self.issues = issues
        super().__init__("；".join(issue.message for issue in issues))


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
    student_signature: str | None = None,
    llm: LLMClient | None = None,
) -> EmailResult:
    """执行第 5 步的前半段：生成 + 校验草稿。不确认、不发送。

    顺序固定：先生成，再删无据句，最后核对收件人。收件人不一致时抛异常，
    因为"点一下复制"之后就没有第二次拦截机会了。
    """
    draft = build_draft(
        match,
        contact,
        ledger,
        student_voice_samples=student_points,
        student_signature=student_signature,
        llm=llm,
    )
    draft, report = validate_draft(draft, ledger)
    issues = check_recipient(draft, contact)
    if issues:
        raise RecipientMismatch(issues)
    return EmailResult(draft=draft, report=report)


__all__ = ["run_matching", "run_email", "EmailResult", "RecipientMismatch", "StageLog"]
