"""数据结构层的伦理约束测试：结构本身就是防线。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from advisor_match.schemas import (
    Advisor,
    AdvisorContact,
    AdvisorMatch,
    EmailDraft,
    EmailParagraph,
    EmailSentence,
    LedgerRecord,
    SourceRef,
    StudentProfile,
)


def make_record() -> LedgerRecord:
    return LedgerRecord(
        id="L-0001",
        kind="paper",
        statement="以二作身份发表 ACL Findings 2025 论文《X》",
        status="published",
        evidence=[SourceRef(doc_id="resume", locator="p.1 第2段", quote="ACL Findings 2025, 二作")],
    )


def test_ledger_record_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        LedgerRecord(
            id="L-0001",
            kind="paper",
            statement="发表了一篇论文",
            evidence=[],
        )


def test_student_profile_rejects_identity_fields() -> None:
    """姓名、性别、籍贯等字段必须无法被塞进画像（extra=forbid）。"""
    with pytest.raises(ValidationError):
        StudentProfile(ledger=[make_record()], name="张三")  # type: ignore[call-arg]


def test_match_has_no_score_field() -> None:
    match = AdvisorMatch(
        advisor=Advisor(name="某老师"),
        tier="core",
        match_points=[],
        gaps=[],
    )
    assert "score" not in match.model_dump()
    with pytest.raises(ValidationError):
        AdvisorMatch(
            advisor=Advisor(name="某老师"),
            tier="core",
            match_points=[],
            gaps=[],
            score=91.5,  # type: ignore[call-arg]
        )


def test_advisor_has_no_synthetic_id() -> None:
    """导师以姓名作为唯一天然键，系统不另设 ID。"""
    assert "id" not in Advisor.model_fields


def test_draft_cannot_be_sent() -> None:
    """草稿只有 draft / student_confirmed 两种状态，没有 sent。"""
    contact = AdvisorContact(display_name="某老师", email="a@example.edu.cn")
    with pytest.raises(ValidationError):
        EmailDraft(
            recipient=contact,
            subject="推免自荐",
            paragraphs=[
                EmailParagraph(
                    role="greeting",
                    sentences=[EmailSentence(text="某老师您好，我是……", ledger_refs=[])],
                )
            ],
            status="sent",  # type: ignore[arg-type]
        )


def test_ledger_id_format_enforced() -> None:
    with pytest.raises(ValidationError):
        LedgerRecord(
            id="record-1",
            kind="skill",
            statement="熟悉 PyTorch",
            evidence=[SourceRef(doc_id="resume", locator="p.1", quote="PyTorch")],
        )
