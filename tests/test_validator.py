"""校验器测试：无据陈述必须整段丢弃。"""

from __future__ import annotations

from advisor_match.email.validator import audit_refs, validate_draft
from advisor_match.schemas import (
    AdvisorContact,
    EmailDraft,
    EmailParagraph,
    EmailSentence,
    LedgerRecord,
    SourceRef,
)

LEDGER = {
    "L-0001": LedgerRecord(
        id="L-0001",
        kind="paper",
        statement="以二作身份发表 ACL Findings 2025 论文《X》",
        status="published",
        evidence=[SourceRef(doc_id="resume", locator="p.1", quote="ACL Findings 2025（二作）")],
    )
}


def make_draft() -> EmailDraft:
    contact = AdvisorContact(display_name="李老师", email="li@example.edu.cn")
    return EmailDraft(
        recipient=contact,
        subject="推免自荐",
        paragraphs=[
            EmailParagraph(
                role="greeting",
                sentences=[
                    EmailSentence(text="李老师您好，我是计算机系的学生，冒昧打扰。", ledger_refs=[]),
                ],
            ),
            EmailParagraph(
                role="fit",
                sentences=[
                    EmailSentence(
                        text="我以二作身份发表了 ACL Findings 2025 的论文。",
                        ledger_refs=["L-0001"],
                    ),
                    EmailSentence(
                        text="我已在 Nature 上发表三篇一作论文。",
                        ledger_refs=[],  # 无依据
                    ),
                    EmailSentence(
                        text="我还获得过图灵奖。",
                        ledger_refs=["L-9999"],  # 台账中不存在
                    ),
                ],
            ),
        ],
    )


def test_unsupported_sentences_are_dropped() -> None:
    draft, report = validate_draft(make_draft(), LEDGER)
    kept_text = "".join(s.text for p in draft.paragraphs for s in p.sentences)

    assert "ACL Findings 2025" in kept_text
    assert "Nature" not in kept_text
    assert "图灵奖" not in kept_text
    assert len(report.dropped) == 2


def test_boilerplate_without_refs_is_allowed() -> None:
    _, report = validate_draft(make_draft(), LEDGER)
    assert any("李老师您好" in text for text in report.kept)


def test_every_kept_factual_sentence_has_resolvable_ref() -> None:
    draft, _ = validate_draft(make_draft(), LEDGER)
    for paragraph in draft.paragraphs:
        for sentence in paragraph.sentences:
            for ref in sentence.ledger_refs:
                assert ref in LEDGER


def test_audit_refs_deduplicates() -> None:
    draft, _ = validate_draft(make_draft(), LEDGER)
    assert audit_refs(draft, LEDGER) == ["L-0001"]


def test_empty_paragraph_is_replaced_with_prompt() -> None:
    contact = AdvisorContact(display_name="李老师", email="li@example.edu.cn")
    draft = EmailDraft(
        recipient=contact,
        subject="推免自荐",
        paragraphs=[
            EmailParagraph(
                role="fit",
                sentences=[EmailSentence(text="我在 Nature 上发表了三篇论文。", ledger_refs=[])],
            )
        ],
    )
    cleaned, _ = validate_draft(draft, LEDGER)
    text = "".join(s.text for s in cleaned.paragraphs[0].sentences)
    assert "Nature" not in text
    assert "请依据台账自行补写" in text
