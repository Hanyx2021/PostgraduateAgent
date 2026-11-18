"""邮件草稿生成测试：四段式、依据绑定、学生自写要点映射。"""

from __future__ import annotations

from advisor_match.email.draft import (
    SIGNATURE_PLACEHOLDER,
    build_draft,
    match_student_points,
)
from advisor_match.email.validator import validate_draft
from advisor_match.schemas import (
    Advisor,
    AdvisorContact,
    AdvisorMatch,
    EmailDraft,
    Gap,
    LedgerRecord,
    MatchPoint,
    SourceRef,
)


def make_ledger() -> dict[str, LedgerRecord]:
    return {
        "L-0001": LedgerRecord(
            id="L-0001",
            kind="paper",
            statement="以二作身份发表 ACL Findings 2025 论文",
            status="published",
            evidence=[SourceRef(doc_id="resume", locator="p.1", quote="ACL Findings 2025 二作")],
        ),
        "L-0002": LedgerRecord(
            id="L-0002",
            kind="course_project",
            statement="完成多模态检索增强生成的课程项目",
            evidence=[SourceRef(doc_id="resume", locator="p.2", quote="多模态检索增强生成课程项目")],
        ),
    }


def make_contact(name: str = "李老师", email: str = "li@example.edu.cn") -> AdvisorContact:
    return AdvisorContact(display_name=name, email=email)


def make_match() -> AdvisorMatch:
    return AdvisorMatch(
        advisor=Advisor(name="李老师", research_tags=["信息检索"]),
        tier="core",
        match_points=[
            MatchPoint(
                dimension="direction",
                statement="课题组的方向与我的多模态检索经历相符",
                student_ledger_refs=["L-0001", "L-0002"],
                advisor_refs=[SourceRef(doc_id="profile", locator="研究领域", quote="信息检索")],
            )
        ],
        gaps=[Gap(statement="缺少大规模实验经验", student_ledger_refs=["L-0002"])],
    )


def test_template_has_four_paragraphs() -> None:
    draft = build_draft(make_match(), make_contact(), make_ledger())
    assert [paragraph.role for paragraph in draft.paragraphs] == ["greeting", "fit", "gap", "closing"]


def test_every_fact_sentence_has_resolvable_refs() -> None:
    ledger = make_ledger()
    draft = build_draft(make_match(), make_contact(), ledger)
    for paragraph in draft.paragraphs:
        for sentence in paragraph.sentences:
            if sentence.kind == "fact":
                assert sentence.ledger_refs
                assert all(ref in ledger for ref in sentence.ledger_refs)


def test_greeting_contains_advisor_name_and_signature() -> None:
    draft = build_draft(make_match(), make_contact(), make_ledger(), student_signature="张三")
    text = "".join(s.text for p in draft.paragraphs for s in p.sentences)
    assert "李老师" in text
    assert "张三" in text


def test_missing_signature_uses_placeholder() -> None:
    draft = build_draft(make_match(), make_contact(), make_ledger())
    text = "".join(s.text for p in draft.paragraphs for s in p.sentences)
    assert SIGNATURE_PLACEHOLDER in text


def test_student_point_is_kept_only_when_grounded() -> None:
    ledger = make_ledger()
    points = ["我做过多模态检索增强生成的课程项目", "我曾经获得图灵奖"]
    matched = match_student_points(points, ledger)
    assert matched == [("我做过多模态检索增强生成的课程项目", "L-0002")]

    draft = build_draft(
        make_match(), make_contact(), ledger, student_voice_samples=points
    )
    text = "".join(s.text for p in draft.paragraphs for s in p.sentences)
    assert "多模态检索增强生成" in text
    assert "图灵奖" not in text


def test_validator_keeps_whole_template_draft() -> None:
    draft, report = validate_draft(build_draft(make_match(), make_contact(), make_ledger()), make_ledger())
    assert report.dropped == []
    assert report.kept


def test_draft_starts_in_draft_state() -> None:
    draft = build_draft(make_match(), make_contact(), make_ledger())
    assert isinstance(draft, EmailDraft)
    assert draft.status == "draft"
