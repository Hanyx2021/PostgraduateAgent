"""邮件流水线测试：生成 → 校验 → 收件人核对。"""

from __future__ import annotations

import json

import pytest

from advisor_match.email.draft import build_draft
from advisor_match.llm.client import LLMError
from advisor_match.guards.recipient import check_recipient
from advisor_match.pipeline import RecipientMismatch, run_email
from tests.test_email_draft import make_contact, make_ledger, make_match


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


def test_run_email_returns_validated_draft() -> None:
    result = run_email(
        make_match(),
        make_contact(),
        make_ledger(),
        student_points=["我做过多模态检索增强生成的课程项目"],
        student_signature="张三",
    )
    assert result.draft.status == "draft"
    assert result.report.dropped == []
    assert result.draft.subject.startswith("推免自荐")


def test_recipient_mismatch_blocks_confirmation() -> None:
    draft = build_draft(make_match(), make_contact("李老师"), make_ledger())
    other = make_contact("王老师", "wang@example.edu.cn")
    issues = check_recipient(draft, other)
    assert {issue.code for issue in issues} >= {"advisor_mismatch", "email_mismatch"}


def test_run_email_rejects_draft_with_wrong_salutation() -> None:
    """正文称呼与联系人不一致时必须抛异常，而不是给警告。"""
    match = make_match()
    contact = make_contact("王老师", "wang@example.edu.cn")
    draft = build_draft(match, contact, make_ledger())
    draft.paragraphs[0].sentences[0].text = "尊敬的李老师：您好！"
    issues = check_recipient(draft, contact)
    assert any(issue.code == "wrong_salutation" for issue in issues)


def greeting_draft(text: str):
    contact = make_contact("黄民烈", "huang@example.edu.cn")
    draft = build_draft(make_match(), contact, make_ledger())
    draft.paragraphs[0].sentences[0].text = text
    return draft, contact


def test_surname_salutation_is_accepted() -> None:
    draft, contact = greeting_draft("黄老师：您好！")
    assert check_recipient(draft, contact) == []


def test_professor_title_is_accepted() -> None:
    draft, contact = greeting_draft("尊敬的黄教授：您好！")
    assert check_recipient(draft, contact) == []


def test_different_surname_is_rejected() -> None:
    draft, contact = greeting_draft("王老师：您好！")
    codes = {issue.code for issue in check_recipient(draft, contact)}
    assert "wrong_salutation" in codes


def test_llm_draft_is_used_when_valid() -> None:
    response = json.dumps(
        {
            "subject": "模型生成的主题",
            "paragraphs": [
                {
                    "role": "greeting",
                    "sentences": [{"text": "李老师您好！", "kind": "greeting", "ledger_refs": []}],
                },
                {
                    "role": "fit",
                    "sentences": [
                        {"text": "我发表过论文。", "kind": "fact", "ledger_refs": ["L-0001"]}
                    ],
                },
                {
                    "role": "gap",
                    "sentences": [{"text": "我还需要补充实验经验。", "kind": "intent", "ledger_refs": []}],
                },
                {
                    "role": "closing",
                    "sentences": [{"text": "期待您的回复。", "kind": "intent", "ledger_refs": []}],
                },
            ],
        },
        ensure_ascii=False,
    )
    llm = FakeLLM(response)
    draft = build_draft(make_match(), make_contact(), make_ledger(), llm=llm)
    assert draft.subject == "模型生成的主题"
    text = "".join(sentence.text for paragraph in draft.paragraphs for sentence in paragraph.sentences)
    assert "（请在发送前填写" in text, "模型漏署名时应自动补上"
    assert llm.calls


class RaisingLLM:
    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        raise LLMError("down")


def test_llm_failure_falls_back_to_template() -> None:
    draft = build_draft(make_match(), make_contact(), make_ledger(), llm=RaisingLLM())
    assert draft.subject.startswith("推免自荐")


def test_invalid_llm_output_falls_back_to_template() -> None:
    llm = FakeLLM("这不是 JSON")
    draft = build_draft(make_match(), make_contact(), make_ledger(), llm=llm)
    assert draft.subject.startswith("推免自荐")


def test_llm_missing_paragraph_falls_back_to_template() -> None:
    response = json.dumps(
        {
            "subject": "只有两段",
            "paragraphs": [
                {"role": "greeting", "sentences": [{"text": "李老师您好！", "kind": "greeting", "ledger_refs": []}]},
                {"role": "fit", "sentences": [{"text": "我发表过论文。", "kind": "fact", "ledger_refs": ["L-0001"]}]},
            ],
        },
        ensure_ascii=False,
    )
    draft = build_draft(make_match(), make_contact(), make_ledger(), llm=FakeLLM(response))
    assert draft.subject.startswith("推免自荐")


def test_llm_fabricated_refs_are_dropped() -> None:
    response = json.dumps(
        {
            "subject": "主题",
            "paragraphs": [
                {
                    "role": "greeting",
                    "sentences": [{"text": "尊敬的李老师：您好！", "kind": "greeting", "ledger_refs": []}],
                },
                {
                    "role": "fit",
                    "sentences": [
                        {"text": "我在 Nature 发表过三篇论文。", "kind": "fact", "ledger_refs": ["L-9999"]}
                    ],
                },
                {
                    "role": "gap",
                    "sentences": [{"text": "我还需要补充实验经验。", "kind": "intent", "ledger_refs": []}],
                },
                {
                    "role": "closing",
                    "sentences": [{"text": "期待您的回复。", "kind": "intent", "ledger_refs": []}],
                },
            ],
        },
        ensure_ascii=False,
    )
    result = run_email(make_match(), make_contact(), make_ledger(), llm=FakeLLM(response))
    text = "".join(s.text for p in result.draft.paragraphs for s in p.sentences)
    assert "Nature" not in text
    assert result.report.dropped


def test_recipient_error_exposes_issues() -> None:
    error = RecipientMismatch([])
    assert error.issues == []
