"""发信前收件人核对。

系统没有发信权限，但草稿会被学生复制到邮件客户端。误发的主要原因是
收件人串行 / 正文称呼与收件人不一致，这里只做确定性检查，不依赖模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from advisor_match.schemas import AdvisorContact, EmailDraft


@dataclass(frozen=True)
class RecipientIssue:
    code: str
    message: str


def check_recipient(draft: EmailDraft, contact: AdvisorContact) -> list[RecipientIssue]:
    """检查草稿与"当前选中的联系人"是否一致。

    返回空列表才允许进入逐句确认。调用方必须把它做成硬阻塞，而不是警告。
    """
    issues: list[RecipientIssue] = []

    if draft.recipient.display_name != contact.display_name:
        issues.append(
            RecipientIssue("advisor_mismatch", "草稿收件人与当前联系人不是同一位导师")
        )
    if draft.recipient.email.lower() != contact.email.lower():
        issues.append(
            RecipientIssue("email_mismatch", "草稿收件邮箱与当前联系人邮箱不一致")
        )

    # 正文首段必须出现导师姓名，否则视为模板未替换。
    first_paragraph = "".join(s.text for s in draft.paragraphs[0].sentences)
    salutation = re.search(r"尊敬的\s*([^\s，,：:！!。]{1,10}?)\s*老师", first_paragraph)
    if salutation is not None:
        # "尊敬的X老师"中的 X 必须是联系人姓名的子串，防止串行到另一位导师。
        if salutation.group(1) not in contact.display_name:
            issues.append(
                RecipientIssue(
                    "wrong_salutation",
                    f"正文称呼「{salutation.group(1)}老师」与联系人「{contact.display_name}」不一致",
                )
            )
    elif contact.display_name not in first_paragraph:
        issues.append(
            RecipientIssue("missing_salutation", f"正文开头未出现导师姓名「{contact.display_name}」")
        )

    return issues
