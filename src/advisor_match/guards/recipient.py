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

    # 正文首段必须出现导师称呼，否则视为模板未替换。
    # 允许"尊敬的X老师""X老师""X教授"；X 可以是姓氏或全名。
    first_paragraph = "".join(s.text for s in draft.paragraphs[0].sentences)
    title_pattern = r"(?:副教授|老师|教授|研究员)"
    salutation = re.search(
        rf"尊敬的\s*([^\s，,：:！!。]{{1,10}}?)\s*{title_pattern}", first_paragraph
    )
    if salutation is None:
        salutation = re.search(rf"([一-龥]{{1,4}})\s*{title_pattern}", first_paragraph)

    if salutation is not None:
        captured = salutation.group(1)
        # 姓氏或全名都算匹配；完全不同的姓氏视为串错导师。
        if captured not in contact.display_name and contact.display_name not in captured:
            issues.append(
                RecipientIssue(
                    "wrong_salutation",
                    f"正文称呼「{captured}」与联系人「{contact.display_name}」不一致",
                )
            )
    elif contact.display_name not in first_paragraph:
        issues.append(
            RecipientIssue("missing_salutation", f"正文开头未出现导师姓名「{contact.display_name}」")
        )

    return issues
