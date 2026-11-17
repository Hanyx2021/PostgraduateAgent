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
    if contact.display_name not in first_paragraph:
        issues.append(
            RecipientIssue("missing_salutation", f"正文开头未出现导师姓名「{contact.display_name}」")
        )

    # 正文里不应残留其他导师的称呼（串行时最常见的错误）。
    all_text = "".join(s.text for p in draft.paragraphs for s in p.sentences)
    for other in re.findall(r"([一-龥]{1,3})老师", all_text):
        if other not in contact.display_name:
            issues.append(
                RecipientIssue("possible_wrong_salutation", f"正文出现其他称呼「{other}老师」")
            )
            break

    return issues
