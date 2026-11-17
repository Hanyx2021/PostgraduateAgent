"""生成四段式自荐邮件草稿。

四段：问候与来意 / 契合点 / 差距与补充 / 收尾。

两条硬约束写在接口上：
- 返回值是 ``EmailDraft``，调用方**无法**把它变成已发送状态（schema 没有该状态）；
- 每个事实句必须带 ``ledger_refs``，由 ``validator.validate_draft`` 二次把关。
"""

from __future__ import annotations

from advisor_match.llm.client import LLMClient
from advisor_match.schemas import (
    AdvisorContact,
    AdvisorMatch,
    EmailDraft,
    LedgerRecord,
)


def build_draft(
    match: AdvisorMatch,
    contact: AdvisorContact,
    ledger: dict[str, LedgerRecord],
    *,
    student_voice_samples: list[str] | None = None,
    llm: LLMClient | None = None,
) -> EmailDraft:
    """生成草稿。

    ``student_voice_samples``：学生先自写的三句主张。系统只修改表达，
    不替换主张本身——这样"写什么"的决定权留在学生手里（幻灯片 5）。
    """
    raise NotImplementedError("由陈英豪实现：接入 LLM 并按四段式产出 EmailSentence")


def render_plain_text(draft: EmailDraft, ledger: dict[str, LedgerRecord]) -> str:
    """渲染成可直接粘贴的纯文本。

    逐句确认通过后才应调用；本函数不发送任何东西。
    """
    lines = [f"收件人：{draft.recipient.display_name} <{draft.recipient.email}>", f"主题：{draft.subject}", ""]
    for paragraph in draft.paragraphs:
        lines.append("".join(sentence.text for sentence in paragraph.sentences))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
