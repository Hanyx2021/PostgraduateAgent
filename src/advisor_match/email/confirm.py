"""逐句确认。

产品含义：学生在界面上逐句点"认可 / 修改 / 删除"，全部处理后草稿才能变成
``student_confirmed``，并生成一份留痕记录（谁、何时、确认了哪些句子、依据是哪条台账）。

注意：这里没有 ``send``。确认只改变草稿状态，发送由学生自己的邮件客户端完成。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from advisor_match.schemas import AdvisorContact, EmailDraft


@dataclass
class ConfirmationRecord:
    """一封已确认草稿的留痕。可写入本地日志，供事后追责（幻灯片 4 "责任"）。"""

    advisor_name: str
    confirmed_at: datetime
    sentence_count: int
    refs_used: list[str] = field(default_factory=list)
    student_edit_ratio: float = 0.0


def mark_confirmed(draft: EmailDraft, *, all_sentences_confirmed: bool) -> EmailDraft:
    """仅当每一句都被学生处理过时，才允许改为已确认。"""
    if not all_sentences_confirmed:
        raise ValueError("仍有句子未确认，不能标记为 student_confirmed")
    draft.status = "student_confirmed"
    return draft


def build_record(
    draft: EmailDraft, contact: AdvisorContact, *, refs_used: list[str], edited: int
) -> ConfirmationRecord:
    total = sum(len(p.sentences) for p in draft.paragraphs)
    return ConfirmationRecord(
        advisor_name=contact.display_name,
        confirmed_at=datetime.now(),
        sentence_count=total,
        refs_used=refs_used,
        student_edit_ratio=0.0 if total == 0 else edited / total,
    )


def save_confirmation(record: ConfirmationRecord, path: Path) -> None:
    """追加到本地日志。只记行为与台账 ID，不记邮件正文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")


def load_confirmations(path: Path, *, limit: int = 10) -> list[dict]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]


__all__ = [
    "ConfirmationRecord",
    "build_record",
    "load_confirmations",
    "mark_confirmed",
    "save_confirmation",
]
