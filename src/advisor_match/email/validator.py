"""claim–evidence 校验器：无据陈述不进草稿。

规则（对齐幻灯片 5 "诚实"）：
- 每个事实性句子必须携带 ``ledger_refs``，且每个 ref 都能在台账里查到；
- 查不到证据的句子**整段丢弃**，不是加个黄标留着——标签会被人忽略；
- 不含事实主张的寒暄句（``ledger_refs`` 为空且命中 ``GREETING_PATTERNS``）放行。

被丢弃的句子要留存，用于评估"模型想编多少"，但不进入学生对外的草稿。
"""

from __future__ import annotations

import re

from advisor_match.schemas import (
    EmailDraft,
    EmailSentence,
    LedgerRecord,
    ValidationReport,
)

#: 允许不含事实依据的礼节性句子。
GREETING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(尊敬的|您好|老师您好|感谢|祝|此致|顺颂|期待)"),
    re.compile(r"(打扰|冒昧)"),
)


def _is_boilerplate(text: str) -> bool:
    return any(pattern.search(text) for pattern in GREETING_PATTERNS)


def validate_draft(
    draft: EmailDraft, ledger: dict[str, LedgerRecord]
) -> tuple[EmailDraft, ValidationReport]:
    """返回（清理后的草稿, 校验报告）。

    ``ledger`` 必须覆盖草稿引用的所有记录；缺失的 ref 视同无据。
    """
    report = ValidationReport()

    for paragraph in draft.paragraphs:
        kept: list[EmailSentence] = []
        for sentence in paragraph.sentences:
            if sentence.ledger_refs:
                missing = [ref for ref in sentence.ledger_refs if ref not in ledger]
                if missing:
                    report.dropped.append(sentence.text)
                    continue
                kept.append(sentence)
                report.kept.append(sentence.text)
            elif _is_boilerplate(sentence.text):
                kept.append(sentence)
                report.kept.append(sentence.text)
            else:
                report.dropped.append(sentence.text)

        # 整段被清空时保留一个占位段，提示学生自己补写，而不是静默消失。
        if not kept:
            kept = [
                EmailSentence(
                    text="【此处原句因缺少材料依据已被删除，请依据台账自行补写】",
                    ledger_refs=[],
                )
            ]
        paragraph.sentences = kept

    return draft, report


def audit_refs(draft: EmailDraft, ledger: dict[str, LedgerRecord]) -> list[str]:
    """列出草稿中引用到的台账 ID，供"每句可核对"界面展示。"""
    refs: list[str] = []
    for paragraph in draft.paragraphs:
        for sentence in paragraph.sentences:
            for ref in sentence.ledger_refs:
                if ref in ledger and ref not in refs:
                    refs.append(ref)
    return refs


def sentence_origin(sentence: EmailSentence, ledger: dict[str, LedgerRecord]) -> list[LedgerRecord]:
    """把一句话映射回台账记录，供 UI 点击查看原文。"""
    return [ledger[ref] for ref in sentence.ledger_refs if ref in ledger]


__all__ = [
    "validate_draft",
    "audit_refs",
    "sentence_origin",
]
