"""成果台账构建：从解析文本 + 模型抽取结果，生成可校验的 ``LedgerRecord``。

这层是"防编造"的实际执行点：
- 模型给出的每条记录必须能在原文里找到 ``quote``（做归一化后的包含匹配）；
- 找不到就丢弃，不写"疑似"记录；
- 台账 ID 由本地顺序分配，模型不能自己指定 ID。
"""

from __future__ import annotations

import re

from advisor_match.schemas import LedgerRecord, SourceRef

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """用于证据匹配的归一化：去空白、全半角统一、小写化。"""
    text = _WHITESPACE.sub("", text)
    return (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("：", ":")
        .replace("；", ";")
        .lower()
    )


def quote_is_grounded(quote: str, source_text: str) -> bool:
    """原文中是否真的包含这段引文。防止模型编造出处。"""
    if not quote.strip():
        return False
    return normalize(quote) in normalize(source_text)


def assign_ids(records: list[LedgerRecord], *, start: int = 1) -> list[LedgerRecord]:
    """按固定顺序重新编号，保证 ID 与内容一一对应、可复现。"""
    for offset, record in enumerate(records):
        record.id = f"L-{start + offset:04d}"
    return records


def build_ledger(
    extracted: list[dict[str, object]], sources: dict[str, str]
) -> tuple[list[LedgerRecord], list[str]]:
    """返回（台账, 被丢弃条目的原因）。

    ``extracted``：模型输出的原始 JSON 条目；``sources``：doc_id → 全文。
    """
    kept: list[LedgerRecord] = []
    rejected: list[str] = []
    for item in extracted:
        quote = str(item.get("quote", "") or "")
        doc_id = str(item.get("doc_id", "") or "")
        if doc_id not in sources or not quote_is_grounded(quote, sources[doc_id]):
            rejected.append(f"{doc_id or '<无doc_id>'}: 原文未找到引文「{quote[:40]}」")
            continue
        kept.append(
            LedgerRecord(
                id="L-0000",  # 占位，稍后统一编号
                kind=item["kind"],  # type: ignore[arg-type]
                statement=str(item["statement"]),
                status=item.get("status"),  # type: ignore[arg-type]
                evidence=[SourceRef(doc_id=doc_id, locator=str(item["locator"]), quote=quote)],
            )
        )
    return assign_ids(kept), rejected
