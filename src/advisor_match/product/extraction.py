"""用模型把原始材料整理成成果台账。

流程：脱敏 → 包成不可信数据块 → 模型输出 JSON → 逐条校验引文 → 编号。
任何一条引文在原文里找不到就丢弃；模型整体失败时退回按行解析。

隐私：姓名行与联系方式在发送前移除；本模块不落盘任何 prompt。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from advisor_match.guards.injection import wrap_untrusted
from advisor_match.ledger.build import quote_is_grounded
from advisor_match.llm import prompts
from advisor_match.llm.client import LLMClient
from advisor_match.product.materials import redact_material, statements_from_text
from advisor_match.schemas import LedgerRecord, SourceRef

KINDS = {"education", "course_project", "paper", "internship", "skill", "award", "other"}
STATUSES = {"published", "accepted", "in_submission", "in_progress", "coursework"}
MAX_RECORDS = 15


@dataclass
class ExtractionResult:
    records: list[LedgerRecord] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出中没有 JSON 对象")
    return json.loads(raw[start : end + 1])


def extract_records(
    text: str,
    llm: LLMClient,
    *,
    doc_id: str = "upload",
    max_records: int = MAX_RECORDS,
) -> ExtractionResult:
    cleaned, warnings = redact_material(text)
    warnings = list(warnings)

    user = prompts.EXTRACTION_USER.format(
        doc_id=doc_id,
        content=wrap_untrusted(cleaned, source=doc_id),
    )
    data = None
    for _ in range(2):
        try:
            parsed_data = _extract_json(llm.complete(system=prompts.EXTRACTION_SYSTEM, user=user))
            if not isinstance(parsed_data, dict):
                raise ValueError("模型输出的 JSON 不是对象")
            data = parsed_data
            break
        except Exception:  # noqa: BLE001 - 模型不可用/输出异常都进入重试
            continue

    records: list[LedgerRecord] = []
    directions: list[str] = []
    rejected = 0
    if data is not None:
        raw_directions = data.get("directions", [])
        if isinstance(raw_directions, list):
            directions = [str(item).strip() for item in raw_directions if str(item).strip()][:4]
        raw_records = data.get("records", [])
        if not isinstance(raw_records, list):
            warnings.append("模型返回的 records 不是列表，已忽略")
            raw_records = []
        seen: set[str] = set()
        for item in raw_records[:max_records]:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote", "")).strip()
            statement = str(item.get("statement", "")).strip()
            if not quote or not statement or len(statement) > 300:
                rejected += 1
                continue
            if not quote_is_grounded(quote, cleaned):
                rejected += 1
                continue
            key = re.sub(r"\s+", "", statement)
            if key in seen:
                continue
            seen.add(key)

            kind = item.get("kind")
            if kind not in KINDS:
                kind = "other"
            status = item.get("status")
            if kind != "paper" or status not in STATUSES:
                status = None

            records.append(
                LedgerRecord(
                    id=f"L-{len(records) + 1:04d}",
                    kind=kind,
                    statement=statement,
                    status=status,
                    evidence=[
                        SourceRef(
                            doc_id=doc_id,
                            locator=str(item.get("locator") or "模型抽取"),
                            quote=quote[:800],
                        )
                    ],
                )
            )
        if rejected:
            warnings.append(f"{rejected} 条引文在原文中找不到，已丢弃")

    if not records:
        statements, line_warnings = statements_from_text(cleaned)
        warnings.extend(line_warnings)
        records = [
            LedgerRecord(
                id=f"L-{index:04d}",
                kind="other",
                statement=statement,
                evidence=[SourceRef(doc_id=doc_id, locator="按行解析", quote=statement)],
            )
            for index, statement in enumerate(statements[:max_records], start=1)
        ]
        warnings.append("模型未给出可用条目，已退回按行解析")

    return ExtractionResult(records=records, directions=directions, warnings=warnings)


__all__ = ["ExtractionResult", "extract_records"]
