"""台账的本地存储（jsonl，一条一行）。"""

from __future__ import annotations

from pathlib import Path

from advisor_match.schemas import LedgerRecord, StudentProfile


def save_ledger(records: list[LedgerRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def load_ledger(path: Path) -> dict[str, LedgerRecord]:
    """返回 id → 记录 的字典，供校验器 O(1) 查证。"""
    ledger: dict[str, LedgerRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = LedgerRecord.model_validate_json(line)
            ledger[record.id] = record
    return ledger


def load_profile(ledger_path: Path, *, interests: list[str] | None = None) -> StudentProfile:
    return StudentProfile(
        ledger=list(load_ledger(ledger_path).values()),
        interests=interests or [],
    )
