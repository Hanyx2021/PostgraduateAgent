"""确认留痕测试：只记行为，不记正文。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from advisor_match.email.confirm import (
    ConfirmationRecord,
    load_confirmations,
    save_confirmation,
)


def test_save_and_load_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "confirmations.jsonl"
    record = ConfirmationRecord(
        advisor_name="李老师",
        confirmed_at=datetime(2026, 11, 21, 10, 0),
        sentence_count=7,
        refs_used=["L-0001", "L-0002"],
        student_edit_ratio=0.0,
    )
    save_confirmation(record, path)
    save_confirmation(record, path)

    records = load_confirmations(path)
    assert len(records) == 2
    assert records[0]["advisor_name"] == "李老师"
    assert records[0]["sentence_count"] == 7
    assert records[0]["refs_used"] == ["L-0001", "L-0002"]
    # 留痕不含邮件正文
    assert all("text" not in item and "subject" not in item for item in records)


def test_load_missing_log_returns_empty(tmp_path: Path) -> None:
    assert load_confirmations(tmp_path / "missing.jsonl") == []
