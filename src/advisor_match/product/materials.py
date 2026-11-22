"""材料文本解析：把上传的 txt/md 转成台账候选陈述。

只做本地文本处理，不联网、不保存文件。每条陈述会成为一条台账记录，
因此长度限制与台账一致（不超过 300 字）；超长的行直接跳过并在界面提示，
而不是悄悄截断成一句不完整的话。
"""

from __future__ import annotations

import re

MAX_LINES = 50
MAX_CHARS = 300

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.、)]|\(\d+\))\s*")
_HEADING = re.compile(r"^\s*#+\s*")


def statements_from_text(
    text: str, *, max_lines: int = MAX_LINES, max_chars: int = MAX_CHARS
) -> tuple[list[str], list[str]]:
    """返回（陈述列表, 提示信息）。

    处理：去项目符号/序号、跳过空行与 Markdown 标题、去重、超长行跳过、总量截断。
    """
    warnings: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    statements: list[str] = []
    seen: set[str] = set()
    skipped_long = 0
    duplicates = 0

    for raw in lines:
        if _HEADING.match(raw):
            continue
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue
        if len(line) > max_chars:
            skipped_long += 1
            continue
        if line in seen:
            duplicates += 1
            continue
        seen.add(line)
        statements.append(line)

    if skipped_long:
        warnings.append(f"{skipped_long} 行超过 {max_chars} 字，已跳过")
    if duplicates:
        warnings.append(f"{duplicates} 行重复，已去重")
    if len(statements) > max_lines:
        warnings.append(f"超过 {max_lines} 条，只保留前 {max_lines} 条")
        statements = statements[:max_lines]

    return statements, warnings


def decode_material(raw: bytes) -> str:
    """优先 UTF-8，失败再按 GB18030（兼容中文 Windows 记事本）。"""
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


__all__ = ["decode_material", "statements_from_text"]
