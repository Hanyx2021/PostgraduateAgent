"""简历 / 成绩单解析。

这一层的输出是"原始文本 + 定位"，不做任何判断。
成绩单只抽取与研究方向相关的课程与项目，不抽取分数段排名（避免进入排序）。
"""

from __future__ import annotations

from pathlib import Path

from advisor_match.ingest.papers import ParsedDocument


def parse_resume(path: Path, *, doc_id: str = "resume") -> ParsedDocument:
    """TODO: 支持 PDF / docx。docx 用 python-docx，PDF 复用 ``parse_pdf``。"""
    raise NotImplementedError("由韩宇轩实现")


def parse_transcript(path: Path, *, doc_id: str = "transcript") -> ParsedDocument:
    """TODO: 成绩单解析。只保留课程名与相关项目，分数不进入台账陈述。"""
    raise NotImplementedError("由韩宇轩实现")
