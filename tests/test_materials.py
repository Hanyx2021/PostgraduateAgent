"""材料文本解析测试。"""

from __future__ import annotations

import pytest

from advisor_match.product.materials import (
    decode_material,
    extract_pdf_text,
    statements_from_text,
)


def test_strips_bullets_and_numbers() -> None:
    text = "1. 完成信息检索课程项目\n- 发表 ACL Findings 论文\n• 熟悉 PyTorch\n(3) 获得奖学金"
    statements, warnings = statements_from_text(text)
    assert statements == [
        "完成信息检索课程项目",
        "发表 ACL Findings 论文",
        "熟悉 PyTorch",
        "获得奖学金",
    ]
    assert warnings == []


def test_skips_headings_blank_lines_and_duplicates() -> None:
    text = "# 教育背景\n\n完成项目 A\n完成项目 A\n## 项目\n完成项目 B\n"
    statements, warnings = statements_from_text(text)
    assert statements == ["完成项目 A", "完成项目 B"]
    assert any("重复" in warning for warning in warnings)
    assert "教育背景" not in statements
    assert "项目" not in statements


def test_skips_overlong_lines() -> None:
    text = "短句\n" + "长" * 301
    statements, warnings = statements_from_text(text)
    assert statements == ["短句"]
    assert any("超过 300 字" in warning for warning in warnings)


def test_caps_line_count() -> None:
    text = "\n".join(f"第 {index} 条" for index in range(60))
    statements, warnings = statements_from_text(text, max_lines=50)
    assert len(statements) == 50
    assert any("只保留前 50 条" in warning for warning in warnings)


def test_extract_pdf_text_and_statements() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Completed a multimodal retrieval course project")
    page.insert_text((72, 100), "Published a paper at ACL Findings 2025")
    data = document.tobytes()
    document.close()

    text, warnings = extract_pdf_text(data)
    assert "multimodal retrieval" in text
    statements, _ = statements_from_text(text)
    assert any("multimodal retrieval" in statement for statement in statements)
    assert warnings == []


def test_blank_pdf_warns_about_scan() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    document.new_page()
    data = document.tobytes()
    document.close()

    text, warnings = extract_pdf_text(data)
    assert text == ""
    assert any("扫描版" in warning for warning in warnings)


def test_decode_prefers_utf8_then_gb18030() -> None:
    assert decode_material("中文材料".encode("utf-8")) == "中文材料"
    assert decode_material("中文材料".encode("gb18030")) == "中文材料"
