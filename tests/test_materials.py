"""材料文本解析测试。"""

from __future__ import annotations

import pytest

from advisor_match.product.materials import (
    decode_material,
    extract_docx_text,
    extract_pdf_text,
    extract_signature,
    statements_from_text,
    upload_token,
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


def test_skips_section_titles_and_person_names() -> None:
    text = "科研兴趣\n韩宇轩\n数据库\n机器学习\n完成检索项目"
    statements, _ = statements_from_text(text)
    assert statements == ["数据库", "机器学习", "完成检索项目"]


def test_upload_token_depends_on_content() -> None:
    first = upload_token("cv.pdf", 10, b"abc")
    assert upload_token("cv.pdf", 10, b"abc") == first
    assert upload_token("cv.pdf", 10, b"abd") != first


def test_extracts_signature_from_name_and_school() -> None:
    text = "韩宇轩\n清华大学 计算机科学与技术系\n完成检索项目"
    signature = extract_signature(text)
    assert signature is not None
    assert signature.startswith("韩宇轩")
    assert "清华大学" in signature


def test_extracts_signature_from_explicit_name_field() -> None:
    text = "姓名：李四\n某某大学 计算机学院"
    assert extract_signature(text) == "李四　某某大学 计算机学院"


def test_missing_signature_returns_none() -> None:
    assert extract_signature("完成课程项目\n熟悉 Python") is None


def test_removes_contact_information() -> None:
    text = "邮箱：user@example.com\n电话：13800138000\n完成项目，材料可通过 a@b.com 获取\n技能：Python"
    statements, warnings = statements_from_text(text)
    joined = "\n".join(statements)
    assert "user@example.com" not in joined
    assert "13800138000" not in joined
    assert "a@b.com" not in joined
    assert any("联系方式" in warning for warning in warnings)
    assert "技能：Python" in statements


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


def test_extract_docx_text_and_tables() -> None:
    import io

    from docx import Document

    document = Document()
    document.add_heading("基本信息", level=1)
    document.add_paragraph("完成检索课程项目")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "技能：Python"
    buffer = io.BytesIO()
    document.save(buffer)

    text, warnings = extract_docx_text(buffer.getvalue())
    assert "基本信息" in text
    assert "完成检索课程项目" in text
    assert "技能：Python" in text
    assert warnings == []


def test_corrupted_docx_returns_warning() -> None:
    text, warnings = extract_docx_text(b"not a docx")
    assert text == ""
    assert warnings


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
