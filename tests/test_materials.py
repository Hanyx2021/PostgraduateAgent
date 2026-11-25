"""材料文本解析测试。"""

from __future__ import annotations

import pytest

from advisor_match.product.materials import (
    decode_material,
    extract_docx_text,
    extract_pdf_text,
    extract_signature,
    redact_material,
    statements_from_text,
    upload_token,
)

ENGLISH_RESUME = """BASIC INFORMATION
Alex Chen   Phone: +86 138 0000 0004   Email: alex.chen@example.com
Research interests: computer vision, multimodal learning
EDUCATION
Example University, B.Eng. in Computer Science, 2022-2026
GPA 3.9/4.0, ranked top 5%
PROJECTS
Real-time object detection on edge devices: YOLO variant with TensorRT, 45 FPS on Jetson
Diffusion-based image editing: implemented classifier guidance and evaluation pipeline
RESEARCH
Second author, ICCV 2025 paper: Efficient Multimodal Fusion for Video Understanding
INTERNSHIP
Computer vision intern at a robotics startup (summer 2025)
SKILLS
Python, PyTorch, OpenCV, TensorRT, C++
AWARDS
National Scholarship (2024); ACM-ICPC Regional Silver Medal (2023)
"""


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


def test_parses_english_resume_and_drops_headers_and_education() -> None:
    statements, warnings = statements_from_text(ENGLISH_RESUME)
    joined = "\n".join(statements)

    assert "Real-time object detection" in joined
    assert "Second author, ICCV 2025" in joined
    assert "Computer vision intern" in joined
    assert "Python, PyTorch" in joined
    assert "National Scholarship" in joined

    assert "Alex Chen" not in joined
    assert "+86" not in joined
    assert "alex.chen@example.com" not in joined
    assert "BASIC INFORMATION" not in joined
    assert "EDUCATION" not in joined
    assert "PROJECTS" not in joined
    assert "GPA" not in joined
    assert "Example University" not in joined
    assert any("教育背景" in warning for warning in warnings)


def test_extracts_english_signature() -> None:
    signature = extract_signature(ENGLISH_RESUME)
    assert signature == "Alex Chen　Example University"


def test_redacts_english_name_and_contact() -> None:
    cleaned, warnings = redact_material(ENGLISH_RESUME)
    assert "Alex Chen" not in cleaned
    assert "+86" not in cleaned
    assert "alex.chen@example.com" not in cleaned
    assert "computer vision" in cleaned
    assert warnings


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


def test_pdf_block_lines_keep_english_sections_separate() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    lines = [
        "BASIC INFORMATION",
        "Alex Chen   Phone: +86 138 0000 0004   Email: alex.chen@example.com",
        "EDUCATION",
        "Example University, B.Eng. in Computer Science, 2022-2026",
        "PROJECTS",
        "Real-time object detection on edge devices",
    ]
    for index, line in enumerate(lines):
        page.insert_text((72, 72 + index * 24), line)
    data = document.tobytes()
    document.close()

    text, _ = extract_pdf_text(data)
    statements, _ = statements_from_text(text)
    assert statements == ["Real-time object detection on edge devices"]


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
    assert decode_material("中文材料".encode()) == "中文材料"
    assert decode_material("中文材料".encode("gb18030")) == "中文材料"
