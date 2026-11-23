"""模型整理台账的测试：证据校验、脱敏、兜底。"""

from __future__ import annotations

import json

from advisor_match.product.extraction import extract_records

RESUME = """韩宇轩
邮箱：han@example.com
电话：13800138000
教育背景
完成多模态检索增强生成的课程项目
以二作身份发表 ACL Findings 2025 论文
"""


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.user = ""

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.user = user
        return self.response


def response(records: list[dict], *, directions: list[str] | None = None) -> str:
    return json.dumps(
        {"directions": directions or [], "records": records}, ensure_ascii=False
    )


def test_keeps_grounded_records_and_drops_fabricated_quote() -> None:
    llm = FakeLLM(
        response(
            [
                {
                    "kind": "course_project",
                    "statement": "完成多模态检索增强生成的课程项目",
                    "status": None,
                    "locator": "第5行",
                    "quote": "完成多模态检索增强生成的课程项目",
                },
                {
                    "kind": "paper",
                    "statement": "以二作身份发表 ACL Findings 2025 论文",
                    "status": "published",
                    "locator": "第6行",
                    "quote": "以二作身份发表 ACL Findings 2025 论文",
                },
                {
                    "kind": "skill",
                    "statement": "编造的技能",
                    "status": None,
                    "locator": "无",
                    "quote": "原文中不存在这句话",
                },
            ]
        )
    )
    result = extract_records(RESUME, llm)

    assert [record.id for record in result.records] == ["L-0001", "L-0002"]
    assert result.records[1].status == "published"
    assert all(record.evidence for record in result.records)
    assert any("引文在原文中找不到" in warning for warning in result.warnings)


def test_returns_suggested_directions() -> None:
    llm = FakeLLM(
        response(
            [
                {
                    "kind": "course_project",
                    "statement": "完成多模态检索增强生成的课程项目",
                    "status": None,
                    "locator": "第5行",
                    "quote": "完成多模态检索增强生成的课程项目",
                }
            ],
            directions=["信息检索", "多模态学习"],
        )
    )
    result = extract_records(RESUME, llm)
    assert result.directions == ["信息检索", "多模态学习"]


def test_redacts_name_and_contact_before_sending() -> None:
    llm = FakeLLM(response([]))
    extract_records(RESUME, llm)

    assert "han@example.com" not in llm.user
    assert "13800138000" not in llm.user
    assert "韩宇轩" not in llm.user


def test_invalid_llm_output_falls_back_to_line_parser() -> None:
    llm = FakeLLM("这不是 JSON")
    result = extract_records(RESUME, llm)

    statements = [record.statement for record in result.records]
    assert "完成多模态检索增强生成的课程项目" in statements
    assert all(record.kind == "other" for record in result.records)
    assert any("退回按行解析" in warning for warning in result.warnings)


def test_duplicate_statements_are_removed() -> None:
    item = {
        "kind": "other",
        "statement": "完成课程项目",
        "status": None,
        "locator": "第5行",
        "quote": "完成多模态检索增强生成的课程项目",
    }
    result = extract_records(RESUME, FakeLLM(response([item, item])))
    assert len(result.records) == 1
