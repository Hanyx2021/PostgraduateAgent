"""人工标注表生成测试。"""

from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from advisor_match.annotation import build_annotation_rows, write_xlsx
from advisor_match.evaluation import build_catalog_profiles
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.schemas import AdmissionEntry, Advisor, SourceRef


def make_advisor(name: str, direction: str) -> Advisor:
    return Advisor(
        name=name,
        research_tags=[direction],
        sources=[SourceRef(doc_id=f"p-{name}", locator="研究领域", quote=direction)],
        admissions=[
            AdmissionEntry(
                year=2027,
                program_code="081200",
                program_name="计算机科学与技术",
                program_type="学术学位",
                direction_code="01",
                direction=direction,
                source="https://example.edu.cn/catalog",
            )
        ],
    )


def test_rows_are_blind_and_labels_are_empty(tmp_path: Path) -> None:
    advisors = [make_advisor(name, "信息检索") for name in ["甲", "乙", "丙", "丁"]]
    index = AdvisorIndex.build(advisors)
    profiles = build_catalog_profiles(advisors, n=1, min_relevant=1)

    rows = build_annotation_rows(profiles, index, pool_size=3, extras=1)
    assert len(rows) == 4
    assert len({row.candidate_no for row in rows}) == 4
    assert all(row.to_csv()["label"] == "" for row in rows)
    assert sum(1 for row in rows if row.is_extra) == 1

    path = tmp_path / "sheet.xlsx"
    write_xlsx(rows, path)
    sheet = load_workbook(path).active
    assert sheet.max_row == 5
    assert sheet["L2"].value is None
    assert sheet.data_validations.dataValidation
