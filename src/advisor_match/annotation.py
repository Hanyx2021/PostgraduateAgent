"""生成人工标注表：合成画像 × 候选导师。

每个画像取系统召回的候选池，再加少量随机导师（用于发现漏召），打乱顺序后
导出为 CSV；标注者只需填写 label（2/1/0）与 note。

同时输出一份 key（候选编号 → 导师姓名），供后续评分使用，不发给标注者。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from advisor_match.evaluation import EvalProfile, build_catalog_profiles, select_subset
from advisor_match.ingest.advisors import load_advisors
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.retrieval.recall import recall
from advisor_match.retrieval.vector import load_embedding
from advisor_match.schemas import Advisor

CSV_COLUMNS = [
    "profile_id",
    "目标方向",
    "学生材料摘要",
    "候选编号",
    "导师姓名",
    "单位",
    "职称",
    "研究标签",
    "招生目录方向",
    "近期论文（最多3篇）",
    "详情页",
    "label",
    "note",
]


@dataclass(frozen=True)
class AnnotationRow:
    profile_id: str
    direction: str
    student_summary: str
    candidate_no: str
    advisor: Advisor
    is_extra: bool = False

    def to_csv(self) -> dict[str, str]:
        tags = "、".join(self.advisor.research_tags[:8])
        directions = "、".join(
            dict.fromkeys(admission.direction for admission in self.advisor.admissions)
        )
        papers = "；".join(paper.title[:80] for paper in self.advisor.recent_papers[:3])
        return {
            "profile_id": self.profile_id,
            "目标方向": self.direction,
            "学生材料摘要": self.student_summary,
            "候选编号": self.candidate_no,
            "导师姓名": self.advisor.name,
            "单位": self.advisor.unit or "",
            "职称": self.advisor.title or "",
            "研究标签": tags,
            "招生目录方向": directions,
            "近期论文（最多3篇）": papers,
            "详情页": self.advisor.profile_url or "",
            "label": "",
            "note": "",
        }


def build_annotation_rows(
    profiles: list[EvalProfile],
    index: AdvisorIndex,
    *,
    pool_size: int = 12,
    extras: int = 3,
    seed: int = 2026,
) -> list[AnnotationRow]:
    rows: list[AnnotationRow] = []
    rng = random.Random(seed)

    for number, profile in enumerate(profiles, start=1):
        profile_id = f"P-{number:02d}"
        pool = recall(profile.profile, index, k=pool_size)
        pool_names = {advisor.name for advisor in pool}

        others = [advisor for advisor in index.advisors if advisor.name not in pool_names]
        extra_advisors = rng.sample(others, min(extras, len(others)))

        candidates = [(advisor, False) for advisor in pool] + [
            (advisor, True) for advisor in extra_advisors
        ]
        random.Random(f"{seed}-{profile_id}").shuffle(candidates)

        direction = " / ".join(profile.profile.target_directions)
        summary = profile.profile.ledger[0].statement if profile.profile.ledger else ""
        for index_no, (advisor, is_extra) in enumerate(candidates, start=1):
            rows.append(
                AnnotationRow(
                    profile_id=profile_id,
                    direction=direction,
                    student_summary=summary,
                    candidate_no=f"{profile_id}-C{index_no:02d}",
                    advisor=advisor,
                    is_extra=is_extra,
                )
            )
    return rows


def write_csv(rows: list[AnnotationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv())


def write_xlsx(rows: list[AnnotationRow], path: Path) -> None:
    """Excel 版本：label 列是下拉框，只需要选 0/1/2。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "标注"

    sheet.append(CSV_COLUMNS)
    for row in rows:
        item = row.to_csv()
        sheet.append([item[column] for column in CSV_COLUMNS])

    header_fill = PatternFill("solid", fgColor="DDEBF7")
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    widths = [10, 12, 28, 12, 10, 18, 10, 32, 42, 50, 40, 8, 24]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[chr(64 + index)].width = width
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    label_column = CSV_COLUMNS.index("label") + 1
    letter = chr(64 + label_column)
    validation = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True)
    sheet.add_data_validation(validation)
    validation.add(f"{letter}2:{letter}{sheet.max_row}")

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def write_key(rows: list[AnnotationRow], path: Path) -> None:
    key: dict[str, dict[str, object]] = {}
    for row in rows:
        key.setdefault(row.profile_id, {})[row.candidate_no] = {
            "advisor": row.advisor.name,
            "is_extra": row.is_extra,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")


INSTRUCTIONS = """# 标注说明

每行是一位候选导师，同一个 `profile_id` 代表同一个学生画像。

只需填写两列：

- `label`：
  - `2` 很契合，值得优先联系
  - `1` 方向相关，可以考虑
  - `0` 不相关
- `note`：一句话理由（可选）

判断依据：研究标签、招生目录方向、近期论文、单位/职称、详情页。
请**不要**参考系统排序（表里也没有），按你自己的判断填。

表格已打乱顺序，其中包含少量随机导师，用于检查系统是否漏召。
打开 `annotation_sheet.xlsx`，在 `label` 列下拉选择 0/1/2，`note` 列写一句话即可。
全部填完后把文件发回。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成人工标注表")
    parser.add_argument(
        "--advisors", type=Path, default=Path("data/processed/advisors.jsonl")
    )
    parser.add_argument("--profiles", type=int, default=10)
    parser.add_argument("--pool", type=int, default=12)
    parser.add_argument("--extras", type=int, default=3)
    parser.add_argument("--out-csv", type=Path, default=Path("data/processed/annotation_sheet.csv"))
    parser.add_argument("--out-key", type=Path, default=Path("data/processed/annotation_key.json"))
    parser.add_argument("--out-xlsx", type=Path, default=Path("data/processed/annotation_sheet.xlsx"))
    parser.add_argument("--out-instructions", type=Path, default=Path("data/processed/annotation_instructions.md"))
    args = parser.parse_args(argv)

    advisors = load_advisors(args.advisors)
    index = AdvisorIndex.build(advisors, embedding=load_embedding())
    profiles = select_subset(build_catalog_profiles(advisors, n=30), args.profiles)

    rows = build_annotation_rows(
        profiles, index, pool_size=args.pool, extras=args.extras
    )
    write_csv(rows, args.out_csv)
    write_xlsx(rows, args.out_xlsx)
    write_key(rows, args.out_key)
    args.out_instructions.parent.mkdir(parents=True, exist_ok=True)
    args.out_instructions.write_text(INSTRUCTIONS, encoding="utf-8")

    print(f"profiles: {len(profiles)}  rows: {len(rows)}")
    print(f"written: {args.out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
