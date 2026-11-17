"""研究生招生专业目录解析：把"具有招生资格"变成可核验字段。

为什么需要这一层：官方目录**不公布每位导师的名额**，只说明"所列导师均具有
博士生招生资格，与招生人数不完全对应"。因此系统能给出的最强表述是：

- 这位导师出现在某年某方向目录中（招生资格，有出处）；
- 目录里的明示限制，例如"不招收直博生"、"仅推免"。

系统**不能**推断"今年有几个名额"，也不能把往年被录取人数当概率。

页面结构（研招网）::

    <td style="padding-left: 40px" rowspan="N">01（全日制）方向名
        </br><span class="label">仅推免</span></td>
    ... 嵌套 <tr><td>导师名</td><td>招生说明</td></tr> ...

注意：源页面把 ``<tr>`` 直接嵌在 ``<td>`` 里，不是合法 HTML，因此这里用
按方向分块的正则解析，而不是构建 DOM 树。
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field

from advisor_match.schemas import AdmissionEntry

DIRECTION_PATTERN = re.compile(r"^(\d{2})（(全日制|非全日制)）(.+)$")
PROGRAM_PATTERN = re.compile(r"^(\d{6})\s*（(学术学位|专业学位)）\s*(.+)$")
_DIRECTION_BLOCK = re.compile(r'<td[^>]*rowspan="\d+"[^>]*>(.*?)</td>', re.S)
_PROGRAM_BLOCK = re.compile(r'<td[^>]*padding-left:\s*20px[^>]*>(.*?)</td>', re.S)
_ADVISOR_BLOCK = re.compile(
    r'<td[^>]*text-align:\s*center[^>]*>\s*(.*?)\s*</td>\s*<td[^>]*>(.*?)</td>', re.S
)
_NON_PERSON = {"导师组", "仅推免"}
_CJK_NAME = re.compile(r"^[一-龥·]{2,6}$")
_LATIN_NAME = re.compile(r"^[A-Za-z][A-Za-z .'\-]{2,40}$")


@dataclass(frozen=True)
class CatalogAdvisor:
    name: str
    note: str | None = None


@dataclass(frozen=True)
class CatalogDirection:
    program_code: str
    program_name: str
    program_type: str
    code: str
    mode: str
    name: str
    only_recommendation: bool = False
    advisors: list[CatalogAdvisor] = field(default_factory=list)

    @property
    def label(self) -> str:
        suffix = "（仅推免）" if self.only_recommendation else ""
        return f"{self.code} {self.name}{suffix}"

    @property
    def program_label(self) -> str:
        return f"{self.program_code} {self.program_name}（{self.program_type}）"


@dataclass(frozen=True)
class Catalog:
    year: int
    directions: list[CatalogDirection]

    def advisors(self) -> list[CatalogAdvisor]:
        """按出现顺序去重，保留第一次出现的说明。"""
        seen: dict[str, CatalogAdvisor] = {}
        for direction in self.directions:
            for advisor in direction.advisors:
                seen.setdefault(advisor.name, advisor)
        return list(seen.values())


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _is_person_name(candidate: str) -> bool:
    if not candidate or candidate in _NON_PERSON:
        return False
    if "@" in candidate or re.search(r"\d", candidate):
        return False
    return bool(_CJK_NAME.match(candidate) or _LATIN_NAME.match(candidate))


def parse_catalog(page: str, *, year: int) -> Catalog:
    """解析一页招生目录。只接受"专业 → 方向 → 导师 + 说明"这一层信息。"""
    blocks = list(_DIRECTION_BLOCK.finditer(page))
    programs = [
        (match.start(), _parse_program(_text(match.group(1))))
        for match in _PROGRAM_BLOCK.finditer(page)
    ]
    directions: list[CatalogDirection] = []

    for index, block in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(page)
        program = _latest_program(programs, block.start())
        if program is None:
            continue

        raw_direction = _text(block.group(1))
        only_recommendation = "仅推免" in raw_direction
        raw_direction = raw_direction.replace("仅推免", "").strip()

        matched = DIRECTION_PATTERN.match(raw_direction)
        if matched is None:
            continue
        code, mode, name = matched.groups()

        advisors: list[CatalogAdvisor] = []
        for advisor_block in _ADVISOR_BLOCK.finditer(page[block.end() : end]):
            candidate = _text(advisor_block.group(1))
            note = _text(advisor_block.group(2))
            # 说明栏偶尔放的是联系邮箱，不是招生限制，不能混进 notes。
            if "@" in note:
                note = ""
            if _is_person_name(candidate):
                advisors.append(CatalogAdvisor(name=candidate, note=note or None))

        directions.append(
            CatalogDirection(
                program_code=program[0],
                program_name=program[1],
                program_type=program[2],
                code=code,
                mode=mode,
                name=name.strip(),
                only_recommendation=only_recommendation,
                advisors=advisors,
            )
        )

    return Catalog(year=year, directions=directions)


def _parse_program(text: str) -> tuple[str, str, str] | None:
    matched = PROGRAM_PATTERN.match(text)
    if matched is None:
        return None
    code, program_type, name = matched.groups()
    return code, name.strip(), program_type


def _latest_program(
    programs: list[tuple[int, tuple[str, str, str] | None]], position: int
) -> tuple[str, str, str] | None:
    current = None
    for start, program in programs:
        if start > position:
            break
        if program is not None:
            current = program
    return current


def apply_catalog(
    advisors: list, catalog: Catalog, *, source_url: str
) -> tuple[list[str], list[str]]:
    """把目录信息写入 ``Advisor.admissions``，返回（未匹配到的导师名, 匹配到的导师名）。

    只写"在目录中"与"明示限制"，不写任何推测性名额。匹配用姓名精确相等；
    同名风险交给人工核对清单。同一年同一专业同一方向只保留一条记录。
    """
    by_name = {advisor.name: advisor for advisor in advisors}
    matched: list[str] = []
    unmatched: list[str] = []

    for direction in catalog.directions:
        for entry in direction.advisors:
            advisor = by_name.get(entry.name)
            if advisor is None:
                unmatched.append(entry.name)
                continue
            key = (catalog.year, direction.program_code, direction.code)
            existing = {
                (item.year, item.program_code, item.direction_code) for item in advisor.admissions
            }
            if key not in existing:
                advisor.admissions.append(
                    AdmissionEntry(
                        year=catalog.year,
                        program_code=direction.program_code,
                        program_name=direction.program_name,
                        program_type=direction.program_type,
                        direction_code=direction.code,
                        direction=direction.name,
                        only_recommendation=direction.only_recommendation,
                        notes=[entry.note] if entry.note else [],
                        source=source_url,
                    )
                )
            matched.append(entry.name)

    return unmatched, matched


__all__ = ["Catalog", "CatalogAdvisor", "CatalogDirection", "apply_catalog", "parse_catalog"]
