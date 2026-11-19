"""全系统的核心数据结构。

这里的每个约束都对应开题报告里的一个伦理承诺，改动前请先读对应测试：

1. ``StudentProfile`` 刻意不含姓名、性别、院校、籍贯等字段
   —— 守住"敏感属性不进入模型"（幻灯片 5）。``guards/sensitive.py`` 会检查。
2. ``LedgerRecord`` 必须至少携带一条 ``SourceRef``
   —— 守住"每条记录可回溯至原文"（幻灯片 6 第 1 步）。
3. ``AdvisorMatch`` 故意没有 score 字段
   —— 系统只给分档与理由，不输出单一总分（幻灯片 6 第 4 步）。
4. ``EmailDraft`` 没有 sent 状态
   —— 进程不具备发信权限，发送只能由学生手动完成（幻灯片 5 "不自动发送"）。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LedgerKind = Literal[
    "education",
    "course_project",
    "paper",
    "internship",
    "skill",
    "award",
    "other",
]

PaperStatus = Literal[
    "published",
    "accepted",
    "in_submission",
    "in_progress",
    "coursework",
]

Tier = Literal["core", "adjacent", "needs_support"]

Dimension = Literal["direction", "method", "experience", "resource"]


# --------------------------------------------------------------------------
# 可追溯性
# --------------------------------------------------------------------------
class SourceRef(BaseModel):
    """一条证据：指向某份材料中的某个位置。

    ``doc_id`` 是材料清单里的稳定 ID（如 ``resume-2026`` / ``paper-acl-findings``），
    ``locator`` 是文档内定位（如 ``p.2/第3段`` / ``abstract``），
    ``quote`` 是原文片段，用于学生逐字核对。
    """

    doc_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=800)


# --------------------------------------------------------------------------
# 学生侧：成果台账
# --------------------------------------------------------------------------
class LedgerRecord(BaseModel):
    """成果台账的一条记录。

    这是整个系统的主干：匹配理由和邮件句子都要绑定到 ``id`` 上。
    一条记录如果没有原文出处，就不应该存在。
    """

    id: str = Field(pattern=r"^L-\d{4}$")
    kind: LedgerKind
    statement: str = Field(min_length=1, max_length=300)
    status: PaperStatus | None = None
    evidence: list[SourceRef] = Field(min_length=1)
    notes: str | None = None

    @field_validator("evidence")
    @classmethod
    def _must_have_evidence(cls, value: list[SourceRef]) -> list[SourceRef]:
        if not value:
            raise ValueError("台账记录必须绑定至少一条原文证据")
        return value


class StudentProfile(BaseModel):
    """学生的非敏感画像。

    注意：这里没有 name / gender / hometown / school / birth 等字段，且不允许
    用 ``model_extra`` 的方式偷偷塞进来。任何想加敏感字段的改动都会让
    ``tests/test_fairness.py`` 失败——这是设计意图，不是阻碍。
    """

    ledger: list[LedgerRecord] = Field(min_length=1)
    interests: list[str] = Field(default_factory=list)
    target_directions: list[str] = Field(default_factory=list)
    #: 学生自己写下的、希望系统尊重的限制（如"只考虑北京""每周可投入 3 天"）。
    #: 只影响筛选，不参与排序得分。
    constraints: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------
# 导师侧
# --------------------------------------------------------------------------
class Paper(BaseModel):
    title: str
    venue: str | None = None
    year: int | None = None
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None


class AdmissionEntry(BaseModel):
    """一条招生记录：某年在某专业的某方向在列，以及该方向的明示限制。

    一条记录只描述一个方向，因为同一位导师可能在多个方向、多个专业中出现，
    且各方向限制不同（如某方向"不招收直博生"）。这**不表示名额**。
    """

    year: int
    program_code: str = Field(pattern=r"^\d{6}$")
    program_name: str = Field(min_length=1)
    program_type: Literal["学术学位", "专业学位"]
    direction_code: str = Field(pattern=r"^\d{2}$")
    direction: str = Field(min_length=1)
    #: 目录标注"仅推免"的方向（强基计划等）。
    only_recommendation: bool = False
    #: 目录中的明示限制，如"不招收直博生"。
    notes: list[str] = Field(default_factory=list)
    source: str = Field(min_length=1)


class Advisor(BaseModel):
    #: 目录中的姓名，作为导师身份的唯一天然键（系统中不再单设 ID）。
    name: str = Field(min_length=1)
    #: 名录、英文名等别名，仅用于匹配与论文消歧，不用于展示。
    aliases: list[str] = Field(default_factory=list)
    affiliation: str | None = None
    #: 研究所，如"计算机软件研究所"。
    unit: str | None = None
    #: 职称，如"教授""副教授""研究员"。招生口径与职称相关，需单独保留。
    title: str | None = None
    #: 系内公开工作邮箱。数据文件只存本地，不入库。
    email: str | None = None
    #: 系内详情页，作为 research_tags 的证据来源。
    profile_url: str | None = None
    homepage: str | None = None
    research_tags: list[str] = Field(default_factory=list)
    #: 详情页「研究概况」原文，重排时给模型更多上下文。
    research_summary: str | None = None
    recent_papers: list[Paper] = Field(default_factory=list)
    #: 招生目录中的在列记录。"有记录"只说明具有招生资格，不说明名额。
    admissions: list[AdmissionEntry] = Field(default_factory=list)
    #: 招生状态三态：True 明确招生 / False 明确不招 / None 未知。
    #: 未知必须显示为"未知"，不能靠猜测替学生做决定。
    recruiting: bool | None = None
    #: 信息抓取日期。界面必须显示"信息截止于 X 月 X 日"。
    fetched_at: date | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class AdvisorContact(BaseModel):
    """发信前需要人工核对的最小联系信息。"""

    display_name: str = Field(min_length=1)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------------------------------------------------------
# 匹配
# --------------------------------------------------------------------------
class MatchPoint(BaseModel):
    """一条"为什么契合"的证据化判断。"""

    dimension: Dimension
    statement: str = Field(min_length=1, max_length=300)
    student_ledger_refs: list[str] = Field(min_length=1)
    advisor_refs: list[SourceRef] = Field(min_length=1)


class Gap(BaseModel):
    """一条"哪里不契合 / 还需要补充什么"的判断。

    参照幻灯片 4 的"主动列出不匹配之处"：Gap 不是可选装饰，
    ``AdvisorMatch`` 在 core 档位也要求至少给出 1 条 gap（见 tests）。
    """

    statement: str = Field(min_length=1, max_length=300)
    student_ledger_refs: list[str] = Field(default_factory=list)
    advisor_refs: list[SourceRef] = Field(default_factory=list)


class AdvisorMatch(BaseModel):
    """一位导师的匹配结果：分档 + 理由。

    刻意没有 ``score`` 字段。如果需要排序，按
    (档位, 证据条数, 维度覆盖) 的确定性规则排，而不是造一个总分。
    """

    advisor: Advisor
    tier: Tier
    match_points: list[MatchPoint] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class MatchResult(BaseModel):
    """一次完整匹配的产出：三档候选 + 被舍弃时的理由。"""

    core: list[AdvisorMatch] = Field(default_factory=list)
    adjacent: list[AdvisorMatch] = Field(default_factory=list)
    needs_support: list[AdvisorMatch] = Field(default_factory=list)
    #: 系统主动加入的探索位（不按证据数排），界面需要明确标注。
    exploration: list[str] = Field(default_factory=list)
    #: 召回但未进入三档的候选及剔除原因，便于学生追问"为什么没有他"。
    excluded: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 邮件
# --------------------------------------------------------------------------
class EmailSentence(BaseModel):
    text: str = Field(min_length=1)
    #: 句子类型。事实句必须绑定台账；意向句（"我希望……"）与礼节句没有事实主张。
    kind: Literal["fact", "intent", "greeting"] = "fact"
    #: 该句依赖的台账记录 ID。事实句必须非空且能查到，否则整段丢弃。
    ledger_refs: list[str] = Field(default_factory=list)


class EmailParagraph(BaseModel):
    """四段式中的一段。"""

    role: Literal["greeting", "fit", "gap", "closing"]
    sentences: list[EmailSentence] = Field(min_length=1)


class EmailDraft(BaseModel):
    """待学生逐句确认的草稿。没有 sent 状态，也不应该有。"""

    recipient: AdvisorContact
    subject: str = Field(min_length=1)
    paragraphs: list[EmailParagraph] = Field(min_length=1)
    status: Literal["draft", "student_confirmed"] = "draft"
    created_at: datetime = Field(default_factory=datetime.now)


class ValidationReport(BaseModel):
    """校验器报告：查不到证据的句子被整段丢弃，而不是标记后保留。"""

    kept: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)

    @property
    def unsupported_ratio(self) -> float:
        total = len(self.kept) + len(self.dropped)
        return 0.0 if total == 0 else len(self.dropped) / total
