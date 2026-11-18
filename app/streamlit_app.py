"""Streamlit 界面。

界面只做三件事：展示依据、让学生做决定、生成可粘贴的纯文本。
这里**没有发送按钮**——不是遗漏，是设计。演示数据全部为虚构。
"""

from __future__ import annotations

import streamlit as st

from advisor_match.config import settings
from advisor_match.email.draft import render_plain_text
from advisor_match.email.validator import sentence_origin
from advisor_match.pipeline import RecipientMismatch, run_email
from advisor_match.schemas import (
    Advisor,
    AdvisorContact,
    AdvisorMatch,
    Gap,
    LedgerRecord,
    MatchPoint,
    SourceRef,
)

st.set_page_config(page_title="推研导师匹配与自荐邮件 Agent", layout="wide")

st.title("推研导师匹配与自荐邮件 Agent")
st.caption("依据公开可核验的信息匹配，依据你本人的材料写邮件；判断由你核对。")

LEDGER = {
    "L-0001": LedgerRecord(
        id="L-0001",
        kind="paper",
        statement="以二作身份发表 ACL Findings 2025 论文《Multimodal Retrieval》",
        status="published",
        evidence=[SourceRef(doc_id="resume", locator="论文列表第 1 条", quote="ACL Findings 2025, 二作")],
    ),
    "L-0002": LedgerRecord(
        id="L-0002",
        kind="course_project",
        statement="完成多模态检索增强生成的课程项目",
        status="coursework",
        evidence=[SourceRef(doc_id="resume", locator="项目经历第 2 段", quote="多模态检索增强生成课程项目")],
    ),
    "L-0003": LedgerRecord(
        id="L-0003",
        kind="skill",
        statement="熟悉 PyTorch 与向量检索",
        evidence=[SourceRef(doc_id="resume", locator="技能栏", quote="PyTorch、向量检索")],
    ),
}

ADVISOR = Advisor(
    name="李老师",
    affiliation="某大学计算机系",
    unit="计算机软件研究所",
    title="教授",
    email="li@example.edu.cn",
    research_tags=["信息检索", "多模态学习"],
)
CONTACT = AdvisorContact(display_name="李老师", email="li@example.edu.cn")
MATCH = AdvisorMatch(
    advisor=ADVISOR,
    tier="core",
    match_points=[
        MatchPoint(
            dimension="direction",
            statement="课题组近年在多模态检索方向有持续工作",
            student_ledger_refs=["L-0001", "L-0002"],
            advisor_refs=[
                SourceRef(doc_id="advisor-profile", locator="研究领域", quote="信息检索，多模态学习")
            ],
        )
    ],
    gaps=[
        Gap(
            statement="学生目前缺少大规模实验与系统实现经验",
            student_ledger_refs=["L-0003"],
        )
    ],
)

with st.sidebar:
    st.subheader("流程")
    st.markdown(
        "1. 材料 → 成果台账\n"
        "2. 召回 30 位候选 → 保留 8 位\n"
        "3. 三档呈现（核心 / 相邻 / 需要补充）\n"
        "4. 生成四段式草稿\n"
        "5. 逐句确认 → 复制到你自己的邮箱"
    )
    st.divider()
    st.caption(f"召回上限：{settings.retrieval.recall_k}　保留：{settings.retrieval.keep_k}")
    st.caption("本应用不具备发信权限。演示数据均为虚构。")

tab_ledger, tab_match, tab_email = st.tabs(["① 材料与台账", "② 导师匹配", "③ 邮件草稿"])

with tab_ledger:
    st.subheader("成果台账")
    st.caption("每条记录都绑定原文出处；邮件里的每一句事实都会回到这里。")
    st.dataframe(
        [
            {
                "ID": record.id,
                "类型": record.kind,
                "陈述": record.statement,
                "状态": record.status or "—",
                "原文": f"{record.evidence[0].doc_id} · {record.evidence[0].locator}",
            }
            for record in LEDGER.values()
        ],
        width="stretch",
        hide_index=True,
    )

with tab_match:
    st.subheader("核心契合：李老师")
    st.caption("只展示理由与差距，不输出单一总分。")
    for point in MATCH.match_points:
        st.markdown(f"**契合点**（{point.dimension}）：{point.statement}")
        st.caption("依据：" + "、".join(point.student_ledger_refs))
    for gap in MATCH.gaps:
        st.markdown(f"**需要补充**：{gap.statement}")

with tab_email:
    st.subheader("四段式草稿")

    signature = st.text_input(
        "署名（只用于生成草稿，不保存）",
        placeholder="例如：张三　计算机系 2023 级",
    )
    points_text = st.text_area(
        "你自己的想法（每行一句，系统只改写能对上台账的句子）",
        placeholder="我做过多模态检索增强生成的课程项目",
        height=100,
    )
    generate = st.button("生成草稿", type="primary")

    if generate:
        try:
            result = run_email(
                MATCH,
                CONTACT,
                LEDGER,
                student_points=[line.strip() for line in points_text.splitlines() if line.strip()],
                student_signature=signature,
            )
        except RecipientMismatch as error:
            st.error(f"收件人核对未通过：{error}")
            st.session_state.pop("email_result", None)
        else:
            st.session_state["email_result"] = result
            st.session_state["confirmed"] = {}

    result = st.session_state.get("email_result")
    if result is not None:
        draft = result.draft
        st.markdown(f"**主题：** {draft.subject}")
        st.caption(
            f"校验：保留 {len(result.report.kept)} 句，丢弃无据陈述 {len(result.report.dropped)} 句"
        )

        confirmed: dict[str, bool] = st.session_state.setdefault("confirmed", {})
        for paragraph in draft.paragraphs:
            st.markdown(f"**{paragraph.role}**")
            for index, sentence in enumerate(paragraph.sentences):
                key = f"{paragraph.role}-{index}"
                col_text, col_check = st.columns([5, 1])
                with col_text:
                    st.write(sentence.text)
                    if sentence.ledger_refs:
                        origins = sentence_origin(sentence, LEDGER)
                        st.caption("依据：" + "；".join(f"{item.id} {item.statement}" for item in origins))
                    elif sentence.kind != "fact":
                        st.caption("非事实陈述（不绑定台账）")
                with col_check:
                    confirmed[key] = st.checkbox("已核对", key=key, label_visibility="collapsed")

        all_confirmed = bool(confirmed) and all(confirmed.values())
        st.divider()
        if all_confirmed:
            st.text_area(
                "复制到你自己的邮件客户端（本应用不会发送）",
                value=render_plain_text(draft, LEDGER),
                height=320,
            )
        else:
            st.info("请逐句核对后再生成可粘贴文本。")
            st.text_area("草稿预览", value=render_plain_text(draft, LEDGER), height=320, disabled=True)
