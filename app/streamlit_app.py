"""Streamlit 界面骨架。

界面只做三件事：展示依据、让学生做决定、生成可粘贴的纯文本。
界面上不存在"发送"按钮——这不是遗漏，是设计。
"""

from __future__ import annotations

import streamlit as st

from advisor_match.config import settings

st.set_page_config(page_title="推研导师匹配与自荐邮件 Agent", layout="wide")

st.title("推研导师匹配与自荐邮件 Agent")
st.caption("依据公开可核验的信息匹配，依据你本人的材料写邮件；判断由你核对。")

with st.sidebar:
    st.subheader("流程")
    st.markdown(
        "1. 上传材料 → 生成成果台账\n"
        "2. 召回 30 位候选 → 保留 8 位\n"
        "3. 三档呈现（核心 / 相邻 / 需要补充）\n"
        "4. 选一位导师 → 生成四段式草稿\n"
        "5. 逐句确认 → 复制到你自己的邮箱"
    )
    st.divider()
    st.caption(f"召回上限：{settings.retrieval.recall_k}　保留：{settings.retrieval.keep_k}")
    st.caption("本应用不具备发信权限。")

tab_upload, tab_match, tab_email = st.tabs(["① 材料与台账", "② 导师匹配", "③ 邮件草稿"])

with tab_upload:
    st.file_uploader("简历 / 成绩单 / 论文 PDF", accept_multiple_files=True)
    st.info("上传后逐条核对台账：每条记录都应能点开看到原文出处。")
    st.data_editor([], num_rows="dynamic", use_container_width=True)

with tab_match:
    st.subheader("三档呈现")
    for tier, label in (("core", "核心契合"), ("adjacent", "相邻方向"), ("needs_support", "需要补充")):
        with st.expander(f"{label}", expanded=tier == "core"):
            st.write("（待接入匹配结果）每位导师展示：匹配点、差距、依据原文、信息截止日期")

with tab_email:
    st.subheader("四段式草稿")
    st.write("（待接入生成结果）每句右侧显示其依据的台账 ID，可点击回看原文。")
    st.text_area("确认后可复制到你自己的邮件客户端", height=300)
    st.checkbox("我已逐句核对，事实与材料一致")
    st.button("生成可粘贴文本（不会发送）", type="primary", disabled=True)
