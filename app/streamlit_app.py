"""Streamlit 界面：真实导师数据 → 匹配 → 草稿 → 逐句确认。

界面只做三件事：展示依据、让学生做决定、生成可粘贴的纯文本。
这里**没有发送按钮**——不是遗漏，是设计。

数据文件默认读 ``data/processed/advisors.jsonl``（本地生成、不入库）；
找不到时退回两位虚构导师，仅用于演示界面，并明确标注。
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from advisor_match.config import settings
from advisor_match.email.confirm import (
    build_record,
    load_confirmations,
    mark_confirmed,
    save_confirmation,
)
from advisor_match.email.draft import render_plain_text
from advisor_match.email.validator import audit_refs, sentence_origin, validate_draft
from advisor_match.ingest.advisors import load_advisors
from advisor_match.pipeline import EmailResult, RecipientMismatch, run_email, run_matching
from advisor_match.retrieval.index import AdvisorIndex
from advisor_match.retrieval.vector import load_embedding
from advisor_match.schemas import (
    Advisor,
    AdvisorContact,
    LedgerRecord,
    SourceRef,
    StudentProfile,
)

st.set_page_config(page_title="推研导师匹配与自荐邮件 Agent", layout="wide")

DEFAULT_DATA = os.getenv("AM_ADVISORS_PATH", "data/processed/advisors.jsonl")
CONFIRMATION_LOG = Path(
    os.getenv("AM_CONFIRMATIONS_PATH", "data/processed/confirmations.jsonl")
)


def render_advisor_meta(advisor) -> None:
    """导师信息必须带来源与截止日期；招生状态只写"具资格"，不写名额。"""
    st.caption(
        f"{advisor.unit or '单位未收录'}　{advisor.title or ''}　"
        f"{advisor.email or '邮箱未收录'}"
    )
    if advisor.fetched_at:
        st.caption(f"信息截止：{advisor.fetched_at.isoformat()}")
    if advisor.admissions:
        st.caption("招生资格：2027 招生目录在列，名额未公布")
        for admission in advisor.admissions[:3]:
            detail = "仅推免" if admission.only_recommendation else ""
            notes = "；".join(admission.notes)
            suffix = "；".join(item for item in (detail, notes) if item)
            st.caption(
                f"· {admission.program_name} / {admission.direction}"
                + (f"（{suffix}）" if suffix else "")
            )
    if advisor.profile_url:
        st.markdown(f"[系内详情页]({advisor.profile_url})")


def demo_advisors() -> list[Advisor]:
    """没有导师数据时的演示对象，全部虚构。"""
    return [
        Advisor(
            name="李老师",
            affiliation="某大学计算机系",
            unit="计算机软件研究所",
            title="教授",
            email="li@example.edu.cn",
            research_tags=["信息检索", "多模态学习"],
            research_summary="研究信息检索与多模态学习。",
            sources=[SourceRef(doc_id="demo-profile", locator="研究领域", quote="信息检索，多模态学习")],
        ),
        Advisor(
            name="王老师",
            affiliation="某大学计算机系",
            unit="计算机软件研究所",
            title="副教授",
            email="wang@example.edu.cn",
            research_tags=["数据库", "数据管理"],
            research_summary="研究数据库系统与数据管理。",
            sources=[SourceRef(doc_id="demo-profile", locator="研究领域", quote="数据库，数据管理")],
        ),
    ]


@st.cache_resource(show_spinner=False)
def load_index(path: str, mtime: float, use_vector: bool) -> tuple[AdvisorIndex, bool]:
    advisors = load_advisors(Path(path))
    embedding = load_embedding() if use_vector else None
    return AdvisorIndex.build(advisors, embedding=embedding), bool(embedding)


@st.cache_data(show_spinner=False)
def read_mtime(path: str) -> float:
    return Path(path).stat().st_mtime


def build_profile(
    targets: list[str], interests: list[str], statements: list[str]
) -> tuple[StudentProfile, dict[str, LedgerRecord]]:
    ledger: dict[str, LedgerRecord] = {}
    for index, statement in enumerate(statements, start=1):
        record_id = f"L-{index:04d}"
        ledger[record_id] = LedgerRecord(
            id=record_id,
            kind="other",
            statement=statement,
            evidence=[SourceRef(doc_id="student-input", locator="手动填写", quote=statement)],
        )
    profile = StudentProfile(
        ledger=list(ledger.values()),
        target_directions=targets,
        interests=interests,
    )
    return profile, ledger


st.title("推研导师匹配与自荐邮件 Agent")
st.caption("依据公开可核验的信息匹配，依据你本人的材料写邮件；判断由你核对。")

with st.sidebar:
    st.subheader("数据")
    data_path = st.text_input("导师主表路径", value=DEFAULT_DATA, key="data_path")
    use_vector = st.checkbox("使用向量召回（需要本地模型）", value=True, key="use_vector")

    st.subheader("你的材料")
    target_text = st.text_input("目标方向（逗号分隔）", placeholder="信息检索, 多模态学习", key="targets")
    interest_text = st.text_input("研究兴趣（可选，逗号分隔）", placeholder="大模型, 检索增强", key="interests")
    statement_text = st.text_area(
        "成果陈述（每行一条，会成为台账记录）",
        placeholder="完成多模态检索增强生成的课程项目\n以二作身份发表 ACL Findings 2025 论文",
        height=120,
        key="statements",
    )
    start = st.button("开始匹配", type="primary", key="run_matching")

    st.divider()
    st.caption(f"召回上限 {settings.retrieval.recall_k}，主榜 {settings.retrieval.keep_k}")
    st.caption("本应用不具备发信权限。")

if start:
    statements = [line.strip() for line in statement_text.splitlines() if line.strip()]
    targets = [item.strip() for item in target_text.split(",") if item.strip()]
    interests = [item.strip() for item in interest_text.split(",") if item.strip()]

    if not statements:
        st.error("至少填写一条成果陈述，否则匹配没有可核验材料。")
    else:
        path = Path(data_path)
        if path.exists():
            index, vector_on = load_index(data_path, read_mtime(data_path), use_vector)
        else:
            demo = demo_advisors()
            embedding = load_embedding() if use_vector else None
            index = AdvisorIndex.build(demo, embedding=embedding)
            vector_on = bool(embedding)
            st.warning("未找到导师数据文件，当前使用两位虚构导师演示。")

        profile, ledger = build_profile(targets, interests, statements)
        result, logs = run_matching(profile, index)

        st.session_state.update(
            {
                "result": result,
                "ledger": ledger,
                "profile": profile,
                "index_size": len(index.advisors),
                "index_skipped": len(index.skipped),
                "vector_on": vector_on,
                "logs": logs,
                "data_source": "真实数据" if path.exists() else "演示数据",
            }
        )
        st.session_state.pop("draft", None)
        st.session_state["confirmed"] = {}

result = st.session_state.get("result")
ledger = st.session_state.get("ledger") or {}

tab_ledger, tab_match, tab_email = st.tabs(["① 材料与台账", "② 导师匹配", "③ 邮件草稿"])

with tab_ledger:
    st.subheader("成果台账")
    if ledger:
        st.caption(f"数据来源：{st.session_state.get('data_source', '—')}")
        st.dataframe(
            [
                {
                    "ID": record.id,
                    "陈述": record.statement,
                    "原文": f"{record.evidence[0].doc_id} · {record.evidence[0].locator}",
                }
                for record in ledger.values()
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("在左侧填写材料并点击「开始匹配」。")

with tab_match:
    if result is None:
        st.info("尚未开始匹配。")
    else:
        st.caption(
            f"索引导师 {st.session_state.get('index_size')} 位，"
            f"跳过无研究方向 {st.session_state.get('index_skipped')} 位，"
            f"向量召回 {'开' if st.session_state.get('vector_on') else '关'}"
        )
        for log in st.session_state.get("logs", []):
            st.caption(f"[{log.stage}] {log.input_count} → {log.output_count}")

        for tier, label in (
            ("core", "核心契合"),
            ("adjacent", "相邻方向"),
            ("needs_support", "需要补充"),
        ):
            matches = getattr(result, tier)
            if not matches:
                continue
            st.markdown(f"### {label}（{len(matches)} 位）")
            for match in matches:
                flag = "　`探索位`" if match.advisor.name in result.exploration else ""
                with st.expander(f"{match.advisor.name} · {len(match.match_points)} 条依据{flag}"):
                    advisor = match.advisor
                    render_advisor_meta(advisor)
                    for point in match.match_points:
                        st.write(f"· {point.statement}")
                        st.caption("依据：" + "、".join(point.student_ledger_refs))
                    for gap in match.gaps:
                        st.write(f"· 需要补充：{gap.statement}")

        if result.excluded:
            with st.expander(f"未列入（{len(result.excluded)} 位）"):
                for name, reason in result.excluded.items():
                    st.caption(f"{name}：{reason}")

with tab_email:
    if result is None:
        st.info("先完成匹配，再选择一位导师生成草稿。")
    else:
        selected = [*result.core, *result.adjacent, *result.needs_support]
        names = [match.advisor.name for match in selected]
        if not names:
            st.warning("当前没有可写的导师。")
        else:
            name = st.selectbox("选择导师", names, key="advisor_select")
            match = selected[names.index(name)]
            advisor = match.advisor

            signature = st.text_input(
                "署名（只用于生成草稿，不保存）", placeholder="张三　计算机系 2023 级", key="signature"
            )
            points_text = st.text_area(
                "你自己的想法（每行一句，只保留能对上台账的句子）", height=90, key="points"
            )
            generate = st.button("生成草稿", key="generate_draft")

            if generate:
                if not advisor.email:
                    st.error("该导师邮箱未收录，无法确定收件人。")
                else:
                    contact = AdvisorContact(display_name=advisor.name, email=advisor.email)
                    try:
                        email_result = run_email(
                            match,
                            contact,
                            ledger,
                            student_points=[
                                line.strip() for line in points_text.splitlines() if line.strip()
                            ],
                            student_signature=signature,
                        )
                    except RecipientMismatch as error:
                        st.error(f"收件人核对未通过：{error}")
                    else:
                        st.session_state["draft"] = email_result
                        st.session_state["contact"] = contact
                        st.session_state["confirmed"] = {}
                        st.session_state["revision"] = 0
                        st.session_state["edited_total"] = 0
                        st.session_state.pop("confirmation", None)

            email_result = st.session_state.get("draft")
            if email_result is not None:
                draft = email_result.draft
                st.markdown(f"**主题：** {draft.subject}")
                st.caption(
                    f"校验：保留 {len(email_result.report.kept)} 句，"
                    f"丢弃无据陈述 {len(email_result.report.dropped)} 句"
                )

                revision = st.session_state.setdefault("revision", 0)
                edited_total = st.session_state.setdefault("edited_total", 0)
                confirmed: dict[str, bool] = st.session_state.setdefault("confirmed", {})

                for paragraph in draft.paragraphs:
                    st.markdown(f"**{paragraph.role}**")
                    for index, sentence in enumerate(paragraph.sentences):
                        sid = f"{revision}-{paragraph.role}-{index}"
                        col_text, col_ops = st.columns([5, 1])
                        with col_text:
                            st.text_area(
                                "句子",
                                value=sentence.text,
                                key=f"edit-{sid}",
                                height=70,
                                label_visibility="collapsed",
                            )
                            if sentence.ledger_refs:
                                origins = sentence_origin(sentence, ledger)
                                st.caption(
                                    "依据："
                                    + "；".join(f"{item.id} {item.statement}" for item in origins)
                                )
                            elif sentence.kind != "fact":
                                st.caption("非事实陈述（不绑定台账）")
                        with col_ops:
                            confirmed[sid] = st.checkbox(
                                "已核对", key=f"check-{sid}", label_visibility="collapsed"
                            )
                            st.checkbox("删除", key=f"del-{sid}")

                if st.button("应用修改并重新校验", key="apply_edits"):
                    new_paragraphs = []
                    edited = 0
                    for paragraph in draft.paragraphs:
                        kept = []
                        for index, sentence in enumerate(paragraph.sentences):
                            sid = f"{revision}-{paragraph.role}-{index}"
                            text = str(st.session_state.get(f"edit-{sid}", sentence.text)).strip()
                            if st.session_state.get(f"del-{sid}") or not text:
                                continue
                            if text != sentence.text:
                                edited += 1
                            kept.append(sentence.model_copy(update={"text": text}))
                        if kept:
                            new_paragraphs.append(paragraph.model_copy(update={"sentences": kept}))

                    if not new_paragraphs:
                        st.error("至少要保留一句。")
                    else:
                        updated = draft.model_copy(update={"paragraphs": new_paragraphs})
                        updated, report = validate_draft(updated, ledger)
                        st.session_state["draft"] = EmailResult(draft=updated, report=report)
                        st.session_state["edited_total"] = edited_total + edited
                        st.session_state["revision"] = revision + 1
                        st.session_state["confirmed"] = {}
                        st.session_state.pop("confirmation", None)
                        st.rerun()

                st.divider()
                all_confirmed = bool(confirmed) and all(confirmed.values())
                if all_confirmed and st.button("完成确认并生成文本", key="confirm_draft"):
                    mark_confirmed(draft, all_sentences_confirmed=True)
                    record = build_record(
                        draft,
                        st.session_state["contact"],
                        refs_used=audit_refs(draft, ledger),
                        edited=st.session_state.get("edited_total", 0),
                    )
                    save_confirmation(record, CONFIRMATION_LOG)
                    st.session_state["confirmation"] = record

                saved = st.session_state.get("confirmation")
                if saved is not None and saved.advisor_name == draft.recipient.display_name:
                    st.success(
                        f"已留痕：{saved.advisor_name}，{saved.sentence_count} 句，"
                        f"{len(saved.refs_used)} 条台账依据（只记行为，不记正文）"
                    )
                    st.text_area(
                        "复制到你自己的邮件客户端（本应用不会发送）",
                        value=render_plain_text(draft, ledger),
                        height=320,
                    )
                elif all_confirmed:
                    st.info("逐句核对完成。点击「完成确认」生成留痕与可粘贴文本。")
                    st.text_area(
                        "草稿预览",
                        value=render_plain_text(draft, ledger),
                        height=320,
                        disabled=True,
                    )
                else:
                    st.info("请逐句核对后再生成可粘贴文本。")
                    st.text_area(
                        "草稿预览",
                        value=render_plain_text(draft, ledger),
                        height=320,
                        disabled=True,
                    )

            with st.expander("最近确认记录（只记行为，不记正文）"):
                records = load_confirmations(CONFIRMATION_LOG, limit=5)
                if not records:
                    st.caption("暂无记录。")
                for item in reversed(records):
                    st.caption(
                        f"{item['confirmed_at']} · {item['advisor_name']} · "
                        f"{item['sentence_count']} 句 · {len(item['refs_used'])} 条依据"
                    )
