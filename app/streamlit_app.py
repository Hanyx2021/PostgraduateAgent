"""Streamlit 界面：真实导师数据 → 匹配 → 草稿 → 逐句确认。

界面只做三件事：展示依据、让学生做决定、生成可粘贴的纯文本。
这里**没有发送按钮**——不是遗漏，是设计。

数据文件默认读 ``data/processed/advisors.jsonl``（本地生成、不入库）；
找不到时退回两位虚构导师，仅用于演示界面，并明确标注。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from advisor_match.config import settings
from advisor_match.email.confirm import (
    build_record,
    mark_confirmed,
    save_confirmation,
)
from advisor_match.email.draft import render_plain_text
from advisor_match.email.validator import audit_refs, sentence_origin, validate_draft
from advisor_match.ingest.advisors import load_advisors
from advisor_match.llm.client import get_client
from advisor_match.pipeline import EmailResult, RecipientMismatch, run_email, run_matching
from advisor_match.product.export import confirmation_json, match_result_json
from advisor_match.product.extraction import extract_records
from advisor_match.product.materials import (
    decode_material,
    extract_docx_text,
    extract_pdf_text,
    extract_signature,
    statements_from_text,
    upload_token,
)
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
st.markdown(
    """
    <style>
    /* 提示框停留约 3 秒后淡出，与 Streamlit 的自动关闭衔接 */
    @keyframes am-toast-fade {
        0%, 70% { opacity: 1; }
        100% { opacity: 0; }
    }
    [data-testid="stToast"] {
        animation: am-toast-fade 4s ease-in forwards;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_DATA = os.getenv("AM_ADVISORS_PATH", "data/processed/advisors.jsonl")

# 能力默认自动启用：有 key 用模型，有本地模型用向量；测试用环境变量关闭。
USE_VECTOR = os.getenv("AM_DISABLE_VECTOR", "0") != "1"
USE_LLM = bool(settings.llm_api_key) and os.getenv("AM_DISABLE_LLM", "0") != "1"
USE_LLM_EXTRACT = USE_LLM
USE_LLM_RERANK = USE_LLM
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


def ensure_index() -> tuple[AdvisorIndex, bool, str]:
    """首次匹配时才加载索引，页面本身不必等向量模型。"""
    if "advisor_index" not in st.session_state:
        if Path(DEFAULT_DATA).exists():
            loaded_index, loaded_vector = load_index(
                DEFAULT_DATA, read_mtime(DEFAULT_DATA), USE_VECTOR
            )
            source = "真实数据"
        else:
            embedding = load_embedding() if USE_VECTOR else None
            loaded_index = AdvisorIndex.build(demo_advisors(), embedding=embedding)
            loaded_vector = bool(embedding)
            source = "演示数据"
        st.session_state["advisor_index"] = loaded_index
        st.session_state["vector_on"] = loaded_vector
        st.session_state["database_source"] = source
    return (
        st.session_state["advisor_index"],
        st.session_state["vector_on"],
        st.session_state["database_source"],
    )


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

# 解析台账时在主区域常驻状态条；解析完成或重跑后自动消失。
upload_status = st.empty()

with st.sidebar:
    st.subheader("你的材料")
    uploaded = st.file_uploader(
        "上传简历（必填，PDF / DOCX / txt / md）",
        type=["txt", "md", "pdf", "docx"],
        key="materials_upload",
        help="上传后会解析成台账；PDF 若为扫描版可能没有文字层。",
    )
    if uploaded is not None:
        token = ""
        try:
            raw = uploaded.getvalue()
            token = upload_token(uploaded.name, uploaded.size, raw)
        except Exception as error:  # noqa: BLE001 - 会话失效时文件读不出来，界面不能崩
            print(f"[upload] read failed: {type(error).__name__}: {error}", file=sys.stderr)
            st.session_state["upload_summary"] = "上传文件读取失败"
            st.session_state["upload_warnings"] = [
                f"读取失败（{type(error).__name__}），请重新上传"
            ]
            st.session_state["upload_token"] = ""
            st.session_state.pop("uploaded_records", None)
            st.session_state.pop("uploaded_snapshot", None)

        # 解析失败后按 file_id 记住，避免每次交互都重试；重新上传同一文件会换 file_id，仍可重试。
        file_key = (getattr(uploaded, "file_id", ""), token)
        if (
            token
            and st.session_state.get("upload_token") != token
            and st.session_state.get("upload_failed") != file_key
        ):
            try:
                extra_warnings: list[str] = []
                suffix = uploaded.name.lower().rsplit(".", 1)[-1]
                if suffix == "pdf":
                    text, extra_warnings = extract_pdf_text(raw)
                elif suffix == "docx":
                    text, extra_warnings = extract_docx_text(raw)
                else:
                    text = decode_material(raw)

                suggested_signature = extract_signature(text)
                st.session_state["suggested_signature"] = suggested_signature

                if USE_LLM_EXTRACT and settings.llm_api_key:
                    with upload_status.status(
                        "模型正在整理台账，约 10–30 秒，请勿关闭页面…",
                        expanded=False,
                    ) as status:
                        result = extract_records(text, get_client())
                        status.update(
                            label=f"台账整理完成：{len(result.records)} 条",
                            state="complete",
                        )
                    parsed = [record.statement for record in result.records]
                    warnings = list(result.warnings)
                    summary = f"模型整理完成：{len(result.records)} 条台账，引文已逐条校验"
                    st.session_state["uploaded_records"] = [
                        record.model_dump(mode="json") for record in result.records
                    ]
                    st.session_state["uploaded_snapshot"] = parsed

                    # 自动填目标方向，但不覆盖用户手改过的内容
                    suggested = ", ".join(result.directions)
                    if suggested:
                        current = st.session_state.get("targets", "")
                        previous = st.session_state.get("targets_autofilled", "")
                        if not current.strip() or current == previous:
                            st.session_state["targets"] = suggested
                            st.session_state["targets_autofilled"] = suggested
                            st.session_state["targets_note"] = (
                                "已根据材料自动填写目标方向，可以直接修改。"
                            )
                else:
                    parsed, warnings = statements_from_text(text)
                    summary = f"按行解析：{len(parsed)} 条陈述"
                    st.session_state.pop("uploaded_records", None)
                    st.session_state.pop("uploaded_snapshot", None)
            except Exception as error:  # noqa: BLE001 - 解析失败要留下线索，不能静默留空
                print(f"[upload] parse failed: {type(error).__name__}: {error}", file=sys.stderr)
                st.session_state["upload_summary"] = "上传文件解析失败"
                st.session_state["upload_warnings"] = [
                    f"解析失败（{type(error).__name__}），请重新上传，或在左侧手动填写成果陈述"
                ]
                st.session_state["upload_failed"] = file_key
                st.session_state.pop("uploaded_records", None)
                st.session_state.pop("uploaded_snapshot", None)
            else:
                st.session_state["upload_failed"] = None
                st.session_state["statements"] = "\n".join(parsed)
                st.session_state["upload_summary"] = summary
                st.session_state["upload_token"] = token
                st.session_state["upload_warnings"] = [*extra_warnings, *warnings]

                # 台账立刻换成新材料；旧的匹配与草稿作废，等学生重新点匹配。
                stored = st.session_state.get("uploaded_records")
                if stored:
                    st.session_state["ledger"] = {
                        item["id"]: LedgerRecord.model_validate(item) for item in stored
                    }
                elif parsed:
                    st.session_state["ledger"] = build_profile([], [], parsed)[1]
                else:
                    st.session_state["ledger"] = {}
                for stale in (
                    "result",
                    "profile",
                    "index_size",
                    "index_skipped",
                    "vector_on",
                    "logs",
                    "data_source",
                    "drafts",
                    "draft",
                    "contact",
                    "revision",
                    "edited_total",
                    "confirmation",
                ):
                    st.session_state.pop(stale, None)
                st.rerun()

    if st.session_state.get("upload_summary"):
        st.caption(st.session_state["upload_summary"])
    for warning in st.session_state.get("upload_warnings", []):
        st.caption(f"· {warning}")
    st.caption("上传后会自动整理台账；启用模型时先脱敏，约 10–30 秒。")

    target_text = st.text_input(
        "目标方向（可选，逗号分隔）",
        placeholder="可留空，上传材料后会自动填写",
        key="targets",
        help="用于召回导师；留空时只靠材料陈述召回。上传后可用模型推断。",
    )
    if st.session_state.get("targets_note"):
        st.caption(st.session_state["targets_note"])

    interest_text = st.text_input(
        "研究兴趣（可选，逗号分隔）",
        placeholder="可选，例如：大模型, 检索增强",
        key="interests",
    )

    statement_text = st.text_area(
        "成果陈述（必填，每行一条）",
        placeholder=(
            "每行一条，例如：\n完成多模态检索增强生成的课程项目\n"
            "以二作身份发表 ACL Findings 2025 论文"
        ),
        height=120,
        key="statements",
        help="上传文件会自动填入；也可以直接手写。",
    )

    start = st.button("开始匹配", type="primary", key="run_matching")

    st.divider()
    st.caption(f"召回上限 {settings.retrieval.recall_k}，主榜 {settings.retrieval.keep_k}")
    st.caption("本应用不具备发信权限。")

if start:
    statements = [line.strip() for line in statement_text.splitlines() if line.strip()]
    targets = [item.strip() for item in target_text.split(",") if item.strip()]
    interests = [item.strip() for item in interest_text.split(",") if item.strip()]

    if uploaded is None and not statements:
        st.error("请先上传简历 PDF；如果没有文件，也可以手动填写成果陈述。")
    elif not statements:
        name = uploaded.name if uploaded is not None else "上传的文件"
        summary = st.session_state.get("upload_summary")
        hint = f"（{summary}）" if summary else ""
        st.error(
            f"已收到《{name}》{hint}，但没有解析出陈述。"
            "如果 PDF 是扫描版（没有文字层），请换文字版；"
            "也可以在左侧「成果陈述」里手动补充一条。"
        )
    else:
        stored = st.session_state.get("uploaded_records")
        snapshot = st.session_state.get("uploaded_snapshot")
        if stored and snapshot and statements == snapshot:
            ledger = {
                item["id"]: LedgerRecord.model_validate(item) for item in stored
            }
            profile = StudentProfile(
                ledger=list(ledger.values()),
                target_directions=targets,
                interests=interests,
            )
        else:
            profile, ledger = build_profile(targets, interests, statements)
        first_run = "advisor_index" not in st.session_state
        if first_run:
            message = (
                "首次匹配：正在加载导师索引并调用模型重排，约 30–90 秒…"
                if USE_LLM_RERANK
                else "首次匹配：正在加载导师索引并召回，约 5–10 秒…"
            )
        elif USE_LLM_RERANK:
            message = "模型重排中，约 30–90 秒，请勿关闭页面…"
        else:
            message = "正在召回与重排…"
        with st.spinner(message):
            index, vector_on, database_source = ensure_index()
            result, logs = run_matching(
                profile,
                index,
                llm=get_client() if USE_LLM_RERANK else None,
                rerank_mode="hybrid",
            )
        if database_source == "演示数据":
            st.warning("未找到导师数据文件，当前使用两位虚构导师演示。")
        matched_count = len(result.core) + len(result.adjacent) + len(result.needs_support)
        extra = f"，另有 {len(result.recall_only)} 位仅方向相关" if result.recall_only else ""
        st.toast(f"匹配完成：{matched_count} 位导师进入推荐{extra}。", icon="✅")

        st.session_state.update(
            {
                "result": result,
                "ledger": ledger,
                "profile": profile,
                "index_size": len(index.advisors),
                "index_skipped": len(index.skipped),
                "vector_on": vector_on,
                "logs": logs,
                "data_source": database_source,
            }
        )
        st.session_state["drafts"] = {}
        for stale in ("draft", "contact", "revision", "edited_total", "confirmation"):
            st.session_state.pop(stale, None)

result = st.session_state.get("result")
ledger = st.session_state.get("ledger") or {}

section = st.radio(
    "页面",
    ["① 材料与台账", "② 导师匹配", "③ 邮件草稿"],
    horizontal=True,
    label_visibility="collapsed",
    key="section",
)

if section == "① 材料与台账":
    st.subheader("成果台账")
    if ledger:
        lines = ["| ID | 陈述 | 原文 |", "| --- | --- | --- |"]
        for record in ledger.values():
            statement = record.statement.replace("|", chr(92) + "|")
            source = f"{record.evidence[0].doc_id} · {record.evidence[0].locator}".replace("|", chr(92) + "|")
            lines.append(f"| {record.id} | {statement} | {source} |")
        st.markdown("\n".join(lines))
    else:
        st.info("在左侧填写材料并点击「开始匹配」。")

if section == "② 导师匹配":
    if result is None:
        if st.session_state.get("ledger"):
            st.info("材料已更新，请点击左侧「开始匹配」生成推荐。")
        else:
            st.info("尚未开始匹配。")
    else:
        if not (result.core or result.adjacent or result.needs_support):
            stage_logs = {log.stage: log for log in st.session_state.get("logs", [])}
            if stage_logs.get("recall") and stage_logs["recall"].output_count == 0:
                st.info(
                    "召回没有候选：目标方向与导师库的词汇差异太大，或该方向没有导师。"
                    "可以换更常见的方向词，或补充研究兴趣。"
                )
            else:
                rerank_log = stage_logs.get("rerank")
                count = rerank_log.input_count if rerank_log else 0
                st.info(
                    f"召回到 {count} 位导师，但没有任何判断能绑定到你的台账。"
                    "请确认陈述里写到了具体研究方向或项目；系统不会生成没有依据的推荐。"
                )

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

        if result.recall_only:
            st.markdown("### 其他召回候选（未进入三档）")
            st.caption(
                "这些导师被系统召回但没有进入前三档，仅供你自行判断；"
                "不会为它们生成邮件草稿。"
            )
            for match in result.recall_only:
                with st.expander(match.advisor.name):
                    render_advisor_meta(match.advisor)
                    for point in match.match_points:
                        st.write(f"· {point.statement}")
                    for gap in match.gaps:
                        st.write(gap.statement)

        if result.excluded:
            with st.expander(f"未列入（{len(result.excluded)} 位）"):
                for name, reason in result.excluded.items():
                    st.caption(f"{name}：{reason}")

        st.download_button(
            "下载匹配结果（JSON）",
            data=match_result_json(result),
            file_name="match_result.json",
            mime="application/json",
            key="download_match",
        )

if section == "③ 邮件草稿":
    if result is None:
        st.info("先完成匹配，再选择一位导师生成草稿。")
    else:
        selected = [*result.core, *result.adjacent, *result.needs_support]
        names = [match.advisor.name for match in selected]
        if not names:
            st.warning("当前没有可写的导师。")
        else:
            drafts: dict[str, dict] = st.session_state.setdefault("drafts", {})
            if drafts:
                st.caption("已保存草稿：" + "、".join(drafts))

            name = st.selectbox("选择导师", names, key="advisor_select")
            match = selected[names.index(name)]
            advisor = match.advisor
            entry = drafts.get(name)

            suggested = st.session_state.get("suggested_signature")
            if suggested:
                current = st.session_state.get("signature", "")
                previous = st.session_state.get("signature_autofilled", "")
                if not current.strip() or current == previous:
                    st.session_state["signature"] = suggested
                    st.session_state["signature_autofilled"] = suggested

            signature = st.text_input(
                "署名（只用于生成草稿，不保存）",
                placeholder="张三　计算机系 2023 级",
                key="signature",
                help="从简历自动填写姓名与院系，可直接修改。",
            )
            if suggested:
                st.caption("署名已根据简历自动填写，可直接修改。")
            points_text = st.text_area(
                "你自己的想法（每行一句，只保留能对上台账的句子）", height=90, key="points"
            )
            generate = st.button("生成草稿", key="generate_draft")

            if generate:
                if not advisor.email:
                    st.error("该导师邮箱未收录，无法确定收件人。")
                else:
                    contact = AdvisorContact(display_name=advisor.name, email=advisor.email)
                    message = (
                        "模型正在生成草稿，约 10–20 秒，请勿关闭页面…"
                        if USE_LLM
                        else "正在生成草稿…"
                    )
                    try:
                        with st.spinner(message):
                            email_result = run_email(
                                match,
                                contact,
                                ledger,
                                student_points=[
                                    line.strip()
                                    for line in points_text.splitlines()
                                    if line.strip()
                                ],
                                student_signature=signature,
                                llm=get_client() if USE_LLM else None,
                            )
                    except RecipientMismatch as error:
                        st.error(f"收件人核对未通过：{error}")
                    else:
                        st.session_state["draft_seq"] = (
                            st.session_state.get("draft_seq", 0) + 1
                        )
                        entry = {
                            "result": email_result,
                            "contact": contact,
                            "generation": st.session_state["draft_seq"],
                            "revision": 0,
                            "edited_total": 0,
                            "confirmation": None,
                        }
                        drafts[name] = entry
                        # 生成和渲染分开：先结束本次运行，再由下一次运行一次性铺开草稿，
                        # 避免长草稿在模型调用结束后边生成边往下刷。
                        st.rerun()

            if entry is not None:
                if st.button("清除该导师草稿", key="clear_draft"):
                    drafts.pop(name, None)
                    st.rerun()
                email_result = entry["result"]
                draft = email_result.draft
                st.markdown(f"**主题：** {draft.subject}")
                st.caption(
                    f"校验：保留 {len(email_result.report.kept)} 句，"
                    f"丢弃无据陈述 {len(email_result.report.dropped)} 句"
                )

                revision = entry["revision"]
                generation = entry.get("generation", 0)
                for paragraph in draft.paragraphs:
                    st.markdown(f"**{paragraph.role}**")
                    for index, sentence in enumerate(paragraph.sentences):
                        sid = f"{name}-{generation}-{revision}-{paragraph.role}-{index}"
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
                            st.checkbox(
                                "删除",
                                key=f"del-{sid}",
                                help="勾选后点「完成确认」时移除这句。",
                            )

                if st.button("完成确认并生成文本", key="confirm_draft"):
                    new_paragraphs = []
                    edited = 0
                    for paragraph in draft.paragraphs:
                        kept = []
                        for index, sentence in enumerate(paragraph.sentences):
                            sid = f"{name}-{generation}-{revision}-{paragraph.role}-{index}"
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
                        mark_confirmed(updated, all_sentences_confirmed=True)
                        record = build_record(
                            updated,
                            entry["contact"],
                            refs_used=audit_refs(updated, ledger),
                            edited=entry["edited_total"] + edited,
                        )
                        save_confirmation(record, CONFIRMATION_LOG)
                        entry["result"] = EmailResult(draft=updated, report=report)
                        entry["edited_total"] += edited
                        entry["revision"] += 1
                        entry["confirmation"] = record
                        entry["just_confirmed"] = True
                        st.rerun()

                saved = entry.get("confirmation")
                if saved is not None:
                    if entry.pop("just_confirmed", False):
                        st.toast(
                            f"已留痕：{saved.advisor_name}，{saved.sentence_count} 句，"
                            f"{len(saved.refs_used)} 条台账依据",
                            icon="✅",
                        )
                    st.text_area(
                        "复制到你自己的邮件客户端（本应用不会发送）",
                        value=render_plain_text(draft, ledger),
                        height=320,
                    )
                else:
                    st.info("确认内容与材料一致后，点击「完成确认」生成留痕与可粘贴文本。")
                    st.text_area(
                        "草稿预览",
                        value=render_plain_text(draft, ledger),
                        height=320,
                        disabled=True,
                    )

                st.download_button(
                    "下载草稿（txt）",
                    data=render_plain_text(draft, ledger),
                    file_name=f"draft_{name}.txt",
                    mime="text/plain",
                    key=f"download_draft_{name}",
                )
                if saved is not None:
                    st.download_button(
                        "下载确认记录（JSON）",
                        data=confirmation_json(saved),
                        file_name=f"confirmation_{name}.json",
                        mime="application/json",
                        key=f"download_confirmation_{name}",
                    )
