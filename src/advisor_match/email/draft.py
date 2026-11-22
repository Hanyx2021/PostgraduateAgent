"""生成四段式自荐邮件草稿。

四段：问候与来意 / 契合点 / 差距与补充 / 收尾。

两条硬约束写在接口上：
- 返回值是 ``EmailDraft``，调用方**无法**把它变成已发送状态（schema 没有该状态）；
- 每个事实句必须带 ``ledger_refs``，由 ``validator.validate_draft`` 二次把关。

默认走确定性模板（不需要模型、不联网、可复现）；传入 ``llm`` 时尝试模型改写，
模型输出不可解析或缺少依据时自动退回模板，而不是把异常抛给学生。
"""

from __future__ import annotations

import json
import re

from advisor_match.llm import prompts
from advisor_match.llm.client import LLMClient
from advisor_match.schemas import (
    AdvisorContact,
    AdvisorMatch,
    EmailDraft,
    EmailParagraph,
    EmailSentence,
    LedgerRecord,
)

DIMENSION_LABELS = {
    "direction": "研究方向",
    "method": "技术路线",
    "experience": "项目经历",
    "resource": "资源条件",
}

SIGNATURE_PLACEHOLDER = "（请在发送前填写您的姓名、院系与联系方式）"


def _sentence(text: str, *, kind: str, refs: list[str] | None = None) -> EmailSentence:
    return EmailSentence(text=text, kind=kind, ledger_refs=refs or [])  # type: ignore[arg-type]


def _with_period(text: str) -> str:
    text = text.strip()
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else f"{text}。"


def _teacher_name(name: str) -> str:
    """称呼拼接：已经是"X老师"的不再补一次，避免"李老师老师"。"""
    return name if name.endswith("老师") else f"{name}老师"


def _bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"[\s，。；：、,.;:!?（）()]", "", text)
    return {cleaned[index : index + 2] for index in range(len(cleaned) - 1)}


def match_student_points(
    points: list[str], ledger: dict[str, LedgerRecord]
) -> list[tuple[str, str]]:
    """把学生自写的句子映射到台账记录，返回（句子, 台账 ID）。

    只做字面重合匹配，不做语义推断；对不上的句子直接丢弃，宁可少写一句，
    也不让没有出处的自述进入邮件。
    """
    matched: list[tuple[str, str]] = []
    for point in points:
        point_grams = _bigrams(point)
        best_id, best_score = "", 0
        for record in ledger.values():
            score = len(point_grams & _bigrams(record.statement))
            if score > best_score:
                best_id, best_score = record.id, score
        if best_id and best_score >= 2:
            matched.append((point, best_id))
    return matched


def _template_draft(
    match: AdvisorMatch,
    contact: AdvisorContact,
    ledger: dict[str, LedgerRecord],
    *,
    signature: str,
    student_points: list[str] | None,
) -> EmailDraft:
    greeting = EmailParagraph(
        role="greeting",
        sentences=[
            _sentence(f"尊敬的{_teacher_name(contact.display_name)}：您好！", kind="greeting"),
            _sentence("我是准备申请推免的本科生，冒昧写信向您请教。", kind="greeting"),
        ],
    )

    fit_sentences: list[EmailSentence] = []
    for point_text, record_id in match_student_points(student_points or [], ledger):
        fit_sentences.append(
            _sentence(_with_period(point_text), kind="fact", refs=[record_id])
        )
    for point in match.match_points:
        prefix = DIMENSION_LABELS.get(point.dimension, "契合点")
        fit_sentences.append(
            _sentence(
                _with_period(f"就{prefix}而言，{point.statement}"),
                kind="fact",
                refs=point.student_ledger_refs,
            )
        )
    if not fit_sentences:
        fit_sentences.append(
            _sentence("我认真读过您近期的研究工作，希望有机会进一步了解课题组的方向。", kind="intent")
        )
    fit = EmailParagraph(role="fit", sentences=fit_sentences)

    gap_sentences = [
        _sentence(
            _with_period(f"我也注意到，{gap.statement.rstrip('。')}，这方面我希望在研究生阶段继续加强"),
            kind="intent",
        )
        for gap in match.gaps[:2]
    ]
    if not gap_sentences:
        gap_sentences.append(_sentence("我也希望了解您对申请者背景的具体期待。", kind="intent"))
    gap = EmailParagraph(role="gap", sentences=gap_sentences)

    closing = EmailParagraph(
        role="closing",
        sentences=[
            _sentence("随信附上简历与成绩单，期待您的回复。", kind="intent"),
            _sentence(f"祝好！\n{signature}", kind="greeting"),
        ],
    )

    return EmailDraft(
        recipient=contact,
        subject=f"推免自荐｜{_teacher_name(contact.display_name)}",
        paragraphs=[greeting, fit, gap, closing],
    )


def _extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出中没有 JSON 对象")
    return json.loads(raw[start : end + 1])


def _signature_sentence(closing: EmailParagraph, signature: str) -> EmailSentence:
    """模型常漏署名；已经有"此致/敬礼"时只补姓名，避免重复寒暄。"""
    closing_text = "".join(sentence.text for sentence in closing.sentences)
    if any(marker in closing_text for marker in ("此致", "敬礼", "祝好", "顺颂")):
        return EmailSentence(text=f"\n{signature}", kind="greeting")
    return EmailSentence(text=f"祝好！\n{signature}", kind="greeting")


def _llm_draft(
    match: AdvisorMatch,
    contact: AdvisorContact,
    ledger: dict[str, LedgerRecord],
    llm: LLMClient,
    *,
    signature: str,
    student_points: list[str] | None,
) -> EmailDraft | None:
    user = prompts.EMAIL_USER.format(
        student_points="\n".join(student_points or []) or "（学生未提供）",
        ledger_json=json.dumps(
            [record.model_dump() for record in ledger.values()], ensure_ascii=False
        ),
        match_json=match.model_dump_json(),
    )
    # 模型偶尔会漏段或输出非 JSON；重试一次，仍失败才退回模板。
    for _ in range(2):
        try:
            data = _extract_json(llm.complete(system=prompts.EMAIL_SYSTEM, user=user))
            paragraphs = [
                EmailParagraph(
                    role=paragraph["role"],
                    sentences=[EmailSentence(**sentence) for sentence in paragraph["sentences"]],
                )
                for paragraph in data["paragraphs"]
            ]
            if {paragraph.role for paragraph in paragraphs} != {
                "greeting",
                "fit",
                "gap",
                "closing",
            }:
                continue
            if signature not in "".join(
                sentence.text for paragraph in paragraphs for sentence in paragraph.sentences
            ):
                closing = next(
                    paragraph for paragraph in paragraphs if paragraph.role == "closing"
                )
                closing.sentences.append(_signature_sentence(closing, signature))
            return EmailDraft(
                recipient=contact,
                subject=str(data["subject"]),
                paragraphs=paragraphs,
            )
        except Exception:  # noqa: BLE001 - 模型不可用/输出异常都进入重试
            continue
    return None


def build_draft(
    match: AdvisorMatch,
    contact: AdvisorContact,
    ledger: dict[str, LedgerRecord],
    *,
    student_voice_samples: list[str] | None = None,
    student_signature: str | None = None,
    llm: LLMClient | None = None,
) -> EmailDraft:
    """生成草稿。

    ``student_voice_samples``：学生先自写的要点。系统只把能对上台账的句子放进去，
    对不上的丢弃；"写什么"的决定权仍在学生手里（幻灯片 5）。
    ``student_signature``：署名由学生在生成时填写，不进入画像、不落库。
    """
    signature = (student_signature or "").strip() or SIGNATURE_PLACEHOLDER
    if llm is not None:
        draft = _llm_draft(
            match,
            contact,
            ledger,
            llm,
            signature=signature,
            student_points=student_voice_samples,
        )
        if draft is not None:
            return draft
    return _template_draft(
        match,
        contact,
        ledger,
        signature=signature,
        student_points=student_voice_samples,
    )


def render_plain_text(draft: EmailDraft, ledger: dict[str, LedgerRecord]) -> str:
    """渲染成可直接粘贴的纯文本。

    逐句确认通过后才应调用；本函数不发送任何东西。
    """
    lines = [f"收件人：{draft.recipient.display_name} <{draft.recipient.email}>", f"主题：{draft.subject}", ""]
    for paragraph in draft.paragraphs:
        lines.append("".join(sentence.text for sentence in paragraph.sentences))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
