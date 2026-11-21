"""证据化重排：逐对（学生台账 × 导师材料）抽取匹配点与差距。

判定有两条通道，都必须是可核验的：
1. 字面重合（字符二元组），用于"信息检索 vs 信息检索"这类直接对应；
2. 语义相近（本地 BGE 向量余弦 ≥ 阈值），用于"人工智能 vs 机器学习"这类
   词汇差异。语义通道的措辞必须写成"语义上相近"，不能假装是字面一致。

每条判断仍绑定学生台账 ID 和导师原文引文；``verify_grounding`` 继续兜底。
"""

from __future__ import annotations

import json
import re

import numpy as np

from advisor_match.config import settings
from advisor_match.ledger.build import quote_is_grounded
from advisor_match.llm import prompts
from advisor_match.llm.client import LLMClient
from advisor_match.retrieval.recall import build_query
from advisor_match.retrieval.vector import EmbeddingProvider
from advisor_match.schemas import (
    Advisor,
    AdvisorMatch,
    Gap,
    LedgerRecord,
    MatchPoint,
    SourceRef,
    StudentProfile,
)

PUNCTUATION = re.compile(r"[\s，。；：、,.;:!?（）()]")


def _bigrams(text: str) -> set[str]:
    cleaned = PUNCTUATION.sub("", text)
    return {cleaned[index : index + 2] for index in range(len(cleaned) - 1)}


def _overlap(left: str, right: str) -> int:
    return len(_bigrams(left) & _bigrams(right))


def _student_text(profile: StudentProfile) -> str:
    parts = [*profile.target_directions, *profile.interests, build_query(profile)]
    return " ".join(part for part in parts if part)


def _teacher_name(name: str) -> str:
    return name if name.endswith("老师") else f"{name}老师"


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.dot(left, right))


def _supporting_record(
    tag: str,
    ledger: dict[str, LedgerRecord],
    *,
    tag_vector: np.ndarray | None = None,
    record_vectors: dict[str, np.ndarray] | None = None,
) -> tuple[LedgerRecord | None, bool]:
    """找一条能支撑该方向的台账记录，返回（记录, 是否靠语义匹配）。"""
    best: LedgerRecord | None = None
    best_score = (0, 0.0)
    threshold = settings.retrieval.rerank_min_similarity
    for record in ledger.values():
        lexical = _overlap(tag, record.statement)
        semantic = _cosine(tag_vector, (record_vectors or {}).get(record.id))
        score = (lexical, semantic)
        if score > best_score:
            best, best_score = record, score
    if best is None:
        return None, False
    if best_score[0] >= 1:
        return best, False
    if best_score[1] >= threshold:
        return best, True
    return None, False


def _advisor_sources(advisor: Advisor) -> dict[str, str]:
    return {ref.doc_id: ref.quote for ref in advisor.sources}


def _extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出中没有 JSON 对象")
    return json.loads(raw[start : end + 1])


def _llm_points(
    profile: StudentProfile,
    advisor: Advisor,
    ledger: dict[str, LedgerRecord],
    sources: dict[str, str],
    llm: LLMClient,
) -> list[MatchPoint]:
    """让模型给判断，但引文必须来自给定 sources；不合规的判断直接丢弃。"""
    user = prompts.RERANK_USER.format(
        ledger_json=json.dumps(
            [record.model_dump() for record in ledger.values()], ensure_ascii=False
        ),
        advisor_json=advisor.model_dump_json(),
        sources_json=json.dumps(sources, ensure_ascii=False),
    )
    try:
        data = _extract_json(llm.complete(system=prompts.RERANK_SYSTEM, user=user))
    except Exception:  # noqa: BLE001 - 模型不可用时退回规则版
        return []

    points: list[MatchPoint] = []
    for item in data.get("match_points", [])[:3]:
        if not isinstance(item, dict):
            continue
        refs = [ref for ref in item.get("student_ledger_refs", []) if ref in ledger]
        statement = str(item.get("statement", "")).strip()
        doc_id = str(item.get("source_doc_id", ""))
        quote = str(item.get("advisor_quote", "")).strip()
        dimension = item.get("dimension", "direction")
        if dimension not in {"direction", "method", "experience", "resource"}:
            dimension = "direction"
        if not refs or not statement or doc_id not in sources or not quote:
            continue
        if not quote_is_grounded(quote, sources[doc_id]):
            continue
        points.append(
            MatchPoint(
                dimension=dimension,
                statement=statement[:300],
                student_ledger_refs=refs,
                advisor_refs=[SourceRef(doc_id=doc_id, locator="模型判断", quote=quote[:800])],
            )
        )
    return points


def rerank_one(
    profile: StudentProfile,
    advisor: Advisor,
    *,
    advisor_source_text: dict[str, str] | None = None,
    embedding: EmbeddingProvider | None = None,
    llm: LLMClient | None = None,
) -> AdvisorMatch | None:
    """对一位导师产出匹配结果；证据不足时返回 None（宁可少给，不可编造）。

    证据有两路：详情页的「研究领域」和 2027 招生目录的方向。目录方向是官方公开
    信息，能补上研究标签缺失的导师，也不依赖课题组主页是否维护。
    """
    sources = (
        dict(advisor_source_text)
        if advisor_source_text is not None
        else _advisor_sources(advisor)
    )
    for admission in advisor.admissions:
        existing = sources.get(admission.source, "")
        sources[admission.source] = f"{existing}，{admission.direction}".strip("，")
    if not sources:
        return None

    ledger = {record.id: record for record in profile.ledger}
    student_text = _student_text(profile)
    doc_id = next(iter(sources))
    threshold = settings.retrieval.rerank_min_similarity

    tag_vectors: list[np.ndarray] | None = None
    admission_vectors: list[np.ndarray] | None = None
    direction_vectors: list[np.ndarray] | None = None
    student_vector: np.ndarray | None = None
    record_vectors: dict[str, np.ndarray] = {}
    if embedding is not None:
        student_vector = embedding.encode([student_text])[0]
        if advisor.research_tags:
            tag_vectors = list(embedding.encode(advisor.research_tags))
        if advisor.admissions:
            admission_vectors = list(
                embedding.encode([admission.direction for admission in advisor.admissions])
            )
        if profile.target_directions:
            direction_vectors = list(embedding.encode(profile.target_directions))
        if ledger:
            statement_vectors = embedding.encode([record.statement for record in ledger.values()])
            record_vectors = dict(zip(ledger.keys(), statement_vectors))

    points: list[MatchPoint] = []
    if llm is not None:
        points = _llm_points(profile, advisor, ledger, sources, llm)
    if not points:
        matched_tags: list[str] = []
        matched_tag_vectors: list[np.ndarray | None] = []
        for position, tag in enumerate(advisor.research_tags):
            lexical = _overlap(tag, student_text)
            tag_vector = tag_vectors[position] if tag_vectors is not None else None
            semantic = _cosine(tag_vector, student_vector)
            if lexical < 1 and semantic < threshold:
                continue

            record, used_semantic = _supporting_record(
                tag, ledger, tag_vector=tag_vector, record_vectors=record_vectors
            )
            if record is None:
                continue

            relation = "在语义上相近" if used_semantic else "相关"
            statement = (
                f"你的成果「{record.statement[:40]}」与{_teacher_name(advisor.name)}的研究方向"
                f"「{tag}」{relation}"
            )
            points.append(
                MatchPoint(
                    dimension="direction",
                    statement=statement[:300],
                    student_ledger_refs=[record.id],
                    advisor_refs=[SourceRef(doc_id=doc_id, locator="研究领域", quote=tag)],
                )
            )
            matched_tags.append(tag)
            matched_tag_vectors.append(tag_vector)
            if len(points) >= 3:
                break

        # 招生目录方向：与研究标签去重后作为独立证据，引文指向目录页。
        for position, admission in enumerate(advisor.admissions):
            lexical = _overlap(admission.direction, student_text)
            admission_vector = (
                admission_vectors[position] if admission_vectors is not None else None
            )
            semantic = _cosine(admission_vector, student_vector)
            if lexical < 1 and semantic < threshold:
                continue
            if any(
                _overlap(admission.direction, tag) >= 1
                or _cosine(admission_vector, vector) >= threshold
                for tag, vector in zip(matched_tags, matched_tag_vectors)
            ):
                continue

            record, used_semantic = _supporting_record(
                admission.direction,
                ledger,
                tag_vector=admission_vector,
                record_vectors=record_vectors,
            )
            if record is None:
                continue

            relation = "在语义上相近" if used_semantic else "相关"
            note = "（该方向仅推免）" if admission.only_recommendation else ""
            statement = (
                f"你的成果「{record.statement[:40]}」与{_teacher_name(advisor.name)}在 "
                f"{admission.year} 年招生目录中的「{admission.direction}」方向{relation}{note}"
            )
            points.append(
                MatchPoint(
                    dimension="direction",
                    statement=statement[:300],
                    student_ledger_refs=[record.id],
                    advisor_refs=[
                        SourceRef(
                            doc_id=admission.source,
                            locator=f"{admission.year} 招生目录 / {admission.program_code} {admission.direction_code}",
                            quote=admission.direction,
                        )
                    ],
                )
            )
            if len(points) >= 3:
                break

    if not points:
        return None

    # 档位按"目标方向覆盖"判断：全部覆盖 → 核心；部分覆盖 → 相邻；只有弱重合 → 需要补充。
    covered_flags = []
    for index, direction in enumerate(profile.target_directions):
        covered = any(
            _overlap(direction, tag) >= 1
            or (
                direction_vectors is not None
                and tag_vectors is not None
                and _cosine(direction_vectors[index], tag_vectors[tag_index]) >= threshold
            )
            for tag_index, tag in enumerate(advisor.research_tags)
        ) or any(
            _overlap(direction, admission.direction) >= 1
            or (
                direction_vectors is not None
                and admission_vectors is not None
                and _cosine(direction_vectors[index], admission_vectors[position]) >= threshold
            )
            for position, admission in enumerate(advisor.admissions)
        )
        covered_flags.append(covered)
    if profile.target_directions and all(covered_flags):
        tier = "core"
    elif any(covered_flags):
        tier = "adjacent"
    else:
        tier = "needs_support"

    gaps: list[Gap] = []
    for index, direction in enumerate(profile.target_directions):
        if covered_flags[index]:
            continue
        gaps.append(Gap(statement=f"你的目标方向「{direction}」在该课题组的研究领域中体现较少"))
        if len(gaps) >= 2:
            break

    match = AdvisorMatch(advisor=advisor, tier=tier, match_points=points, gaps=gaps)
    match, _ = verify_grounding(match, ledger, sources)
    if not match.match_points:
        return None
    return match


def verify_grounding(
    match: AdvisorMatch,
    ledger: dict[str, LedgerRecord],
    advisor_source_text: dict[str, str],
) -> tuple[AdvisorMatch, list[str]]:
    """剔除无依据的匹配点与差距，返回（清理后的结果, 剔除原因）。"""
    kept_points: list[MatchPoint] = []
    rejected: list[str] = []

    for point in match.match_points:
        missing_refs = [ref for ref in point.student_ledger_refs if ref not in ledger]
        if missing_refs:
            rejected.append(f"台账中不存在 {missing_refs}")
            continue
        bad_quotes = [
            ref for ref in point.advisor_refs
            if ref.doc_id not in advisor_source_text
            or not quote_is_grounded(ref.quote, advisor_source_text[ref.doc_id])
        ]
        if bad_quotes:
            rejected.append(f"导师材料中未找到引文：{[r.quote[:30] for r in bad_quotes]}")
            continue
        kept_points.append(point)

    match.match_points = kept_points
    if not kept_points:
        return match, rejected + ["没有任何一条匹配点通过证据校验"]
    return match, rejected


__all__ = ["rerank_one", "verify_grounding"]
