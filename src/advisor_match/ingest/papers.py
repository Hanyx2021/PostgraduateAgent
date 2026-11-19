"""论文补全：按英文名检索 arXiv，只保留能对上作者名的条目。

身份消歧是这一步的核心风险。做法：
1. 中文名用拼音生成英文名（如"唐杰"→"Jie Tang"），外籍名直接用；
2. 用 arXiv ``all:`` 检索（``au:`` 字段被限流），再按作者名精确过滤；
3. 用合作者重叠做连通分量，只取最大的一簇——同名的不同研究者通常不共享合作者；
4. 两个簇规模接近时标记为"存疑"，不自动写入，交给人工复核。

**实测结论：仅凭姓名 + 合作者聚类不足以核验身份。** arXiv 不提供作者机构，
2026-11 的试跑中出现了把存储方向导师配上 NLP 论文的错配。因此本模块默认
只输出候选（``--out`` 写候选 JSON），必须人工复核后再用 ``--apply`` 写入；
正式使用应以带机构信息的来源（如 OpenAlex 作者 ID）做核验。

限流与反爬：arXiv 要求请求间隔 ≥ 3 秒，这里加退避重试；失败如实记录，
不做"宁滥勿缺"的补全。
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pypinyin import lazy_pinyin

from advisor_match.ingest.advisors import load_advisors, save_advisors
from advisor_match.schemas import Advisor, Paper

ATOM = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
MIN_INTERVAL_SECONDS = 3.0
ACCEPTED_CATEGORY_PREFIXES = ("cs.", "stat.ML")
_last_request_at = 0.0


def name_variants(name: str) -> list[str]:
    """返回 [西式姓名, 中式姓名]；外籍名只返回原样。"""
    if re.fullmatch(r"[A-Za-z .'\-]+", name):
        return [name]
    syllables = lazy_pinyin(name)
    if len(syllables) < 2:
        return [" ".join(syllables)]
    # 中文名的双字名是一个整体：胡事民 → Shimin Hu，而不是 Shi Min Hu。
    surname = syllables[0]
    given = "".join(syllables[1:])
    western = f"{given.title()} {surname.title()}"
    eastern = f"{surname.title()} {given.title()}"
    return [western, eastern]


def normalize_author(name: str) -> str:
    return re.sub(r"[.\s\-]", "", name).lower()


@dataclass
class SearchResult:
    papers: list[Paper] = field(default_factory=list)
    error: str | None = None


class ArxivClient:
    """最小 arXiv 客户端。测试时可替换为假实现。"""

    def __init__(self, *, min_interval: float = MIN_INTERVAL_SECONDS, retries: int = 3) -> None:
        self.min_interval = min_interval
        self.retries = retries

    def _throttle(self) -> None:
        global _last_request_at
        wait = self.min_interval - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()

    def search_author(self, name: str, *, max_results: int = 50, year_from: int = 2021) -> SearchResult:
        variants = {normalize_author(v) for v in name_variants(name)}
        last_error = ""

        # 先试西式姓名；没有结果再试中式顺序（少数条目按"姓 名"署名）。
        for variant in name_variants(name):
            url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {
                    "search_query": f'all:"{variant}"',
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
            )
            for attempt in range(self.retries):
                self._throttle()
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "AdvisorMatch/0.1"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        payload = response.read()
                    papers = _filter_papers(payload, variants, year_from)
                    if papers:
                        return SearchResult(papers=papers)
                    break  # 查询成功但没有结果，换下一种写法
                except Exception as error:  # noqa: BLE001 - 网络错误统一退避重试
                    last_error = f"{type(error).__name__}: {error}"
                    time.sleep(5 * (attempt + 1))
            else:
                return SearchResult(error=last_error)
        return SearchResult(papers=[])


def _filter_papers(payload: bytes, variants: set[str], year_from: int) -> list[Paper]:
    root = ET.fromstring(payload)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", ATOM):
        authors = [
            (node.find("a:name", ATOM).text or "").strip() for node in entry.findall("a:author", ATOM)
        ]
        if not any(normalize_author(author) in variants for author in authors):
            continue
        published = entry.find("a:published", ATOM)
        year = int(published.text[:4]) if published is not None and published.text else 0
        if year < year_from:
            continue
        category_node = entry.find(f"{ARXIV_NS}primary_category")
        category = category_node.get("term", "") if category_node is not None else ""
        if not category.startswith(ACCEPTED_CATEGORY_PREFIXES):
            continue
        title_node = entry.find("a:title", ATOM)
        summary_node = entry.find("a:summary", ATOM)
        id_node = entry.find("a:id", ATOM)
        papers.append(
            Paper(
                title=" ".join((title_node.text or "").split()) if title_node is not None else "",
                venue=f"arXiv:{category}" if category else "arXiv",
                year=year,
                url=(id_node.text or "").strip() if id_node is not None else None,
                authors=authors,
                abstract=" ".join((summary_node.text or "").split()) if summary_node is not None else None,
            )
        )
    return papers


def select_cluster(papers: list[Paper], name: str) -> tuple[list[Paper], bool]:
    """按合作者重叠取最大簇，返回（论文, 是否存疑）。

    存疑条件：存在第二簇，且规模达到最大簇的一半以上——说明至少有两个活跃的
    同名研究者，不能自动认定。
    """
    if not papers:
        return [], False
    target = {normalize_author(name)}

    parent = list(range(len(papers)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            shared = ({normalize_author(a) for a in papers[i].authors} & {normalize_author(a) for a in papers[j].authors}) - target
            if shared:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(papers)):
        clusters[find(index)].append(index)
    ordered = sorted(clusters.values(), key=len, reverse=True)
    ambiguous = len(ordered) > 1 and len(ordered[1]) * 2 >= len(ordered[0])
    return [papers[index] for index in ordered[0]], ambiguous


@dataclass
class PaperReport:
    attached: int = 0
    empty: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    per_advisor: dict[str, int] = field(default_factory=dict)


def attach_papers(
    advisors: list[Advisor],
    client: ArxivClient,
    *,
    year_from: int = 2021,
    max_per_advisor: int = 10,
) -> PaperReport:
    report = PaperReport()
    for advisor in advisors:
        result = client.search_author(advisor.name, year_from=year_from)
        if result.error:
            report.failed.append((advisor.name, result.error))
            continue
        cluster, ambiguous = select_cluster(result.papers, name_variants(advisor.name)[0])
        if not cluster:
            report.empty.append(advisor.name)
            continue
        if ambiguous:
            report.ambiguous.append(advisor.name)
            continue
        advisor.recent_papers = sorted(cluster, key=lambda paper: paper.year or 0, reverse=True)[
            :max_per_advisor
        ]
        report.attached += 1
        report.per_advisor[advisor.name] = len(advisor.recent_papers)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 arXiv 补全导师近期论文")
    parser.add_argument("--advisors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--year-from", type=int, default=2021)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="把启发式匹配结果写入导师表；默认只输出候选供人工复核",
    )
    args = parser.parse_args(argv)

    advisors = load_advisors(args.advisors)
    report = attach_papers(advisors, ArxivClient(), year_from=args.year_from)

    if args.apply:
        save_advisors(advisors, args.out)
    else:
        candidates = {
            advisor.name: [paper.model_dump() for paper in advisor.recent_papers]
            for advisor in advisors
            if advisor.recent_papers
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(candidates, ensure_ascii=False, indent=1), encoding="utf-8")
        print("warning: heuristic matches are NOT verified; review before --apply")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "attached": report.attached,
                    "per_advisor": report.per_advisor,
                    "empty": report.empty,
                    "ambiguous": report.ambiguous,
                    "failed": report.failed,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    print(f"attached: {report.attached}")
    print(f"empty: {len(report.empty)} ambiguous: {len(report.ambiguous)} failed: {len(report.failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
