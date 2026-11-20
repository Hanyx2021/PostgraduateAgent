"""用 OpenAlex 补全导师论文：先确认作者身份，再取论文。

arXiv 的教训是"有名无机构"，所以这里的主键是 **OpenAlex 作者 ID**：

1. 按拼音名检索候选作者；
2. 只保留 display_name 精确匹配、且机构含 Tsinghua 的候选；
3. 唯一候选直接采用；多条记录时，只有第一名作品数是第二名 3 倍以上才采用；
4. 其余（歧义、查无此人）进人工清单，不写论文。

key 只从本地 ``.env`` 读取（``AM_OPENALEX_KEY``），不入库。
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from advisor_match.config import settings
from advisor_match.ingest.advisors import load_advisors, save_advisors
from advisor_match.ingest.papers import name_variants
from advisor_match.schemas import Advisor, Paper

API_BASE = "https://api.openalex.org"
REQUEST_INTERVAL = 0.2
MAX_RETRIES = 3
DOMINANCE_RATIO = 3

#: OpenAlex 同名记录常把不同人的论文合并；只收计算机科学领域的作品，
#: 宁可少收，也不要把化学/医学/交通的论文挂到计算机系导师名下。
RELEVANT_FIELDS = {"Computer Science"}


def normalize_name(name: str) -> str:
    """去掉空格、点、各类连字符（含 Unicode U+2010）与下划线。"""
    return re.sub(r"[\W_]+", "", name).lower()


@dataclass(frozen=True)
class AuthorCandidate:
    author_id: str
    display_name: str
    works_count: int
    cited_by_count: int
    institutions: list[str]

    @property
    def is_tsinghua(self) -> bool:
        return any("tsinghua" in item.lower() for item in self.institutions)


@dataclass
class Resolution:
    status: str  # unique | picked | ambiguous | none
    author: AuthorCandidate | None = None
    candidates: list[AuthorCandidate] = field(default_factory=list)


class OpenAlexClient:
    def __init__(self, api_key: str, *, timeout: int = 40) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._last_request_at = 0.0

    def _get(self, path: str, params: dict[str, str | int]) -> dict:
        query = urllib.parse.urlencode({**params, "api_key": self.api_key})
        url = f"{API_BASE}{path}?{query}"
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            wait = REQUEST_INTERVAL - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "AdvisorMatch/0.1"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except Exception as error:  # noqa: BLE001 - 网络错误统一退避
                last_error = error
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"OpenAlex 请求失败：{last_error}")

    def search_authors(self, name: str, *, per_page: int = 10) -> list[AuthorCandidate]:
        payload = self._get(
            "/authors", {"search": name, "per-page": per_page}
        )
        candidates: list[AuthorCandidate] = []
        for item in payload.get("results", []):
            institutions = [
                entry.get("display_name") or ""
                for entry in (item.get("last_known_institutions") or [])
            ]
            candidates.append(
                AuthorCandidate(
                    author_id=item["id"].rsplit("/", 1)[-1],
                    display_name=item.get("display_name") or "",
                    works_count=int(item.get("works_count") or 0),
                    cited_by_count=int(item.get("cited_by_count") or 0),
                    institutions=institutions,
                )
            )
        return candidates

    def works_by_author(
        self, author_id: str, *, year_from: int, limit: int
    ) -> list[Paper]:
        payload = self._get(
            "/works",
            {
                "filter": f"author.id:{author_id},from_publication_date:{year_from}-01-01",
                "per-page": min(max(limit * 3, 10), 100),
                "sort": "cited_by_count:desc",
            },
        )
        papers: list[Paper] = []
        for item in payload.get("results", []):
            if is_noise_work(item):
                continue
            paper = work_to_paper(item)
            if paper is not None:
                papers.append(paper)
            if len(papers) >= limit:
                break
        return papers


def is_noise_work(work: dict) -> bool:
    """同名作者记录混入的其他学科作品。"""
    field = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name")
    return field not in RELEVANT_FIELDS


def _abstract_from_inverted(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for word, indexes in inverted.items():
        for position in indexes:
            positions[position] = word
    text = " ".join(positions[index] for index in sorted(positions))
    return text[:2000] or None


def work_to_paper(work: dict) -> Paper | None:
    title = (work.get("title") or work.get("display_name") or "").strip()
    if not title:
        return None
    location = (work.get("primary_location") or {}).get("source") or {}
    authors = [
        (authorship.get("author") or {}).get("display_name") or ""
        for authorship in (work.get("authorships") or [])
    ]
    return Paper(
        title=title,
        venue=location.get("display_name"),
        year=work.get("publication_year"),
        url=work.get("doi") or work.get("id"),
        authors=[author for author in authors if author],
        abstract=_abstract_from_inverted(work.get("abstract_inverted_index")),
    )


def resolve_author(advisor: Advisor, client: OpenAlexClient) -> Resolution:
    variant_list = name_variants(advisor.name)
    variants = {normalize_name(item) for item in variant_list}
    primary = normalize_name(variant_list[0])
    reversed_only = variants - {primary}
    candidates = client.search_authors(variant_list[0])
    exact = [
        candidate
        for candidate in candidates
        if normalize_name(candidate.display_name) in variants and candidate.is_tsinghua
    ]

    if not exact:
        return Resolution(status="none", candidates=candidates)

    # 只靠"名 姓"倒序写法命中的，不能自动认定：CHEN TING 会匹配到同名的另一位
    # Ting Chen。这类一律交人工。
    primary_hits = [c for c in exact if normalize_name(c.display_name) == primary]
    if not primary_hits and any(
        normalize_name(c.display_name) in reversed_only for c in exact
    ):
        return Resolution(status="ambiguous", candidates=exact)

    if len(exact) == 1:
        return Resolution(status="unique", author=exact[0], candidates=exact)

    ordered = sorted(exact, key=lambda item: item.works_count, reverse=True)
    if ordered[0].works_count >= DOMINANCE_RATIO * max(1, ordered[1].works_count):
        return Resolution(status="picked", author=ordered[0], candidates=ordered)
    return Resolution(status="ambiguous", candidates=ordered)


@dataclass
class OpenAlexReport:
    attached: int = 0
    unique: int = 0
    picked: int = 0
    manual: int = 0
    ambiguous: list[str] = field(default_factory=list)
    none: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    per_advisor: dict[str, int] = field(default_factory=dict)


def attach_papers(
    advisors: list[Advisor],
    client: OpenAlexClient,
    *,
    year_from: int = 2021,
    max_per_advisor: int = 10,
    author_map: dict[str, str] | None = None,
    manual_papers: dict[str, list[Paper]] | None = None,
) -> OpenAlexReport:
    """``author_map`` 是人工核对后的 导师姓名 → OpenAlex 作者 ID。

    ``manual_papers`` 用于 OpenAlex 记录被合并、不可信的情况：直接用官方主页
    题录覆盖，跳过自动检索。
    """
    author_map = author_map or {}
    manual_papers = manual_papers or {}
    report = OpenAlexReport()
    for advisor in advisors:
        if advisor.name in manual_papers:
            advisor.recent_papers = manual_papers[advisor.name]
            report.attached += 1
            report.manual += 1
            report.per_advisor[advisor.name] = len(advisor.recent_papers)
            continue
        if advisor.name in author_map:
            author_id = author_map[advisor.name]
            status = "manual"
        else:
            try:
                resolution = resolve_author(advisor, client)
            except RuntimeError as error:
                report.failed.append((advisor.name, str(error)))
                continue

            if resolution.status == "ambiguous":
                # 本次判定不可信：清掉可能存在的旧结果，避免错误论文留在表里。
                advisor.recent_papers = []
                report.ambiguous.append(advisor.name)
                continue
            if resolution.status == "none" or resolution.author is None:
                advisor.recent_papers = []
                report.none.append(advisor.name)
                continue
            author_id = resolution.author.author_id
            status = resolution.status

        try:
            papers = client.works_by_author(author_id, year_from=year_from, limit=max_per_advisor)
        except RuntimeError as error:
            report.failed.append((advisor.name, str(error)))
            continue

        advisor.recent_papers = papers
        report.attached += 1
        report.per_advisor[advisor.name] = len(papers)
        if status == "unique":
            report.unique += 1
        elif status == "picked":
            report.picked += 1
        else:
            report.manual += 1
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="用 OpenAlex 补全导师论文")
    parser.add_argument("--advisors", type=Path, default=Path("data/processed/advisors.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/advisors.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/openalex_report.json"))
    parser.add_argument("--year-from", type=int, default=2021)
    parser.add_argument("--max-per", type=int, default=10)
    parser.add_argument(
        "--author-map",
        type=Path,
        default=None,
        help="人工核对后的 JSON：导师姓名 → OpenAlex 作者 ID",
    )
    parser.add_argument(
        "--manual-papers",
        type=Path,
        default=None,
        help="官方主页题录 JSON：导师姓名 → [{title, year, venue, authors}]",
    )
    args = parser.parse_args(argv)

    if not settings.openalex_key:
        parser.error("缺少 AM_OPENALEX_KEY，请在本地 .env 中配置")

    author_map = (
        json.loads(args.author_map.read_text(encoding="utf-8")) if args.author_map else None
    )
    manual_papers = None
    if args.manual_papers:
        raw = json.loads(args.manual_papers.read_text(encoding="utf-8"))
        manual_papers = {
            name: [Paper(**item) for item in items] for name, items in raw.items()
        }
    advisors = load_advisors(args.advisors)
    report = attach_papers(
        advisors,
        OpenAlexClient(settings.openalex_key),
        year_from=args.year_from,
        max_per_advisor=args.max_per,
        author_map=author_map,
        manual_papers=manual_papers,
    )
    save_advisors(advisors, args.out)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["with_papers"] = sum(1 for advisor in advisors if advisor.recent_papers)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"attached: {report.attached}")
    print(f"ambiguous: {len(report.ambiguous)}  none: {len(report.none)}  failed: {len(report.failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuthorCandidate",
    "OpenAlexClient",
    "OpenAlexReport",
    "Resolution",
    "attach_papers",
    "is_noise_work",
    "resolve_author",
    "work_to_paper",
]
