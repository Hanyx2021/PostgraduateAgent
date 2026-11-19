"""论文补全测试：作者过滤与同名消歧（不访问网络）。"""

from __future__ import annotations

from advisor_match.ingest.papers import (
    ArxivClient,
    SearchResult,
    _filter_papers,
    attach_papers,
    name_variants,
    normalize_author,
    select_cluster,
)
from advisor_match.schemas import Advisor, Paper

ATOM_HEAD = (
    '<feed xmlns="http://www.w3.org/2005/Atom" '
    'xmlns:arxiv="http://arxiv.org/schemas/atom">'
)


def entry(*, author: str, coauthors: list[str], year: str, category: str, title: str) -> str:
    authors = "".join(f"<author><name>{name}</name></author>" for name in [author, *coauthors])
    return (
        "<entry>"
        f"<id>http://arxiv.org/abs/{abs(hash(title))}</id>"
        f"<published>{year}-01-01T00:00:00Z</published>"
        f"<title>{title}</title><summary>abstract</summary>"
        f"{authors}<arxiv:primary_category term='{category}'/>"
        "</entry>"
    )


def paper(title: str, year: int, coauthors: list[str], author: str = "Jie Tang") -> Paper:
    return Paper(title=title, year=year, authors=[author, *coauthors])


def test_name_variants_for_chinese_and_latin() -> None:
    assert name_variants("唐杰") == ["Jie Tang", "Tang Jie"]
    assert name_variants("郑方") == ["Fang Zheng", "Zheng Fang"]
    # 双字名是一个整体，不能拆成 Shi Min Hu
    assert name_variants("胡事民") == ["Shimin Hu", "Hu Shimin"]
    assert name_variants("刘知远") == ["Zhiyuan Liu", "Liu Zhiyuan"]
    assert name_variants("CANNISTRACI CARLO VITTORIO") == ["CANNISTRACI CARLO VITTORIO"]
    # 多音字姓氏：翟作姓氏读 Zhái
    assert name_variants("翟季冬") == ["Jidong Zhai", "Zhai Jidong"]
    # 目录中的大写外籍名补倒序写法
    assert name_variants("CHEN TING") == ["CHEN TING", "Ting Chen"]


def test_normalize_author() -> None:
    assert normalize_author("Jie Tang") == normalize_author("jie.tang")
    assert normalize_author("Jie-Tang") == "jietang"


def test_filter_keeps_only_matching_author_recent_cs_papers() -> None:
    payload = (
        ATOM_HEAD
        + entry(author="Jie Tang", coauthors=["Zhiyuan Liu"], year="2025", category="cs.CL", title="Good")
        + entry(author="Jie Tang", coauthors=["Zhiyuan Liu"], year="2025", category="eess.SP", title="Wrong field")
        + entry(author="Jie Tang", coauthors=["Zhiyuan Liu"], year="2015", category="cs.CL", title="Too old")
        + entry(author="Wei Tang", coauthors=[], year="2025", category="cs.CL", title="Other person")
        + "</feed>"
    ).encode("utf-8")

    papers = _filter_papers(payload, {normalize_author("Jie Tang")}, year_from=2021)
    assert [item.title for item in papers] == ["Good"]
    assert papers[0].venue == "arXiv:cs.CL"


def test_cluster_prefers_the_larger_coauthor_group() -> None:
    papers = [
        paper("A1", 2025, ["Zhiyuan Liu", "Maosong Sun"]),
        paper("A2", 2024, ["Zhiyuan Liu"]),
        paper("A3", 2023, ["Maosong Sun"]),
        paper("B1", 2025, ["Someone Else"]),
    ]
    cluster, ambiguous = select_cluster(papers, "Jie Tang")
    assert {item.title for item in cluster} == {"A1", "A2", "A3"}
    assert ambiguous is False


def test_cluster_flags_two_active_identities() -> None:
    papers = [
        paper("A1", 2025, ["Zhiyuan Liu"]),
        paper("A2", 2024, ["Zhiyuan Liu"]),
        paper("B1", 2025, ["Someone Else"]),
        paper("B2", 2024, ["Someone Else"]),
    ]
    _, ambiguous = select_cluster(papers, "Jie Tang")
    assert ambiguous is True


class FakeClient:
    def __init__(self, results: dict[str, SearchResult]) -> None:
        self.results = results

    def search_author(self, name: str, *, year_from: int = 2021) -> SearchResult:
        return self.results.get(name, SearchResult())


def test_attach_papers_records_empty_and_failed() -> None:
    advisors = [Advisor(name="唐杰"), Advisor(name="王五"), Advisor(name="赵六")]
    client = FakeClient(
        {
            "唐杰": SearchResult(papers=[paper("A1", 2025, ["Zhiyuan Liu"]), paper("A2", 2024, ["Zhiyuan Liu"])]),
            "赵六": SearchResult(error="timeout"),
        }
    )
    report = attach_papers(advisors, client)  # type: ignore[arg-type]

    assert advisors[0].recent_papers
    assert report.per_advisor == {"唐杰": 2}
    assert report.empty == ["王五"]
    assert report.failed and report.failed[0][0] == "赵六"


def test_ambiguous_advisors_are_not_auto_filled() -> None:
    advisors = [Advisor(name="唐杰")]
    client = FakeClient(
        {
            "唐杰": SearchResult(
                papers=[
                    paper("A1", 2025, ["Zhiyuan Liu"]),
                    paper("A2", 2024, ["Zhiyuan Liu"]),
                    paper("B1", 2025, ["Other"]),
                    paper("B2", 2024, ["Other"]),
                ]
            )
        }
    )
    report = attach_papers(advisors, client)  # type: ignore[arg-type]
    assert report.ambiguous == ["唐杰"]
    assert advisors[0].recent_papers == []
