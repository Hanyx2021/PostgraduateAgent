"""OpenAlex 补全测试：作者身份判定与论文转换（不访问网络）。"""

from __future__ import annotations

from advisor_match.ingest.openalex import (
    AuthorCandidate,
    attach_papers,
    resolve_author,
    work_to_paper,
)
from advisor_match.schemas import Advisor, Paper


def candidate(name: str, works: int, *, tsinghua: bool = True, author_id: str = "A1") -> AuthorCandidate:
    return AuthorCandidate(
        author_id=author_id,
        display_name=name,
        works_count=works,
        cited_by_count=works * 10,
        institutions=["Tsinghua University"] if tsinghua else ["Some University"],
    )


class FakeClient:
    def __init__(self, candidates: dict[str, list[AuthorCandidate]], works: dict[str, list[Paper]]) -> None:
        self.candidates = candidates
        self.works = works

    def search_authors(self, name: str, *, per_page: int = 10) -> list[AuthorCandidate]:
        return self.candidates.get(name, [])

    def works_by_author(self, author_id: str, *, year_from: int, limit: int) -> list[Paper]:
        return self.works.get(author_id, [])[:limit]


def test_unique_tsinghua_candidate() -> None:
    client = FakeClient({"Jie Tang": [candidate("Jie Tang", 500), candidate("Jie Tang", 20, tsinghua=False)]}, {})
    resolution = resolve_author(Advisor(name="唐杰"), client)  # type: ignore[arg-type]
    assert resolution.status == "unique"
    assert resolution.author is not None and resolution.author.works_count == 500


def test_dominant_candidate_is_picked() -> None:
    client = FakeClient(
        {"Jie Tang": [candidate("Jie Tang", 500, author_id="A1"), candidate("Jie Tang", 50, author_id="A2")]},
        {},
    )
    resolution = resolve_author(Advisor(name="唐杰"), client)  # type: ignore[arg-type]
    assert resolution.status == "picked"
    assert resolution.author.author_id == "A1"


def test_close_candidates_are_ambiguous() -> None:
    client = FakeClient(
        {"Jie Tang": [candidate("Jie Tang", 120, author_id="A1"), candidate("Jie Tang", 100, author_id="A2")]},
        {},
    )
    resolution = resolve_author(Advisor(name="唐杰"), client)  # type: ignore[arg-type]
    assert resolution.status == "ambiguous"
    assert resolution.author is None


def test_no_tsinghua_candidate() -> None:
    client = FakeClient({"Jie Tang": [candidate("Jie Tang", 500, tsinghua=False)]}, {})
    assert resolve_author(Advisor(name="唐杰"), client).status == "none"


def test_work_to_paper_reconstructs_abstract() -> None:
    work = {
        "title": "A Paper",
        "publication_year": 2024,
        "doi": "https://doi.org/10.1/x",
        "authorships": [{"author": {"display_name": "Jie Tang"}}, {"author": {"display_name": "Alice"}}],
        "primary_location": {"source": {"display_name": "ACL"}},
        "abstract_inverted_index": {"retrieval": [0], "is": [1], "useful": [2]},
    }
    paper = work_to_paper(work)
    assert paper is not None
    assert paper.venue == "ACL" and paper.year == 2024
    assert paper.authors == ["Jie Tang", "Alice"]
    assert paper.abstract == "retrieval is useful"
    assert paper.url.startswith("https://doi.org")


def test_attach_papers_only_for_resolved_authors() -> None:
    advisors = [Advisor(name="唐杰"), Advisor(name="王五")]
    client = FakeClient(
        {
            "Jie Tang": [candidate("Jie Tang", 500, author_id="A1")],
            "Wu Wang": [candidate("Wu Wang", 80), candidate("Wu Wang", 70, author_id="A2")],
        },
        {"A1": [Paper(title="Paper One", year=2024), Paper(title="Paper Two", year=2023)]},
    )
    report = attach_papers(advisors, client)  # type: ignore[arg-type]

    assert advisors[0].recent_papers and len(advisors[0].recent_papers) == 2
    assert advisors[1].recent_papers == []
    assert report.attached == 1
    assert report.ambiguous == ["王五"]
