"""OpenAlex 补全测试：作者身份判定与论文转换（不访问网络）。"""

from __future__ import annotations

from advisor_match.ingest.openalex import (
    AuthorCandidate,
    attach_papers,
    is_noise_work,
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


def test_normalize_name_handles_unicode_hyphen() -> None:
    from advisor_match.ingest.openalex import normalize_name

    assert normalize_name("Shi‐Min Hu") == normalize_name("Shi-Min Hu")
    assert normalize_name("Jie Tang") == normalize_name("JieTang")


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


def test_noise_fields_are_filtered() -> None:
    chemistry = {"primary_topic": {"field": {"display_name": "Chemistry"}}}
    cs = {"primary_topic": {"field": {"display_name": "Computer Science"}}}
    assert is_noise_work(chemistry)
    assert not is_noise_work(cs)


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


def test_reversed_name_only_match_is_ambiguous() -> None:
    client = FakeClient({"CHEN TING": [candidate("Ting Chen", 100, author_id="A1")]}, {})
    resolution = resolve_author(Advisor(name="CHEN TING"), client)  # type: ignore[arg-type]
    assert resolution.status == "ambiguous"


def test_unresolved_advisors_have_stale_papers_cleared() -> None:
    advisors = [Advisor(name="王五", recent_papers=[Paper(title="old")])]
    client = FakeClient(
        {"Wu Wang": [candidate("Wu Wang", 80, author_id="A1"), candidate("Wu Wang", 70, author_id="A2")]}, {}
    )
    attach_papers(advisors, client)  # type: ignore[arg-type]
    assert advisors[0].recent_papers == []


def test_manual_author_map_bypasses_resolution() -> None:
    advisors = [Advisor(name="王五")]
    client = FakeClient({}, {"A9": [Paper(title="Paper", year=2024)]})
    report = attach_papers(advisors, client, author_map={"王五": "A9"})  # type: ignore[arg-type]
    assert advisors[0].recent_papers
    assert report.attached == 1
    assert report.manual == 1


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
