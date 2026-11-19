"""人工核对清单测试。"""

from __future__ import annotations

from datetime import date

from advisor_match.ingest.openalex import AuthorCandidate
from advisor_match.ingest.review import build_review, collect_candidates, render_markdown


def candidate(author_id: str, name: str, works: int, *, tsinghua: bool) -> AuthorCandidate:
    return AuthorCandidate(
        author_id=author_id,
        display_name=name,
        works_count=works,
        cited_by_count=works * 5,
        institutions=["Tsinghua University"] if tsinghua else ["Other University"],
    )


class FakeClient:
    def __init__(self, responses: dict[str, list[AuthorCandidate]]) -> None:
        self.responses = responses

    def search_authors(self, name: str, *, per_page: int = 10) -> list[AuthorCandidate]:
        return self.responses.get(name, [])


def test_collect_candidates_prefers_exact_tsinghua() -> None:
    client = FakeClient(
        {
            "Jie Tang": [
                candidate("A2", "Jie Tang", 500, tsinghua=False),
                candidate("A1", "Jie Tang", 120, tsinghua=True),
            ]
        }
    )
    candidates = collect_candidates("唐杰", client)  # type: ignore[arg-type]
    assert [item.author_id for item in candidates] == ["A1", "A2"]


def test_build_review_and_render() -> None:
    client = FakeClient({"Lin Gan": [candidate("A1", "Lin Gan", 163, tsinghua=True)]})
    entries = build_review([("甘霖", "ambiguous")], client)  # type: ignore[arg-type]
    text = render_markdown(entries, generated_at=date(2026, 11, 19))
    assert "甘霖" in text
    assert "A1" in text
    assert "author-map" in text
    assert entries[0].to_dict()["candidates"][0]["author_id"] == "A1"
