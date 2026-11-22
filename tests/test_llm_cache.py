"""模型响应缓存测试。"""

from __future__ import annotations

from pathlib import Path

from advisor_match.llm.cache import CachedLLMClient, clear_cache


class CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return f"response-{self.calls}"


def test_cache_hit_avoids_second_call(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedLLMClient(inner, cache_dir=tmp_path, ttl_days=7)

    assert client.complete(system="s", user="u") == "response-1"
    assert client.complete(system="s", user="u") == "response-1"
    assert inner.calls == 1


def test_different_prompts_are_separate_entries(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedLLMClient(inner, cache_dir=tmp_path, ttl_days=7)

    first = client.complete(system="s", user="u1")
    second = client.complete(system="s", user="u2")
    assert first != second
    assert inner.calls == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_expired_cache_is_refreshed(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedLLMClient(inner, cache_dir=tmp_path, ttl_days=0)
    client.complete(system="s", user="u")
    client.complete(system="s", user="u")
    assert inner.calls == 2


def test_clear_cache_removes_entries(tmp_path: Path) -> None:
    client = CachedLLMClient(CountingClient(), cache_dir=tmp_path, ttl_days=7)
    client.complete(system="s", user="u")
    assert clear_cache(tmp_path) == 1
    assert list(tmp_path.glob("*.json")) == []
