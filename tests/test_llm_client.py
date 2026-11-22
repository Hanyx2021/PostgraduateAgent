"""OpenAI 兼容客户端测试（不访问网络）。"""

from __future__ import annotations

import json
import urllib.error

import pytest

from advisor_match.llm.client import LLMError, OpenAICompatibleClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_complete_parses_response_and_sends_expected_payload(monkeypatch) -> None:
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data)
        return FakeResponse({"choices": [{"message": {"content": "你好"}}]})

    monkeypatch.setattr("advisor_match.llm.client.urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(base_url="https://api.example.com/", api_key="k-1", model="m-1")

    assert client.complete(system="系统", user="用户") == "你好"
    assert captured["url"] == "https://api.example.com/chat/completions"
    assert captured["auth"] == "Bearer k-1"
    assert captured["body"]["model"] == "m-1"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "系统"}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "用户"}


def test_http_error_becomes_llm_error(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError("https://api.example.com", 400, "Bad Request", {}, None)

    monkeypatch.setattr("advisor_match.llm.client.urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(base_url="https://api.example.com", api_key="k", model="m")
    with pytest.raises(LLMError):
        client.complete(system="s", user="u")


def test_empty_response_becomes_llm_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "advisor_match.llm.client.urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"choices": [{"message": {"content": " "}}]}),
    )
    client = OpenAICompatibleClient(base_url="https://api.example.com", api_key="k", model="m")
    with pytest.raises(LLMError):
        client.complete(system="s", user="u")
