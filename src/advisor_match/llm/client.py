"""模型调用的最小接口。

实现是 OpenAI 兼容的 ``/chat/completions``（当前接 DeepSeek），只用标准库，
不引入额外 SDK。约定：
- 不记录 prompt 内容与密钥；异常信息里也不带请求体；
- 没有配置密钥时返回 ``StubLLMClient``，离线流程不受影响；
- 调用方必须能承受客户端抛错（邮件生成会退回模板，重排会退回规则版）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from advisor_match.config import settings


class LLMError(RuntimeError):
    """模型调用失败。调用方应降级，而不是把异常抛给用户。"""


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        """返回模型文本。实现方不得记录 prompt 内容（除非 settings.persist_prompts）。"""
        ...


class StubLLMClient:
    """占位实现：返回固定文本，便于在没有密钥时开发界面与校验逻辑。"""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return self.response


class OpenAICompatibleClient:
    """OpenAI 兼容接口。默认指向 DeepSeek。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 90,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise LLMError(f"模型接口返回 HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise LLMError(f"模型调用失败：{type(error).__name__}") from error

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("模型响应缺少 choices[0].message.content") from error
        if not content or not str(content).strip():
            raise LLMError("模型返回空内容")
        return str(content)


def get_client(*, use_cache: bool = True) -> LLMClient:
    """按配置返回客户端。未配置密钥时返回 stub，保证离线也能跑测试。"""
    if (
        settings.llm_provider != "stub"
        and settings.llm_api_key
        and settings.llm_base_url
        and settings.llm_model
    ):
        client: LLMClient = OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        if use_cache:
            from advisor_match.llm.cache import CachedLLMClient

            client = CachedLLMClient(client)
        return client
    return StubLLMClient()


__all__ = ["LLMClient", "LLMError", "OpenAICompatibleClient", "StubLLMClient", "get_client"]
