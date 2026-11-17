"""模型调用的最小接口。

刻意保持 provider 无关：中期前先用 ``StubLLMClient`` 把流水线跑通，
选定模型后再实现 ``HTTPLLMClient``。选型需要注意两点，写在这里免得后面忘：
- 是否支持"不要把内容用于训练"的条款（隐私，幻灯片 4）；
- 是否接受单次调用不带任何学生身份信息（脱敏后调用）。
"""

from __future__ import annotations

from typing import Protocol

from advisor_match.config import settings


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


def get_client() -> LLMClient:
    """按配置返回客户端。未配置时返回 stub，保证离线也能跑测试。"""
    if settings.llm_provider == "stub":
        return StubLLMClient()
    raise NotImplementedError(
        "尚未实现真实客户端。实现时请确认：脱敏后调用、不持久化 prompt、不在日志里写学生材料。"
    )
