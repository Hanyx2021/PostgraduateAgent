"""模型响应的本地缓存。

只缓存返回文本，不落盘 prompt；key 是 prompt 的哈希。缓存目录在
``data/cache/``（已 gitignore），可以随时整个删掉。

用途主要是重排：同一份画像对同一位导师的重复调用很常见，缓存后二次运行
几乎是瞬时的。默认 TTL 7 天，避免用过期判断。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from advisor_match.config import REPO_ROOT, settings
from advisor_match.llm.client import LLMClient


class CachedLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        *,
        cache_dir: Path | None = None,
        ttl_days: int | None = None,
    ) -> None:
        self.inner = inner
        self.cache_dir = cache_dir or (REPO_ROOT / settings.llm_cache_dir)
        self.ttl_seconds = (ttl_days if ttl_days is not None else settings.llm_cache_ttl_days) * 86400

    def _path(self, system: str, user: str, temperature: float) -> Path:
        model = getattr(self.inner, "model", "unknown")
        digest = hashlib.sha256(
            f"{model}|{temperature}|{system}|{user}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        path = self._path(system, user, temperature)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if time.time() - float(payload["created_at"]) <= self.ttl_seconds:
                    return str(payload["text"])
            except (KeyError, ValueError, OSError):
                pass  # 缓存损坏就当未命中

        text = self.inner.complete(system=system, user=user, temperature=temperature)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"created_at": time.time(), "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )
        return text


def clear_cache(cache_dir: Path | None = None) -> int:
    target = cache_dir or (REPO_ROOT / settings.llm_cache_dir)
    if not target.exists():
        return 0
    removed = 0
    for path in target.glob("*.json"):
        path.unlink()
        removed += 1
    return removed


__all__ = ["CachedLLMClient", "clear_cache"]
