"""向量检索：本地中文语义模型 + 余弦相似度。

模型是 BAAI/bge-small-zh-v1.5（约 95MB），从 ModelScope 下载到 ``data/models/``，
不入库。没有模型时 ``load_embedding`` 返回 None，召回自动退回纯 BM25——
不能让"没下模型"变成"跑不起来"。

隐私：模型在本地运行，导师与学生文本不出机器；这是选本地模型而不是
embedding API 的主要原因。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from advisor_match.config import REPO_ROOT, settings


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """返回 L2 归一化后的向量矩阵，形状 (len(texts), dim)。"""
        ...


class BGEEmbedding:
    """BGE 中文小模型。首次加载约 2–3 秒，之后常驻内存。"""

    def __init__(self, model_path: Path) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_path), device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def resolve_model_path() -> Path:
    path = Path(settings.embedding_model or "")
    return path if path.is_absolute() else REPO_ROOT / path


def load_embedding() -> EmbeddingProvider | None:
    """模型不存在或缺依赖时返回 None，由调用方决定退回 BM25。"""
    path = resolve_model_path()
    if not path.exists():
        return None
    try:
        return BGEEmbedding(path)
    except Exception:  # noqa: BLE001 - 缺 torch/sentence-transformers 时不应阻塞主流程
        return None


def build_matrix(texts: list[str], embedding: EmbeddingProvider) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return embedding.encode(texts)


__all__ = [
    "BGEEmbedding",
    "EmbeddingProvider",
    "build_matrix",
    "load_embedding",
    "resolve_model_path",
]
