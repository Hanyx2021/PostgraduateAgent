"""集中配置：路径、检索参数、敏感字段禁入表。

参数集中在这里，是为了让"召回 30 → 保留 8"这类承诺在代码里只有一个来源，
评估脚本与产品界面读同一份配置。
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    """极简 .env 加载：只支持 KEY=VALUE，不覆盖已有环境变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"


class RetrievalConfig(BaseModel):
    """检索与重排的规模参数（对齐幻灯片 6 第 2、3 步）。"""

    recall_k: int = 30
    keep_k: int = 8
    #: 权重来自 30 个合成画像的网格搜索：0.3/0.7 的 nDCG 优于 0.5/0.5。
    #: 样本仍小，换数据后应重新跑评估确认。
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    #: 纯向量命中的相似度下限。BGE 对无关中文短句也常给 0.3 左右，不能设得太低。
    vector_min_similarity: float = 0.45
    #: 重排时"语义相近"的判断阈值，比召回更严：要能写进理由里。
    rerank_min_similarity: float = 0.50


class Settings(BaseModel):
    retrieval: RetrievalConfig = RetrievalConfig()
    #: 实验性探索位数量：每档至少保留 1 位"分数不高但值得看"的候选，
    #: 用来对抗排序的自我实现（幻灯片 5 "保留三个探索位"）。
    exploration_slots: int = 3
    #: 是否把模型调用内容写入本地日志。默认关闭（隐私：默认不留存）。
    persist_prompts: bool = False
    #: 本地向量模型目录，相对仓库根目录；不存在时召回退回 BM25。
    embedding_model: str | None = os.getenv(
        "AM_EMBEDDING_MODEL", "data/models/bge-small-zh-v1.5"
    )
    #: OpenAlex 免费 key，仅从本地 .env 读取，绝不入库。
    openalex_key: str | None = os.getenv("AM_OPENALEX_KEY")
    llm_provider: str = os.getenv("AM_LLM_PROVIDER", "stub")
    llm_base_url: str | None = os.getenv("AM_LLM_BASE_URL")
    llm_model: str | None = os.getenv("AM_LLM_MODEL")
    llm_api_key: str | None = os.getenv("AM_LLM_API_KEY")


settings = Settings()
