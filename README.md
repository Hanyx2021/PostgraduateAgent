# 推研导师匹配与自荐邮件 Agent

计算机工程伦理与职业素养课程项目。目标是让推研匹配的依据**从人脉与话术转向可核验的研究契合度**：
用公开、可核验的信息完成导师匹配，用学生本人的材料生成自荐邮件，判断依据由学生本人逐句核对。

- 组员：韩宇轩（数据与检索）、陈英豪（Agent 与产品）、杨骐岗（伦理与用户研究）
- 当前阶段：**v0.2**（导师库、召回、重排、分档已可运行；论文补全待定来源）

## 项目不是做什么

- 不自动发信，也不提供发信接口；
- 不输出录取概率，不按历史录取数据排序；
- 不采集姓名、性别、院校、籍贯等敏感属性；
- 不生成没有材料依据的事实陈述。

这四条由类型、数据表与权限保证，不靠提示词提醒。见 [docs/ethics-acceptance.md](docs/ethics-acceptance.md)。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -e ".[dev]"

# 可选：在 .env 配置 AM_LLM_* 后，界面可勾选“使用模型改写邮件”
pytest                        # 运行测试
streamlit run app/streamlit_app.py

# 可选：启用向量召回（本地模型，约 95MB，不入库）
pip install -e ".[embeddings]"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='data/models/bge-small-zh-v1.5')"

# 可选：用 OpenAlex 补导师论文（需在本地 .env 配置 AM_OPENALEX_KEY）
python -m advisor_match.ingest.openalex --advisors data/processed/advisors.jsonl
# 自动消歧失败时，生成人工核对清单
python -m advisor_match.ingest.review

# 一键重建：抓目录 → 抓名录 → 抓详情页 → OpenAlex → 落盘
python -m advisor_match.ingest.rebuild --catalog-url "<2027 招生目录 URL>"

# 命令行跑一次匹配（需要本地 data/processed/advisors.jsonl）
python -m advisor_match.pipeline --target "信息检索,多模态学习" \
  --statement "完成多模态检索增强生成的课程项目" --top 8

# 跑评估：30 个合成画像，输出 Recall@10 / nDCG@10
python -m advisor_match.evaluation --advisors data/processed/advisors.jsonl
# 规则版 vs 模型版重排对比（需要 AM_LLM_API_KEY）
python -m advisor_match.evaluation --advisors data/processed/advisors.jsonl   --rerank-compare --subset 10
```

## 五步流水线

```
简历/论文 ──► ① 材料结构化 ──► 成果台账（每条记录可回溯原文）
                                   │
系主页/DBLP ─► ② 候选召回（30 位）──► ③ 证据化重排（保留 8 位）
                                   │
                                   ▼
                         ④ 分档呈现（三档，无总分）
                                   │
                                   ▼
                         ⑤ 邮件草稿 + 逐句校验 → 学生确认 → 手动发送
```

第 ④ ⑤ 步由学生决定，系统只提供依据。实现映射见 [docs/architecture.md](docs/architecture.md)。

## 目录

| 路径 | 内容 | 负责人 |
| --- | --- | --- |
| [src/advisor_match/schemas.py](src/advisor_match/schemas.py) | 全部核心数据结构（台账、匹配、草稿） | 共用 |
| [src/advisor_match/ingest/](src/advisor_match/ingest/) | PDF / 简历 / 导师库抓取与解析 | 韩宇轩 |
| [src/advisor_match/retrieval/](src/advisor_match/retrieval/) | BM25 + 向量混合召回 | 韩宇轩 |
| [src/advisor_match/ranking/](src/advisor_match/ranking/) | 证据化重排、三档组装 | 韩宇轩 |
| [src/advisor_match/ledger/](src/advisor_match/ledger/) | 台账构建与存储（防编造的核心） | 韩宇轩 |
| [src/advisor_match/email/](src/advisor_match/email/) | 草稿生成、claim–evidence 校验、逐句确认 | 陈英豪 |
| [src/advisor_match/llm/](src/advisor_match/llm/) | 模型客户端与提示词 | 陈英豪 |
| [src/advisor_match/guards/](src/advisor_match/guards/) | 敏感字段、注入、收件人三重防护 | 杨骐岗 |
| [tests/](tests/) | 公平 / 诚实 / 安全三类验收测试 | 杨骐岗 + 各模块作者 |
| [app/streamlit_app.py](app/streamlit_app.py) | 界面 | 陈英豪 |
| [docs/](docs/) | 架构、伦理验收、开题材料 | 杨骐岗 |

## 开发约定

1. **先改数据结构，再写功能。** 任何涉及"记录 / 判断 / 句子"的改动先动 `schemas.py`，让测试先失败再实现。
2. **不要绕过校验器。** 生成的内容必须经过 `validator.validate_draft` 才能进入界面。
3. **新增字段前自问**：这个字段会不会让院校、性别、籍贯影响排序？会的话不要加进 `StudentProfile`。
4. **提交信息**用英文一句话说明改了什么，前缀用 `feat/fix/test/docs/refactor`。
5. 每个模块的 `NotImplementedError` 里写了负责人和实现要点，接手的同学从那里开始。
