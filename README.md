# 推研导师匹配与自荐邮件 Agent

## 快速开始

### 1. 环境

```bash
cd d:/Learning/2026Autumn/ethics
python -m venv .venv
.venv/Scripts/activate          # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -e ".[dev,embeddings]"
```

### 2. 本地配置

复制 `.env.example` 为 `.env`，填写需要的 key（`.env` 已 gitignore，不会入库）：

```dotenv
# 导师论文（可选）
AM_OPENALEX_KEY=

# 模型（可选；不配置时邮件用模板、重排用规则版）
AM_LLM_PROVIDER=deepseek
AM_LLM_BASE_URL=https://api.deepseek.com
AM_LLM_MODEL=deepseek-flash
AM_LLM_API_KEY=
```

### 3. 向量模型（可选，约 95MB）

```bash
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='data/models/bge-small-zh-v1.5')"
```

没有模型时召回自动退回 BM25，主流程不受影响。

### 4. 启动界面

```bash
streamlit run app/streamlit_app.py
```

如果提示找不到 `advisor_match`，先执行 `pip install -e .`，或临时用：

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## 界面使用

左侧边栏从上到下：

1. **数据**：导师主表路径（默认 `data/processed/advisors.jsonl`）
2. **开关**：
   - 使用向量召回（本地 BGE）
   - 使用模型改写邮件（DeepSeek）
   - 上传后由模型整理台账
   - 使用模型重排（混合模式，仅对前 10 位调用模型）
3. **上传简历**（必填，PDF / txt / md）
   - 解析后由模型整合成台账；姓名、电话、邮箱会先移除再发送
   - 自动推断并填写“目标方向”，自动填写“署名”，都可手动修改
4. **成果陈述**：上传后自动填入，也可以手动补充
5. 点 **开始匹配**

页面顶部有三个分区：

- **① 材料与台账**：每条陈述及其原文出处
- **② 导师匹配**：三档结果、每条理由的依据、招生资格与信息截止日期；“其他召回候选”单独列出但不生成草稿；可下载匹配结果 JSON
- **③ 邮件草稿**：
  1. 选择导师（多导师草稿分别保存，切换不丢）
  2. 生成四段式草稿（模型失败自动退回模板）
  3. 每句可以修改或勾选“删除”，点 **应用修改并重新校验**
  4. 逐句勾选 **已核对**
  5. 点 **完成确认并生成文本**：写入本地留痕，给出可复制文本
  6. 可下载草稿 txt、确认记录 JSON

> 系统不会发送邮件。最后一步由学生本人复制到自己的邮件客户端。

## 数据准备

导师主表与抓取结果都在本地，**不进仓库**：

```
data/processed/advisors.jsonl     导师主表
data/processed/profiles.json      详情页解析结果
data/processed/overrides.json     人工补录
data/processed/aliases.json       姓名别名
data/processed/openalex_authors.json  人工核对的作者 ID
data/processed/manual_papers.json     官方主页题录
data/raw/                         原始 HTML 快照
data/models/                      本地向量模型
data/cache/llm/                   模型响应缓存
```

一键重建导师库：

```bash
python -m advisor_match.ingest.rebuild --catalog-url "<2027 招生目录 URL>"
```

其他数据命令：

```bash
# 补导师论文（需要 AM_OPENALEX_KEY）
python -m advisor_match.ingest.openalex --advisors data/processed/advisors.jsonl

# 自动消歧失败时，生成人工核对清单
python -m advisor_match.ingest.review

# 人工指定作者 ID / 官方题录
python -m advisor_match.ingest.openalex \
  --advisors data/processed/advisors.jsonl \
  --author-map data/processed/openalex_authors.json \
  --manual-papers data/processed/manual_papers.json
```

## 命令行

```bash
# 跑一次匹配
python -m advisor_match.pipeline --target "信息检索,多模态学习" \
  --statement "完成多模态检索增强生成的课程项目" --top 8

# 评估：30 个合成画像，Recall@10 / nDCG@10
python -m advisor_match.evaluation --advisors data/processed/advisors.jsonl

# 规则版 vs 模型版 vs 混合版重排对比（需要 AM_LLM_API_KEY）
python -m advisor_match.evaluation --advisors data/processed/advisors.jsonl \
  --rerank-compare --subset 10
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

第 ④ ⑤ 步由学生决定，系统只提供依据。实现细节见 [docs/architecture.md](docs/architecture.md)。

## 评估

30 个合成画像（2027 目录方向 → 在列导师），k=10：

| 召回方式 | Recall@10 | nDCG@10 |
| --- | ---: | ---: |
| 混合召回（BM25 0.3 + 向量 0.7） | **0.462** | **0.462** |
| 纯向量 | 0.443 | 0.432 |
| 纯 BM25 | 0.434 | 0.387 |
| 主表顺序基线 | 0.141 | 0.109 |

重排策略对比（10 个画像，最终推荐列表）：

| 策略 | Recall@10 | nDCG@10 |
| --- | ---: | ---: |
| 规则版 | **0.301** | 0.311 |
| 模型版 | 0.281 | **0.326** |
| 混合版 | 0.300 | 0.320 |

人工标注（10 个画像、150 位候选，单标注者）：

| 策略 | Recall@10 | nDCG@10 | 覆盖率 |
| --- | ---: | ---: | ---: |
| 召回排序 | 0.782 | **0.795** | 0.936 |
| 规则重排 | 0.773 | 0.730 | 0.908 |
| 模型重排 | 0.750 | 0.679 | 0.835 |
| 混合重排 | **0.781** | 0.691 | 0.900 |

候选池按召回抽样，召回基线在这份评估里天然占优；重排策略之间可直接比较。
保留“其他召回候选”后，重排不再大量丢失人工判定相关的导师。

目录真值来自 2027 招生目录、画像是模板生成；换数据或换模型后应重新校准。

## 目录

| 路径 | 内容 | 负责人 |
| --- | --- | --- |
| [src/advisor_match/schemas.py](src/advisor_match/schemas.py) | 核心数据结构（台账、匹配、草稿） | 共用 |
| [src/advisor_match/ingest/](src/advisor_match/ingest/) | 导师库抓取、解析、重建 | 韩宇轩 |
| [src/advisor_match/retrieval/](src/advisor_match/retrieval/) | BM25 + 向量混合召回 | 韩宇轩 |
| [src/advisor_match/ranking/](src/advisor_match/ranking/) | 证据化重排、三档组装 | 韩宇轩 |
| [src/advisor_match/ledger/](src/advisor_match/ledger/) | 台账构建与存储 | 韩宇轩 |
| [src/advisor_match/email/](src/advisor_match/email/) | 草稿生成、校验、逐句确认 | 陈英豪 |
| [src/advisor_match/llm/](src/advisor_match/llm/) | 模型客户端、缓存、提示词 | 陈英豪 |
| [src/advisor_match/product/](src/advisor_match/product/) | 材料解析、模型整理、导出 | 陈英豪 |
| [src/advisor_match/guards/](src/advisor_match/guards/) | 敏感字段、注入、收件人防护 | 杨骐岗 |
| [src/advisor_match/evaluation.py](src/advisor_match/evaluation.py) | 评估集与指标 | 韩宇轩 |
| [app/streamlit_app.py](app/streamlit_app.py) | 界面 | 陈英豪 |
| [tests/](tests/) | 公平 / 诚实 / 安全等验收测试 | 全体 |

## 数据与隐私

- `.env`、`data/` 全部 gitignore，密钥与简历不会进入仓库
- 模型调用前移除姓名、电话、邮箱、身份证号；不持久化 prompt
- 模型响应缓存只存返回文本，目录 `data/cache/llm/`，可随时删除
- 确认日志 `data/processed/confirmations.jsonl` 只记行为（导师、时间、句数、台账 ID、修改比例），不记正文

## 测试

```bash
pytest
python -m ruff check app src --select E9,F
```

覆盖：台账证据绑定、无据陈述丢弃、收件人核对、提示注入 20 条样本、反事实一致性、模型抽取降级、界面冒烟测试。
