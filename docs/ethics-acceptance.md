# 伦理验收清单

来源：开题报告第 4、5 页。**验收看行为，不看提示词写得是否漂亮。**
下表每一项都能在代码或测试里找到对应位置；未完成的项标注了负责人。

## 三个重点风险（必须通过）

| # | 承诺 | 具体检验 | 当前落点 | 状态 |
| --- | --- | --- | --- | --- |
| F1 | 敏感属性不进入模型 | 仅改变姓名、性别、生源地，推荐结果（排序 + 各维度得分）完全一致 | `tests/test_fairness.py`；`StudentProfile` 无身份字段 | 结构层通过；行为层待 ranking |
| F2 | 不显示录取概率、不按历史录取排序 | 代码中不存在录取概率字段与相关提示词 | `schemas.py` 无该字段 | ✅ |
| F3 | 保留探索位 | 每档至少 1 位非主榜候选，且界面标注"探索位" | `ranking/tiers.py:assemble` | 待实现 |
| H1 | 事实台账 | 每条台账记录至少一条 `SourceRef`；引文必须能在原文中找到 | `schemas.LedgerRecord`、`ledger/build.quote_is_grounded` | 结构层通过 |
| H2 | 无据陈述占比为 0 | 交付草稿中无法解析到台账的句子数为 0 | `email/validator.validate_draft` | ✅ 有测试 |
| H3 | 学生先自写主张 | 草稿生成接口要求传入 `student_points`，对不台账的句子直接丢弃 | `email/draft.build_draft` | ✅ 有测试 |
| S1 | 20 条注入样本越权为 0 | 全部识别 + 无发信 / 无敏感读取能力 | `tests/test_injection.py` | ✅ 20/20 |
| S2 | 收件人校验拦截 100% | 导师姓名、邮箱、正文称呼三重比对，任一不符即阻塞 | `guards/recipient.check_recipient` | ✅ |
| S3 | 不自动发送 | 包内不存在 send/smtp；草稿无 sent 状态 | `tests/test_injection.py::test_no_send_capability_in_codebase` | ✅ |

## 六个维度中的其余各项

| 维度 | 承诺 | 落点 |
| --- | --- | --- |
| 目标设定 | 以证据支持率和学生自主判断次数为指标，不以发送量/回复率为目标 | `ValidationReport`、`ConfirmationRecord.student_edit_ratio` |
| 隐私 | 脱敏后调用、默认不留存、可一键删除 | `settings.persist_prompts=False`；`privacy/redact`（待实现） |
| 可解释性 | 每条理由绑定原文；主动列出不匹配之处 | `MatchPoint` / `Gap` 均要求 `SourceRef`，`AdvisorMatch` 无总分 |
| 安全可靠 | 抓取内容仅作数据；信息不足时停止 | `guards/injection.py`；`rerank_one` 证据不足返回 None |
| 责任 | 逐封人工确认并留痕；导师纠错通道；操作日志 | `email/confirm.py`；导师纠错通道待建 |

## 每次提交前自检

- [ ] 新增字段是否会让院校、性别、籍贯等影响排序？
- [ ] 新增生成路径是否都经过 `validate_draft`？
- [ ] 是否引入了任何发信、自动提交表单的能力？
- [ ] `pytest` 是否全绿（尤其 `test_fairness.py` 与 `test_injection.py`）？

## 中期（12 月 10 日）前必须补上

1. `test_counterfactual_recommendation_is_identical` 从 skip 变为真实测试；
2. `privacy/redact.py` 脱敏器与对应测试；
3. 30 个合成画像 + 博士生双盲标注的评估脚本；
4. 探索位规则与界面标注。
