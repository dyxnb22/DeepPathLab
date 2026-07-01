# 笔记：NLP 微调

## 核心问题

冻结 vs 全量微调各适合什么场景？

## 微调策略

1. **Linear probe**：冻结预训练嵌入，只训练分类头。数据极少时防过拟合。
2. **Partial fine-tune**：解冻最后几层，保留底层通用特征。
3. **Full fine-tune**：全部参数更新。数据足够、任务与预训练差异大时更好。

## 学习率惯例

- 预训练层用**更小 lr**（如 1e-5 ~ 1e-4）
- 新加分类头用**更大 lr**（如 1e-3）

`FinetuneConfig` 在 `from_scratch/finetune_loop.py` 中把这两档 lr 分开配置。

## 与 Module 11/12 的关系

| 模块 | 角色 |
|------|------|
| Module 11 | skip-gram 预训练嵌入 |
| Module 12 | BoW / 嵌入分类基线 |
| Module 13 | 用预训练嵌入初始化，对比 scratch / frozen / full |

本模块连接预训练与下游任务：同一情感语料，不同微调策略。

## 本模块脚本

| 路径 | 作用 |
|------|------|
| `from_scratch/finetune_loop.py` | 微调配置与 epoch 循环工具 |
| `reproduce/finetune_sentiment.py` | PyTorch 微调复现 |
| `experiments/freeze_vs_finetune.py` | 三种策略对照实验 |

```bash
python3 modules/13_nlp_fine_tuning/experiments/freeze_vs_finetune.py
```

## 风险

- 小数据全量微调 → 灾难性遗忘
- 冻结过度 → 表达能力不足
- 验证集过小 → 指标波动大，需早停与更大验证集

## 常见踩坑

1. **词表不一致**：预训练嵌入行数必须与当前 `vocab` 对齐；OOV 词需随机初始化或 `<unk>` 行
2. **学习率不分层**：预训练层与分类头共用过大 lr 容易破坏已有表示
3. **忘记 `eval()` / `train()` 模式**：Dropout、BatchNorm 在验证阶段行为不同
4. **只看训练 loss**：小验证集上 val acc 抖动大，应以验证曲线 + 早停为准
5. **冻结后仍更新嵌入**：需显式 `requires_grad=False` 或优化器 param group 排除

## 自检问题

1. 数据极少时，你会选 linear probe、partial 还是 full fine-tune？理由是什么？
2. 预训练嵌入 lr 为什么通常比分类头小一个数量级？
3. `freeze_vs_finetune.py` 里三种策略的参数更新范围有何不同？
4. 什么是灾难性遗忘？本模块实验里哪种策略最容易出现？
5. 若下游任务与预训练语料领域差异很大，冻结策略可能失败的原因是什么？
