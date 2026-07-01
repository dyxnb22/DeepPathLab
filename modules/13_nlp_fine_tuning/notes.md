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
