# 13 NLP 微调

Track E — 在预训练表示上适配下游任务。

## 核心问题

冻结 vs 全量微调各适合什么场景？

## 核心知识点

- **Linear probe**：冻结预训练嵌入，只训练分类头；小数据、防过拟合
- **Full fine-tune**：解冻全部参数；数据充足、任务与预训练差异大时更合适
- **分层学习率**：预训练层用小 lr，新层用大 lr
- **灾难性遗忘**：全量微调可能破坏通用语言知识
- **预训练初始化**：用 Module 11 skip-gram 嵌入对齐词表后初始化 `nn.Embedding`
- **验证集与早停**：微调必须监控验证集，避免训练集过拟合

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | 三种策略、lr 惯例、风险、自检问题 |
| `from_scratch/finetune_loop.py` | `FinetuneConfig`、策略命名、epoch 循环抽象 |
| `corpus/` | `sentiment_train.tsv`、`sentiment_val.tsv` |
| `reproduce/finetune_sentiment.py` | 实验入口别名 |
| `experiments/freeze_vs_finetune.py` | scratch / frozen / full 对照 |
| `report.md` | 实验结论与复习命令 |

## 如何运行

```bash
# 核心实验：三种微调策略对照（会训练 skip-gram 预训练嵌入）
python modules/13_nlp_fine_tuning/experiments/freeze_vs_finetune.py

# 从零工具：查看策略命名
python -c "from pathlib import Path; import sys; sys.path.insert(0,'modules/13_nlp_fine_tuning/from_scratch'); from finetune_loop import describe_strategy; print(describe_strategy(True)); print(describe_strategy(False))"
```

依赖：`torch`；会读取 Module 11 语料训练 skip-gram。

## 建议学习顺序

1. 阅读 `notes.md` — 理解 freeze / partial / full 的适用场景
2. 浏览 `corpus/sentiment_train.tsv` 与 `sentiment_val.tsv`
3. 阅读 `from_scratch/finetune_loop.py` — 理解微调配置抽象
4. 阅读 `experiments/freeze_vs_finetune.py` — 看预训练矩阵如何对齐词表
5. 运行对照实验，观察训练/验证 acc 差异
6. 阅读 `report.md` — 理解小验证集下指标的局限性

## 与前后模块的联系

- **前置 Module 11**：提供 skip-gram 预训练嵌入
- **前置 Module 12**：情感分类任务与 BoW/嵌入分类器基线
- **后续**：真实 BERT fine-tune 是本模块思想的工业级放大版

## Depth Checklist

- [x] Fine-tuning loop and strategy comparison
- [x] Frozen embedding vs full fine-tune vs scratch
- [x] Validation accuracy report
