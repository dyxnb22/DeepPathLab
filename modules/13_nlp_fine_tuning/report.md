# 报告：NLP Fine-Tuning

## 目标

对比随机初始化、冻结预训练嵌入、全量微调三种策略在情感分类上的表现。

## 实现范围

- `from_scratch/finetune_loop.py` — `FinetuneConfig`、epoch 循环、策略命名
- `reproduce/finetune_sentiment.py` — PyTorch 微调复现
- `experiments/freeze_vs_finetune.py` — skip-gram 预训练 + 三种策略对照

## 实验设置

- 训练集 14 条、验证集 4 条（扩充自 Module 12 语料）
- 预训练：Module 11 skip-gram 嵌入
- 模型：嵌入平均 + 线性分类头

## 运行

```bash
python3 modules/13_nlp_fine_tuning/experiments/freeze_vs_finetune.py
```

## 结果

| 策略 | 训练 acc | 验证 acc |
|------|---------|---------|
| scratch | 100% | 50% |
| frozen_embed | 57% | 50% |
| full_finetune | 100% | 25% |

验证集极小，指标波动大。观察：
- scratch/full 易过拟合训练集
- frozen 欠拟合但更保守
- 真实场景应使用更大验证集和早停

## 深度检查点

- [x] 微调策略对照实验
- [x] 预训练嵌入初始化
- [x] 笔记解释 freeze vs fine-tune 场景

## 收获

微调 = 在预训练表示上适配下游任务；是否解冻取决于数据量与任务差异。

## 核心知识点回顾

- Linear probe / frozen / full fine-tune 的适用边界
- 预训练嵌入对齐词表后初始化
- 小验证集下指标不可靠，需早停与更大数据

## 推荐复习命令

```bash
python modules/13_nlp_fine_tuning/experiments/freeze_vs_finetune.py
```

## 下一步

- 加入 partial fine-tune（只解冻顶层）
- 用更大语料和早停重新评估
