# 模块索引

15 个学习模块，按编号顺序推进效果最佳。每个模块的详细说明见各自 `README.md`。

**全局入口**：[docs/getting-started.md](../docs/getting-started.md) · [docs/learning-guide.md](../docs/learning-guide.md) · [docs/study-checklist.md](../docs/study-checklist.md)

```bash
python3 scripts/verify_all.py          # 一键自检
python3 scripts/run_module.py 03       # 运行指定模块主脚本
```

---

| # | 目录 | 一句话 | Track |
|---|------|--------|-------|
| 01 | [01_preliminaries_autograd](01_preliminaries_autograd/) | 标量 autograd 引擎与梯度检查 | A |
| 02 | [02_linear_models](02_linear_models/) | 线性回归与 softmax 分类 | B |
| 03 | [03_mlp](03_mlp/) | 多层感知机与矩阵 backprop | B |
| 04 | [04_cnn](04_cnn/) | 卷积、池化、LeNet | D |
| 05 | [05_modern_cnn](05_modern_cnn/) | ResNet 与深度退化问题 | D |
| 06 | [06_rnn](06_rnn/) | 字符级 RNN 与 BPTT | C |
| 07 | [07_lstm_gru](07_lstm_gru/) | LSTM/GRU 与 copy problem | C |
| 08 | [08_attention_transformer](08_attention_transformer/) | 自注意力与 mini Transformer | C |
| 09 | [09_optimization](09_optimization/) | SGD、Momentum、Adam 对照 | B |
| 10 | [10_computer_vision_applications](10_computer_vision_applications/) | 迁移学习与数据增强 | D |
| 11 | [11_nlp_pretraining](11_nlp_pretraining/) | Skip-gram 与 tiny MLM | E |
| 12 | [12_nlp_applications](12_nlp_applications/) | 情感分类与 BoW 基线 | E |
| 13 | [13_nlp_fine_tuning](13_nlp_fine_tuning/) | 冻结 vs 全量微调 | E |
| 14 | [14_recommender_systems](14_recommender_systems/) | 矩阵分解推荐 | F |
| 15 | [15_reinforcement_learning](15_reinforcement_learning/) | GridWorld Q-learning | F |

---

## 统一目录结构

```text
modules/XX_name/
├── README.md          # 模块导读（核心知识点、运行命令、学习顺序）
├── notes.md           # 概念笔记、踩坑、自检问题
├── report.md          # 实验结论、知识点回顾、复习命令
├── from_scratch/      # 手写核心实现 ← 学习重点
├── reproduce/         # 框架基线复现
└── experiments/       # 消融、可视化、诊断
```

## 按目标选模块

| 你想学… | 从这里开始 |
|---------|------------|
| 理解 PyTorch autograd 底层 | 01 → 03 |
| 做 CV 项目 | 01–05 → 10 |
| 做 NLP / LLM 基础 | 01–03 → 06–08 → 11–13 |
| 优化与训练技巧 | 03 → 09 |
| 推荐系统入门 | 02, 09 → 14 |
| 强化学习入门 | 15（概率基础即可） |
