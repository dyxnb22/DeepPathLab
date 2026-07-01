# 12 NLP Applications

Track E — 下游 NLP 任务应用。

## 核心问题

预训练模型如何通过 fine-tuning 适配具体任务？

## 核心知识点

- **文本分类流水线**：分词 → 表示 → 分类头 → 训练 → 评估
- **BoW 表示**：词袋计数/二值化，配合 logistic regression 作为强基线
- **嵌入平均**：可学习 Embedding 层 + mean pooling + 线性分类头
- **预训练初始化**：用 skip-gram 等静态嵌入初始化 Embedding 权重
- **Fine-tuning vs 从头训练**：小数据上预训练初始化通常收敛更快
- **错误分析**：检查 misclassified 样本，发现否定词、领域外词等模式

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | 分类流程、迁移策略、陷阱、自检问题 |
| `from_scratch/bow_classifier.py` | numpy 版 BoW + logistic regression |
| `corpus/tiny_sentiment.tsv` | 微型情感标注语料 |
| `reproduce/text_classifier.py` | PyTorch 嵌入平均分类器 |
| `experiments/scratch_vs_finetune.py` | 随机 vs skip-gram 初始化对照 |
| `experiments/error_analysis.py` | 误分类样本检查 |
| `report.md` | 实验结论与复习命令 |

## 如何运行

```bash
# 从零实现：BoW + logistic regression 基线
python modules/12_nlp_applications/from_scratch/bow_classifier.py

# PyTorch 嵌入平均分类器
python modules/12_nlp_applications/reproduce/text_classifier.py

# 预训练 skip-gram 初始化 vs 随机初始化
python modules/12_nlp_applications/experiments/scratch_vs_finetune.py

# 错误分析（检查 misclassified 样本）
python modules/12_nlp_applications/experiments/error_analysis.py
```

依赖：`numpy`（from_scratch）、`torch`（reproduce / experiments）。`scratch_vs_finetune.py` 会调用 Module 11 的 skip-gram。

## 建议学习顺序

1. 阅读 `notes.md`，梳理文本分类端到端流程
2. 浏览 `corpus/tiny_sentiment.tsv`，了解数据格式与规模
3. 运行 `bow_classifier.py`，建立 BoW 基线认知
4. 阅读 `bow_classifier.py` 源码，理解 sigmoid 与交叉熵梯度
5. 运行 `text_classifier.py`，对比神经网络表示
6. 运行 `scratch_vs_finetune.py`，观察预训练初始化效果
7. 运行 `error_analysis.py`，练习误分类诊断
8. 阅读 `report.md`，回顾 Module 01–12 完整路径

## 模块联系

- **← Module 11**：skip-gram 嵌入用于本模块 Embedding 初始化
- **← Module 09**：分类器训练依赖 Adam / SGD 等优化器
- **← Module 10**：视觉迁移学习与 NLP fine-tuning 共享「预训练 + 任务头」范式
- **→ Module 13**：大模型（BERT 等）fine-tuning 是本模块的规模化延伸
- **路线总结**：Module 01–12 形成从 autograd 到 NLP 应用的完整学习路径
