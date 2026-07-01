# 笔记：NLP 下游应用

## 核心问题

预训练模型如何通过 fine-tuning 适配具体任务？

## 文本分类流程

1. 文本 → tokenization
2. 表示（BoW / 嵌入 / 预训练 encoder）
3. 分类头（线性层）
4. 在标注数据上训练

## Fine-tuning vs 从头训练

- **从头训练**：随机初始化嵌入，需要更多数据
- **预训练初始化**：用词向量或语言模型权重初始化，小数据集上更快收敛

## 错误分析

检查 misclassified 样本，常见模式：

- 否定词（"not good" vs "good"）
- 领域外词汇
- 极短文本信息不足

## 本模块实验

- BoW + logistic regression 基线
- PyTorch 嵌入平均分类器
- 预训练 skip-gram 初始化 vs 随机初始化
