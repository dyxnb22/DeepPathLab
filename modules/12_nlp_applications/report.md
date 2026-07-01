# 报告：NLP Applications

## 目标

实现情感分类端到端流程，对比 BoW、嵌入分类器与预训练初始化。

## 实验结果

### 分类器（12 条 tiny 语料）

| 方法 | 准确率 |
|------|--------|
| BoW + logistic regression | 100% |
| PyTorch 嵌入平均分类器 | 100% |
| 随机初始化 | 100% |
| Skip-gram 预训练初始化 | 100% |

数据集极小且线性可分，所有方法均完美拟合。预训练初始化优势在大数据/难任务上更明显。

### 错误分析

训练集上无 misclassified 样本。真实场景应留验证集并检查否定词、讽刺等边界案例。

## 深度检查点

- [x] 端到端下游任务
- [x] scratch vs 预训练初始化对照
- [x] 错误分析流程

## 收获

完整 NLP 应用链路：表示 → 模型 → 训练 → 评估 → 错误分析。BoW 基线不可忽视；预训练初始化是提升小数据场景的有效手段。

## 核心知识点回顾

1. **文本分类流水线**：分词 → 表示（BoW / 嵌入）→ 分类头 → 损失 → 评估
2. **BoW + LR**：简单强基线；忽略词序，对否定句敏感
3. **嵌入平均**：可学习表示 + mean pooling，捕获一定语义
4. **预训练初始化**：复用 Module 11 skip-gram 嵌入，加速收敛、改善小数据泛化
5. **错误分析**：从误分类样本发现数据与模型盲区

## 推荐复习命令

```bash
# BoW 基线
python modules/12_nlp_applications/from_scratch/bow_classifier.py

# PyTorch 嵌入分类器
python modules/12_nlp_applications/reproduce/text_classifier.py

# 预训练 vs 随机初始化（依赖 Module 11 skip-gram）
python modules/12_nlp_applications/experiments/scratch_vs_finetune.py

# 错误分析流程
python modules/12_nlp_applications/experiments/error_analysis.py
```

复习时对照 `notes.md` 自检问题，思考若扩展到 IMDB 规模，pipeline 需哪些改动。

## 路线总结

Module 01–12 形成从 autograd 到 NLP 应用的完整学习路径。后续 Module 13 将深入大模型 fine-tuning。
