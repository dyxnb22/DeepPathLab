# 笔记：NLP 下游应用

## 核心问题

预训练模型如何通过 fine-tuning 适配具体任务？

## 文本分类流程

1. **分词**：文本 → token 序列（本模块用简单正则 `[a-z]+`）
2. **表示**：
   - BoW：\(\mathbf{x} \in \mathbb{R}^{|\mathcal{V}|}\)，第 \(j\) 维为词 \(j\) 的出现次数
   - 嵌入平均：\(\mathbf{h} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{e}_{x_t}\)
3. **分类头**：\(\hat{y} = \sigma(\mathbf{w}^\top \mathbf{h} + b)\)（二分类）
4. **损失**：二元交叉熵
   \[
   \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]
   \]
5. **评估**：准确率、精确率/召回率；**错误分析**检查误分类样本

## BoW + Logistic Regression

\[
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \quad \sigma(z) = \frac{1}{1+e^{-z}}
\]

梯度（单样本）：

\[
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = (\hat{y} - y)\,\mathbf{x}, \quad \frac{\partial \mathcal{L}}{\partial b} = \hat{y} - y
\]

BoW 忽略词序，但对情感等词袋可分的任务仍是强基线。

## Fine-tuning vs 从头训练

| 策略 | 初始化 | 数据需求 | 收敛 |
|------|--------|----------|------|
| 从头训练 | 随机 Embedding | 较多 | 较慢 |
| 预训练初始化 | skip-gram / LM 权重 | 较少 | 较快 |

本模块用 Module 11 的 skip-gram 嵌入初始化 `nn.Embedding.weight`；数据集极小时差异不明显，难任务上优势更显著。

## 错误分析

检查 misclassified 样本的常见模式：

- **否定词**：「not good」vs「good」— BoW 对词序不敏感
- **程度副词**：「very bad」vs「bad」
- **领域外词汇**：训练集未覆盖的词
- **极短文本**：信息不足以判断情感
- **讽刺 / 反语**：需要深层语义理解

应留独立验证集，避免在训练集上过拟合后「无错误可分析」。

## 本模块实验

- BoW + logistic regression 基线（`bow_classifier.py`）
- PyTorch 嵌入平均分类器（`text_classifier.py`）
- skip-gram 预训练初始化 vs 随机初始化（`scratch_vs_finetune.py`）

## 常见陷阱

1. **数据泄漏**：测试文本信息渗入训练（如全局词表在 test 上统计）
2. **在训练集上做错误分析**：应使用验证/测试集
3. **忽视 BoW 基线**：神经网络不一定优于简单 BoW
4. **嵌入未覆盖 OOV 词**：词表外 token 需 `<unk>` 策略
5. **极小数据集过拟合**：本模块 12 条样本，所有方法均可 100%，不能外推结论
6. **混淆预训练域与任务域**：通用语料嵌入对专业领域词汇帮助有限

## 自检问题

1. 写出 BoW logistic regression 的预测公式与损失函数。
2. 嵌入平均分类器如何处理变长句子？padding 时应注意什么？
3. 为什么「not good」对 BoW 是难点？如何改进表示？
4. 预训练初始化在什么条件下比随机初始化更有优势？
5. 本模块所有方法均 100% 准确率，这说明什么？真实项目应如何评估？
6. `bow_classifier.py` 中 `np.clip(x, -500, 500)` 在 sigmoid 里起什么作用？
