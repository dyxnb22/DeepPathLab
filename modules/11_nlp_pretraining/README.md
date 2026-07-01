# 11 NLP Pretraining

Track E — 词嵌入与预训练表示。

## 核心问题

Word2Vec 和 BERT 分别学到了什么类型的语言知识？

## 核心知识点

- **分布式假设**：上下文相似的词，语义往往相近
- **Skip-gram**：用中心词预测上下文，学到静态词向量
- **负采样 / softmax**：词表大时用近似目标加速训练（本模块用全 softmax 简化）
- **词向量性质**：最近邻、类比关系（king - man + woman ≈ queen）
- **MLM（Masked LM）**：随机 mask token，用双向上下文预测，学到动态表示
- **静态 vs 动态嵌入**：Word2Vec 一词一向量；BERT 同词在不同句中向量不同

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | Skip-gram / MLM 公式、陷阱、自检问题 |
| `from_scratch/skipgram.py` | numpy 版 skip-gram 训练与最近邻查询 |
| `corpus/tiny_corpus.txt` | 微型英语语料 |
| `reproduce/tiny_mlm.py` | PyTorch Transformer 微型 MLM |
| `experiments/embedding_viz.py` | PCA 可视化词嵌入 |
| `report.md` | 实验结论与复习命令 |

## 如何运行

```bash
# 从零实现：训练 skip-gram 并打印最近邻
python modules/11_nlp_pretraining/from_scratch/skipgram.py

# 嵌入 PCA 可视化（输出到 outputs/）
python modules/11_nlp_pretraining/experiments/embedding_viz.py

# 微型 MLM 训练与 mask 预测 demo
python modules/11_nlp_pretraining/reproduce/tiny_mlm.py
```

依赖：`numpy`（from_scratch）、`torch` + `matplotlib`（reproduce / experiments）。

## 建议学习顺序

1. 阅读 `notes.md`，对比 Word2Vec 与 BERT 的训练目标
2. 浏览 `corpus/tiny_corpus.txt`，了解语料规模限制
3. 运行 `skipgram.py`，观察 king/queen 等词的最近邻
4. 阅读 `skipgram.py` 源码，理解 softmax 梯度更新
5. 运行 `embedding_viz.py`，在 2D 空间中看语义聚类
6. 运行 `tiny_mlm.py`，体验 mask 预测流程
7. 阅读 `report.md`，思考静态嵌入如何用于 Module 12

## 模块联系

- **← Module 06–08**：RNN / LSTM / Transformer 是 MLM 与后续预训练的架构基础
- **← Module 09**：skip-gram 与 MLM 训练都依赖 SGD / Adam
- **→ Module 12**：skip-gram 嵌入可初始化下游分类器的 Embedding 层
- **→ Module 13**：大模型 fine-tuning 是 MLM 预训练的工业化延伸
