# 08 Attention & Transformer

Track C — 注意力机制与自注意力 Transformer。

## Core Question

Self-attention 如何让每个位置直接 attend 到所有其他位置？

## 核心知识点

- **从 RNN 到 Attention**：递推路径长度 O(T)；self-attention 一步计算所有位置对的相关性，路径 O(1)
- **Scaled Dot-Product Attention**：\(\text{softmax}(QK^T / \sqrt{d_k}) V\)，缩放防止 softmax 饱和
- **Q / K / V 角色**：Query 问什么、Key 索引标签、Value 携带信息
- **Transformer Block**：Self-Attention → Add&Norm → FFN → Add&Norm（残差 + LayerNorm）
- **位置编码**：attention 置换不变，需注入位置信息（正弦编码或学习式）

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | Attention 公式、Transformer 结构、与 RNN 对比 |
| `from_scratch/attention.py` | NumPy scaled dot-product + self-attention |
| `from_scratch/transformer_block.py` | NumPy 单头 encoder block |
| `reproduce/mini_transformer.py` | PyTorch mini transformer on copy task |
| `experiments/attention_viz.py` | 注意力权重热力图 |
| `report.md` | copy 任务结果与可视化解读 |

## 如何运行

在仓库根目录执行（训练与可视化需 `torch`、`matplotlib`）：

```bash
# 1. Scaled dot-product attention 从零实现
python modules/08_attention_transformer/from_scratch/attention.py

# 2. Transformer encoder block（attention + FFN + LayerNorm）
python modules/08_attention_transformer/from_scratch/transformer_block.py

# 3. Mini transformer 在 copy task 上训练
python modules/08_attention_transformer/reproduce/mini_transformer.py

# 4. 注意力权重可视化（生成热力图）
python modules/08_attention_transformer/experiments/attention_viz.py
```

输出图像保存在 `outputs/08_attention_transformer/`。

## 建议学习顺序

1. 阅读 [D2L Ch.11 Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
2. 回顾 Module 07 copy problem，理解本模块复用同一任务的原因
3. 通读 `notes.md`，手推 attention 输出维度
4. 运行 `attention.py`，确认权重行和为 1
5. 阅读 `transformer_block.py`，理解 Add&Norm 与 Module 05 残差的联系
6. 运行 `mini_transformer.py`，对比 Module 07 GRU 结果
7. 运行 `attention_viz.py`，观察 recall 位置对早期符号的注意力

## 与前后模块的联系

- **前置 — Module 05 ResNet**：Transformer 的残差连接 + LayerNorm 延续 skip 思想
- **前置 — Module 07 LSTM/GRU**：本模块 `mini_transformer.py` 复用 `copy_task.py`；attention 是长程依赖的另一条路径
- **后续 — Module 11 NLP Pretraining**：BERT/GPT 等建立在 Transformer 之上
- **后续 — Module 09 Optimization**：Transformer 训练对优化器与学习率更敏感

## Workflow

- Read [D2L Ch.11 Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- Write notes, reproduce, from_scratch, experiments, report

## Depth Checklist

- [x] Scaled dot-product attention from scratch
- [x] Attention weight visualization
- [x] Mini transformer on toy sequence task
