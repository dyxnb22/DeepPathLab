# 笔记：注意力机制与 Transformer

## 核心问题

Self-attention 如何让每个位置直接 attend 到所有其他位置？

## 从 RNN 到 Attention

RNN 按时间递推，位置 \(t\) 只能间接访问 distant tokens。Self-attention 在一步内计算所有位置对 \((i, j)\) 的相关性，路径长度 O(1)。

## Scaled Dot-Product Attention

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

- **Q**（Query）：当前位置在「问什么」
- **K**（Key）：各位置的「索引标签」
- **V**（Value）：各位置携带的信息
- **缩放** \(\sqrt{d_k}\)：防止点积过大导致 softmax 饱和

## Self-Attention

当 \(Q, K, V\) 都来自同一序列 \(X\)：

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

每个位置可以 attend 到包括自己在内的所有位置（需 causal mask 用于自回归生成）。

## Transformer Encoder Block

```text
x -> MultiHeadAttention -> Add&Norm -> FFN -> Add&Norm -> out
```

- **残差连接** + **LayerNorm** 稳定训练
- **FFN** 在 token 维度上做非线性变换

## Positional Encoding

Attention 本身置换不变，需注入位置信息。正弦位置编码：

\[
PE_{pos, 2i} = \sin(pos / 10000^{2i/d}),\quad PE_{pos, 2i+1} = \cos(...)
\]

## 与 Module 07 的对比

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 长程依赖路径 | O(T) 递推 | O(1) 直接 attend |
| 并行度 | 低 | 高（训练时） |
| 归纳偏置 | 时间局部性 | 较弱，需更多数据 |

## 局限

- 注意力复杂度 O(T²)，长序列成本高
- 本模块 mini transformer 规模很小，主要用于机制理解
