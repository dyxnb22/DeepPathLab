# 笔记：注意力机制与 Transformer

## 核心问题

Self-attention 如何让每个位置直接 attend 到所有其他位置？

## 从 RNN 到 Attention

RNN 按时间递推，位置 \(t\) 只能间接访问 distant tokens（路径长度 \(O(T)\)）。

Self-attention 在一步内计算所有位置对 \((i, j)\) 的相关性：

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

每个 query 位置可直接「看到」所有 key 位置，信息路径长度 \(O(1)\)（以层数计）。

## Scaled Dot-Product Attention

\[
\text{scores}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}, \quad
\alpha_{ij} = \frac{\exp(\text{scores}_{ij})}{\sum_{j'} \exp(\text{scores}_{ij'})}, \quad
\text{out}_i = \sum_j \alpha_{ij} v_j
\]

- **Q**（Query）：当前位置在「问什么」
- **K**（Key）：各位置的「索引标签」，决定匹配分数
- **V**（Value）：各位置携带的信息，按权重加权求和
- **缩放** \(\sqrt{d_k}\)：\(d_k\) 较大时点积方差增大，softmax 趋于 one-hot，梯度变小

### 维度检查

\(Q \in \mathbb{R}^{T \times d_k}\)，\(K \in \mathbb{R}^{T \times d_k}\)，\(V \in \mathbb{R}^{T \times d_v}\)

\(\Rightarrow\) 输出 \(\in \mathbb{R}^{T \times d_v}\)，序列长度不变。

## Self-Attention

当 \(Q, K, V\) 都来自同一序列 \(X\)：

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

每个位置可以 attend 到包括自己在内的所有位置。

**Causal mask**（自回归生成）：将 \(j > i\) 的 scores 置为 \(-\infty\)，使位置 \(i\) 只能看到过去。

## Transformer Encoder Block

```text
x -> MultiHeadAttention -> Add&Norm -> FFN -> Add&Norm -> out
```

\[
x' = \text{LayerNorm}(x + \text{Attention}(x))
\]
\[
\text{out} = \text{LayerNorm}(x' + \text{FFN}(x'))
\]

- **残差连接**（Module 05）：稳定深层训练
- **LayerNorm**：在特征维归一化，与 BatchNorm 不同
- **FFN**：\(\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2\)，逐 token 非线性变换

## Positional Encoding

Attention 本身**置换不变**（打乱 token 顺序，若 \(Q,K,V\) 同步打乱则输出等价），需注入位置信息。

正弦位置编码：

\[
PE_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
\]

## 与 Module 07 的对比

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 长程依赖路径 | O(T) 递推 | O(1) 直接 attend |
| 训练并行度 | 低（逐步递推） | 高（整序列并行） |
| 推理复杂度 | O(T) per step | O(T²) attention |
| 归纳偏置 | 时间局部性 | 较弱，需更多数据/正则 |

## 常见误区

1. **「Attention 就是加权平均」** — 权重是**学习**的、**动态**的，且 Q/K/V 经不同投影
2. **忘记 \(\sqrt{d_k}\) 缩放** — 不缩放时高维点积使 softmax 饱和，训练困难
3. **「Transformer 不需要位置信息」** — 必须加 positional encoding 或等价结构
4. **混淆 encoder 与 decoder mask** — encoder 通常双向；decoder 需 causal mask
5. **忽视 O(T²) 复杂度** — 长序列时 memory 与计算是实际瓶颈

## 局限

- 注意力复杂度 O(T²)，长序列成本高
- 本模块 mini transformer 规模很小，主要用于机制理解
- `from_scratch/` 为单头、无 multi-head；未实现完整训练循环

## 自检问题

1. 写出 scaled dot-product attention 公式，并解释为何除以 \(\sqrt{d_k}\)。
2. Self-attention 中 Q、K、V 各自由什么矩阵从输入 \(X\) 投影得到？输出维度如何？
3. 在 copy task 上，recall 位置的 attention 权重应对哪些早期位置较高？如何用 `attention_viz.py` 验证？
