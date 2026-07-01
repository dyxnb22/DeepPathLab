# 笔记：循环神经网络

## 核心问题

RNN 如何处理变长序列？BPTT 中梯度为什么会消失？

## 递推公式

给定输入序列 \(x_1, \ldots, x_T\)：

\[
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
\]
\[
y_t = W_{hy} h_t + b_y
\]

隐藏状态 \(h_t\) 是「对过去的摘要」，同一组权重在每个时间步复用。

**参数量**：\(O((d_x + d_h) \cdot d_h + d_h \cdot d_y)\)，与 \(T\) 无关；但前向/反向计算量均为 \(O(T)\)。

## BPTT（Backpropagation Through Time）

将 RNN 按时间展开成计算图，从最后时刻反向传播到初始时刻。

对 \(W_{hh}\) 的梯度是各时间步贡献之和：

\[
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}
\]

关键连乘项（隐藏状态跨步传播）：

\[
\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \text{diag}(1 - h_t^2) \cdot W_{hh}
\]

当 \(T\) 很大且 \(\|W_{hh}\| < 1\) 时，连乘使梯度指数衰减（**梯度消失**）；若 \(\|W_{hh}\| > 1\) 则可能**梯度爆炸**。

### 直觉

- 每个 \(\tanh\) 的导数 \(\leq 1\)，再乘以 \(W_{hh}\)，长距离回传时信号不断缩小
- 早期时间步对 loss 的贡献梯度极弱 → 难以学习长程依赖

## 字符级语言建模

将文本视为字符序列，预测下一个字符：

\[
P(c_{t+1} \mid c_1, \ldots, c_t) = \text{softmax}(W_{hy} h_t + b_y)
\]

RNN 在每个位置输出 vocab 上的 logits，用交叉熵训练。

## 截断 BPTT（Truncated BPTT）

对超长序列，只在最近 \(K\) 步反向传播：

- **优点**：内存可控，适合长文本
- **缺点**：超过 \(K\) 步的依赖无法直接学习

本模块 `from_scratch/rnn.py` 用完整 BPTT；`experiments/` 用 copy 任务演示梯度随长度变化。

## 与 Module 01/03 的联系

- BPTT 是链式法则在时间维度的应用（Module 01 计算图）
- 与 MLP 的区别：权重共享 + 时间展开；每步 \(h_t\) 等价于一个小 MLP

## 常见误区

1. **「RNN 参数量随序列长度增长」** — 错误；增长的是**计算图节点数**，不是参数量
2. **「单步 loss 足以演示 BPTT」** — 若每步独立预测，早期时间步梯度可能仍较强；需 copy 类任务强制长程依赖
3. **「梯度消失 = 训练不收敛」** — 短序列或强局部模式下 RNN 仍可训练；消失主要影响**长程记忆**
4. **混淆 BPTT 与截断 BPTT** — 完整 BPTT 回传到 \(t=1\)；截断版在固定窗口处切断

## 局限

- 本模块 vanilla RNN 未解决长程依赖（Module 07 LSTM/GRU）
- 字符级语料很小，生成质量有限
- `from_scratch/` 仅实现前向 + 反向，未含完整训练循环

## 自检问题

1. 写出 \(h_t\) 的递推公式，并说明 \(W_{xh}\) 与 \(W_{hh}\) 分别作用于什么。
2. 在 BPTT 中，\(\partial h_T / \partial h_0\) 的连乘项包含哪些因子？为何会导致梯度消失？
3. Copy 任务（首步编码、每步预测同一符号）为何比字符级 LM 更适合观察 BPTT 梯度衰减？
