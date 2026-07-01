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

## BPTT（Backpropagation Through Time）

将 RNN 按时间展开成计算图，从最后时刻反向传播到初始时刻。对 \(W_{hh}\) 的梯度包含连乘项：

\[
\frac{\partial h_T}{\partial h_k} \propto \prod_{t=k+1}^{T} \text{diag}(1 - h_t^2) \cdot W_{hh}
\]

当 \(T\) 很大且 \(\|W_{hh}\|\) 不够大时，连乘使梯度指数衰减（**梯度消失**）；若 \(\|W_{hh}\| > 1\) 则可能**梯度爆炸**。

## 字符级语言建模

将文本视为字符序列，预测下一个字符。任务目标：

\[
P(c_{t+1} \mid c_1, \ldots, c_t)
\]

RNN 在每个位置输出 vocab 上的 logits，用交叉熵训练。

## 截断 BPTT

对超长序列，只在最近 \(K\) 步反向传播，平衡内存与长期依赖学习。本模块实验用完整 BPTT 演示梯度随长度变化。

## 与 Module 01/03 的联系

- BPTT 是链式法则在时间维度的应用
- 与 MLP 的区别：权重共享 + 时间展开

## 局限

- 本模块 vanilla RNN 未解决长程依赖（Module 07 LSTM/GRU）
- 字符级语料很小，生成质量有限
