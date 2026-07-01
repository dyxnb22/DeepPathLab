# 笔记：LSTM 与 GRU

## 核心问题

LSTM 的门控机制如何解决长程依赖问题？

## Vanilla RNN 的瓶颈

长序列 BPTT 时，梯度经 `W_hh` 连乘会指数衰减（消失）或爆炸。Copy problem 需要在延迟 D 步之后回忆 L 个符号，vanilla RNN 在此任务上经常失败。

## LSTM 结构

三个门控 + 细胞状态：

\[
i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i) \quad \text{(输入门)}
\]
\[
f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f) \quad \text{(遗忘门)}
\]
\[
o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o) \quad \text{(输出门)}
\]
\[
c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
\]
\[
h_t = o_t \odot \tanh(c_t)
\]

**关键**：\(c_t\) 是线性累加路径，\(f_t \approx 1\) 时梯度可几乎无损地跨越多个时间步。

## GRU 结构

将 LSTM 简化为两个门：

- **重置门** \(r_t\)：控制历史状态参与候选更新的程度
- **更新门** \(z_t\)：在旧状态与新候选之间插值

\[
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\]

参数更少，许多任务上与 LSTM 相当。

## Copy Problem 任务设计

序列布局：`[L 个符号][D 步空白][L 步回忆输出]`

- 输入：前 L 步为待记忆符号，中间 D 步为 blank token
- 目标：最后 L 步输出最初 L 个符号
- 损失只在目标位置计算

延迟 D 越大，对长程记忆要求越高。

## 门控直觉

| 机制 | 作用 |
|------|------|
| 遗忘门 \(f_t\) | 决定保留多少旧 cell 状态 |
| 输入门 \(i_t\) | 决定写入多少新信息 |
| 更新门 \(z_t\) (GRU) | 决定新旧隐藏状态混合比例 |

## 与 Module 06 的联系

- RNN 是骨架；LSTM/GRU 在 cell 内加入可控信息流
- Copy problem 是检验长程依赖的标准合成任务
