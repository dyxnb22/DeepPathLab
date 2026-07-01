# 笔记：LSTM 与 GRU

## 核心问题

LSTM 的门控机制如何解决长程依赖问题？

## Vanilla RNN 的瓶颈

长序列 BPTT 时，梯度经 \(W_{hh}\) 连乘会指数衰减（消失）或爆炸。Copy problem 需要在延迟 \(D\) 步之后回忆 \(L\) 个符号，vanilla RNN 在此任务上经常失败。

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
\tilde{c}_t = \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c) \quad \text{(候选细胞)}
\]
\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]
\[
h_t = o_t \odot \tanh(c_t)
\]

**关键**：\(c_t\) 是线性累加路径。当 \(f_t \approx 1, i_t \approx 0\) 时，\(c_t \approx c_{t-1}\)，信息可几乎无损跨越多个时间步；\(\partial c_t / \partial c_{t-1} = f_t\) 避免了 \(W_{hh}\) 的反复连乘。

## GRU 结构

将 LSTM 简化为两个门：

\[
r_t = \sigma(W_{xr}x_t + W_{hr}h_{t-1} + b_r) \quad \text{(重置门)}
\]
\[
z_t = \sigma(W_{xz}x_t + W_{hz}h_{t-1} + b_z) \quad \text{(更新门)}
\]
\[
\tilde{h}_t = \tanh(W_{xh}x_t + W_{hh}(r_t \odot h_{t-1}) + b_h)
\]
\[
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\]

- **重置门** \(r_t\)：控制历史状态参与候选更新的程度
- **更新门** \(z_t\)：在旧状态与新候选之间插值（类似 \(f_t\) 与 \(i_t\) 的合并）

参数约为 LSTM 的 3/4，许多任务上与 LSTM 相当。

## Copy Problem 任务设计

序列布局：`[L 个符号][D 步空白][L 步回忆输出]`

- 输入：前 \(L\) 步为待记忆符号，中间 \(D\) 步为 blank token
- 目标：最后 \(L\) 步输出最初 \(L\) 个符号
- 损失只在目标位置计算

延迟 \(D\) 越大，对长程记忆要求越高。总长度约为 \(L + D + L\)。

## 门控直觉

| 机制 | 作用 | 典型行为 |
|------|------|----------|
| 遗忘门 \(f_t\) | 保留多少旧 cell | 接近 1 → 长期记忆 |
| 输入门 \(i_t\) | 写入多少新信息 | 编码阶段较大 |
| 输出门 \(o_t\) | 读出多少 cell | 回忆阶段较大 |
| 更新门 \(z_t\) (GRU) | 新旧隐藏状态混合 | 类似插值 |

## 与 Module 06 的联系

- RNN 是骨架；LSTM/GRU 在 cell 内加入可控信息流
- Copy problem 是检验长程依赖的标准合成任务
- Module 06 的 `sequence_copy_task` 是简化版；本模块 copy 含空白延迟

## 常见误区

1. **「LSTM 完全解决梯度消失」** — 困难 copy（大 \(D\)）上 LSTM 仍可能失败；门控是缓解而非消除
2. **「GRU 总是不如 LSTM」** — 本模块 hard copy 上 GRU 优于 LSTM，说明任务与容量很重要
3. **忽视遗忘门偏置初始化** — 将 \(b_f\) 初始化为 1 使 \(f_t\) 初始接近 1，有助于早期训练
4. **混淆 \(h_t\) 与 \(c_t\)** — 长程信息主要在 \(c_t\)；\(h_t\) 是过滤后的读出

## 局限

- `from_scratch/` 仅实现前向，未含完整 BPTT
- 困难 copy 需要更大 hidden_dim 或更多训练
- 未覆盖双向 RNN、多层堆叠等变体

## 自检问题

1. 写出 LSTM 中 \(c_t\) 的更新公式，并说明 \(f_t\) 和 \(i_t\) 分别控制什么。
2. GRU 的更新门 \(z_t\) 与 LSTM 的遗忘门/输入门有何对应关系？
3. 为何 copy problem 的延迟 \(D\) 增大后，vanilla RNN 准确率接近随机而 GRU 仍能部分回忆？
