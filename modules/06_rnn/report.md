# 报告：Character-level RNN

## 目标

实现 vanilla RNN 前向与 BPTT，训练字符级语言模型，观察序列长度对梯度的影响。

## 实现范围

- `from_scratch/rnn.py` — numpy RNN forward + 全序列 BPTT
- `reproduce/char_rnn.py` — PyTorch 字符级 RNN 生成
- `experiments/sequence_length_gradient.py` — 梯度范数 vs 序列长度

## 实验结果

### 字符级 RNN

200 epoch 后 loss 降至 ~0.0006。生成样本能重复语料片段（小语料上的过拟合式记忆），说明模型学到了局部字符模式。

### BPTT 梯度实验

使用 `W_hh = 0.5 * I` 控制递推权重，在 copy 任务（首步编码、每步预测同一符号）上：

| seq_len | \|dh_0\| after BPTT |
|---------|---------------------|
| 2 | ~0.33 |
| 16 | ~0.23 |
| 64 | 继续衰减 |

更长序列时，回传到初始时刻的隐藏梯度明显缩小，印证梯度消失。

## 深度检查点

- [x] Minimal RNN forward + BPTT
- [x] 序列长度与梯度范数实验
- [x] 字符级生成 demo
- [x] 笔记解释 BPTT 与梯度消失

## 收获

- RNN 的核心是权重共享 + 时间展开
- 单步输出 loss 不足以演示 BPTT；需要跨时间依赖任务
- Vanilla RNN 长程依赖能力有限，引出 LSTM/GRU（Module 07）

## 核心知识点回顾

- **递推**：\(h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)\)，权重跨时间步共享
- **BPTT**：时间展开 + 反向传播；\(W_{hh}\) 梯度含多时间步贡献之和
- **梯度消失**：\(\partial h_T/\partial h_k\) 含 \(\prod \text{diag}(1-h^2) W_{hh}\)，长序列时指数衰减
- **Copy 任务**：强制长程依赖，比字符 LM 更适合观察 BPTT 行为
- **局限**：vanilla RNN 长程记忆弱 → 引出 Module 07 门控机制

## 推荐复习命令

```bash
python modules/06_rnn/from_scratch/rnn.py
python modules/06_rnn/experiments/sequence_length_gradient.py
```

## 下一步

Module 07：门控机制与 copy problem 长程依赖实验。
