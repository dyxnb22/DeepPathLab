# 06 RNN

Track C — 序列建模与 BPTT（通过时间反向传播）。

## Core Question

RNN 如何处理变长序列？BPTT 中梯度为什么会消失？

## 核心知识点

- **递推隐藏状态**：\(h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)\)，同一组权重在每个时间步复用
- **权重共享**：参数量不随序列长度增长，但计算图沿时间展开
- **BPTT**：将 RNN 按时间展开，从 \(t=T\) 反向传播到 \(t=1\)，是链式法则在时间维的应用
- **梯度消失/爆炸**：\(\partial h_T / \partial h_k\) 含 \(\prod W_{hh}\) 连乘项，长序列时指数衰减或增长
- **字符级语言建模**：预测 \(P(c_{t+1} \mid c_1, \ldots, c_t)\)，用交叉熵训练

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | 递推公式、BPTT、梯度消失直觉 |
| `from_scratch/rnn.py` | NumPy RNN 前向 + 全序列 BPTT |
| `reproduce/char_rnn.py` | PyTorch 字符级 RNN 文本生成 |
| `experiments/sequence_length_gradient.py` | 梯度范数 vs 序列长度实验 |
| `report.md` | 实验结论与 Module 07 衔接 |

## 如何运行

在仓库根目录执行（字符级训练需 `torch`）：

```bash
# 1. 从零实现：前向 + BPTT + copy 任务演示
python modules/06_rnn/from_scratch/rnn.py

# 2. PyTorch 字符级 RNN 训练与生成
python modules/06_rnn/reproduce/char_rnn.py

# 3. 序列长度 vs 梯度范数实验（生成曲线图）
python modules/06_rnn/experiments/sequence_length_gradient.py
```

输出图像保存在 `outputs/06_rnn/`。

## 建议学习顺序

1. 阅读 [D2L Ch.9 RNN](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
2. 通读 `notes.md`，手推 BPTT 中 \(W_{hh}\) 梯度的连乘项
3. 阅读 `from_scratch/rnn.py` 的 `rnn_forward` 与 `rnn_backward_full`
4. 运行 `from_scratch/rnn.py`，观察单条 copy 序列的 \(|dh_0|\)
5. 运行 `experiments/sequence_length_gradient.py`，验证梯度随长度衰减
6. 运行 `reproduce/char_rnn.py`，体验字符级生成
7. 阅读 `report.md` 总结

## 与前后模块的联系

- **前置 — Module 01 Autograd**：BPTT 是计算图 + 链式法则在时间维的实例
- **前置 — Module 03 MLP**：RNN 每步隐藏层本质是带共享权重的 MLP
- **后续 — Module 07 LSTM/GRU**：门控机制专门缓解本模块观察到的梯度消失
- **后续 — Module 08 Transformer**：用 attention 替代递推，路径长度从 O(T) 降为 O(1)

## Workflow

- Read [D2L Ch.9 RNN](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
- Write notes, reproduce, from_scratch, experiments, report

## Depth Checklist

- [x] Minimal RNN forward + BPTT implementation
- [x] Loss and gradient norm vs sequence length experiment
- [x] Character-level generation demo
