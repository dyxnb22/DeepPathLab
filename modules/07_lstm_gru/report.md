# 报告：LSTM & GRU 门控序列建模

## 目标

从零实现 LSTM/GRU 单元，在 copy problem 上对比 vanilla RNN、LSTM、GRU 的长程记忆能力。

## 实现范围

- `from_scratch/lstm_cell.py` — numpy LSTM 单步前向 + 门控检查
- `from_scratch/gru_cell.py` — numpy GRU 单步前向
- `copy_task.py` — copy problem 数据生成
- `experiments/copy_problem.py` — 简单/困难两种设置的对照训练

## 实验结果

### Easy Copy（seq_len=10, delay=2, L=4）

| 模型 | 最终 val acc |
|------|-------------|
| RNN | 69.1% |
| LSTM | **82.8%** |
| GRU | **88.3%** |

短延迟下三种模型都能学习，门控模型明显优于 RNN。

### Hard Copy（seq_len=30, delay=10, L=10）

| 模型 | 最终 val acc |
|------|-------------|
| RNN | 24.5% |
| LSTM | 26.6% |
| GRU | **45.9%** |

长延迟下 RNN/LSTM 接近随机（4 类约 25%），GRU 仍能部分回忆。说明门控机制对长程依赖有帮助，但任务难度高时仍需更大模型或更多训练。

## 深度检查点

- [x] LSTM 和 GRU cell 从零实现（前向）
- [x] Copy problem 合成任务
- [x] Gated vs vanilla RNN 对照

## 收获

- 遗忘门初始化为 1 有助于早期训练（numpy demo 中使用）
- One-hot + linear 投影比纯 embedding 在此合成任务上更稳定
- 门控不是万能：困难 copy 任务上 LSTM 也会失效，需结合容量与超参

## 核心知识点回顾

- **LSTM 细胞状态**：\(c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\)，线性路径缓解梯度连乘
- **三门**：输入/遗忘/输出门控制写入、保留、读出
- **GRU**：重置门 + 更新门，参数更少，hard copy 上本实验表现最好
- **Copy problem**：延迟 \(D\) 是长程依赖难度的旋钮
- **门控非万能**：困难设置下 LSTM 也会失效，需结合容量与超参

## 推荐复习命令

```bash
python modules/07_lstm_gru/from_scratch/lstm_cell.py
python modules/07_lstm_gru/from_scratch/gru_cell.py
python modules/07_lstm_gru/experiments/copy_problem.py
```

## 下一步

Module 08：Self-attention 突破递推瓶颈，实现并行序列建模。
