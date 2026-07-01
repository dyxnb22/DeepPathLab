# 07 LSTM & GRU

Track C — 门控机制与长程依赖。

## Core Question

LSTM 的门控机制如何解决长程依赖问题？

## 核心知识点

- **Vanilla RNN 瓶颈**：BPTT 中 \(W_{hh}\) 连乘导致梯度消失，长延迟 copy 任务失败
- **LSTM 细胞状态 \(c_t\)**：线性累加路径 \(c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\)，\(f_t \approx 1\) 时梯度可跨多步传播
- **三门机制**：输入门 \(i_t\)、遗忘门 \(f_t\)、输出门 \(o_t\) 控制写入、保留、读出
- **GRU 简化**：重置门 \(r_t\) + 更新门 \(z_t\)，参数更少，许多任务与 LSTM 相当
- **Copy Problem**：合成任务 \([L\text{符号}][D\text{空白}][L\text{回忆}]\)，延迟 \(D\) 越大越难

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | LSTM/GRU 公式、门控直觉、copy 任务设计 |
| `from_scratch/lstm_cell.py` | NumPy LSTM 单步前向 + 遗忘门偏置初始化 |
| `from_scratch/gru_cell.py` | NumPy GRU 单步前向 |
| `copy_task.py` | Copy problem 数据生成 |
| `experiments/copy_problem.py` | RNN / LSTM / GRU 简单与困难设置对照 |
| `reproduce/gated_rnn_benchmark.py` | 实验入口（转发至 copy_problem） |
| `report.md` | 门控 vs vanilla 对比结论 |

## 如何运行

在仓库根目录执行（实验需 `torch`）：

```bash
# 1. LSTM 单步前向与门控检查
python modules/07_lstm_gru/from_scratch/lstm_cell.py

# 2. GRU 单步前向
python modules/07_lstm_gru/from_scratch/gru_cell.py

# 3. RNN / LSTM / GRU copy problem 对照（easy + hard）
python modules/07_lstm_gru/experiments/copy_problem.py

# 等价入口
python modules/07_lstm_gru/reproduce/gated_rnn_benchmark.py
```

输出图像保存在 `outputs/07_lstm_gru/`。

## 建议学习顺序

1. 阅读 [D2L Ch.10 Modern RNN](https://d2l.ai/chapter_recurrent-modern/index.html)
2. 回顾 Module 06 的 BPTT 梯度消失，明确本模块要解决的问题
3. 通读 `notes.md`，对比 LSTM 与 GRU 的门控结构
4. 运行 `lstm_cell.py` 与 `gru_cell.py`，观察门控激活值
5. 阅读 `copy_task.py` 理解序列布局
6. 运行 `experiments/copy_problem.py`，对比 easy / hard 设置
7. 阅读 `report.md`

## 与前后模块的联系

- **前置 — Module 06 RNN**：本模块在相同递推骨架上增加门控；copy 任务延续长程依赖主题
- **后续 — Module 08 Transformer**：attention 用 O(1) 路径替代门控递推；Module 08 复用本模块 `copy_task.py`
- **横向对比**：LSTM/GRU 仍串行递推；Transformer 可并行训练

## Workflow

- Read [D2L Ch.10 Modern RNN](https://d2l.ai/chapter_recurrent-modern/index.html)
- Write notes, reproduce, from_scratch, experiments, report

## Depth Checklist

- [x] LSTM and GRU cell from scratch
- [x] Copy problem or long-range synthetic task
- [x] Gated vs vanilla RNN comparison
