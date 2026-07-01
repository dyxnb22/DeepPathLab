# DeepPath Lab 学习进度清单

复制本文件到你的笔记，或直接在仓库里勾选 `[x]` 记录进度。  
「完成」标准：读完 notes、跑通 from_scratch + 至少一个 experiment、能回答 notes 自检问题。

---

## 环境与全局

- [ ] 创建虚拟环境并 `pip install -r requirements.txt`
- [ ] `python3 scripts/verify_all.py` → 15/15 passed
- [ ] `python3 scripts/run_tests.py` → 全部通过
- [ ] 阅读 [getting-started.md](getting-started.md)

---

## Track A — 基础

### 01 Preliminaries & Autograd
- [ ] 阅读 `notes.md`（计算图、链式法则、梯度检查）
- [ ] 阅读 `from_scratch/value.py`
- [ ] 运行 `experiments/gradient_check.py`
- [ ] 能手算 `f = a*b + b*c` 时 `b.grad` 的两路贡献
- [ ] 阅读 `report.md`

---

## Track B — 深度学习核心

### 02 Linear Models
- [ ] 阅读 notes + `linear_regression.py` / `softmax_regression.py`
- [ ] 理解闭式解 vs 梯度下降
- [ ] 运行两个 from_scratch 脚本
- [ ] 阅读 report

### 03 Multilayer Perceptrons
- [ ] 阅读 notes + `mlp_numpy.py`（矩阵 backprop）
- [ ] 对照 `mlp_autograd.py`
- [ ] 运行 `experiments/` 中至少一个实验
- [ ] 阅读 report

### 09 Optimization
- [ ] 阅读 notes + `optimizers.py`（SGD / Momentum / Adam）
- [ ] 运行 `experiments/optimizer_comparison.py` 或 `lr_sensitivity.py`
- [ ] 能解释 Adam 的一阶、二阶矩估计
- [ ] 阅读 report

---

## Track D — 计算机视觉

### 04 Convolutional Neural Networks
- [ ] 阅读 notes + `conv2d.py`
- [ ] 理解输出尺寸公式与 padding
- [ ] 运行 LeNet 相关 reproduce
- [ ] 阅读 report

### 05 Modern CNN
- [ ] 阅读 notes + `residual_block.py`
- [ ] 理解 skip connection 缓解退化
- [ ] 运行 plain vs ResNet 实验
- [ ] 阅读 report

### 10 Computer Vision Applications
- [ ] 阅读 notes + `augmentation.py`
- [ ] 运行 transfer learning / augmentation 实验
- [ ] 阅读 report

---

## Track C — 序列与 Transformer

### 06 RNN
- [ ] 阅读 notes + `rnn.py`
- [ ] 理解 BPTT 与梯度消失直觉
- [ ] 运行字符级生成实验
- [ ] 阅读 report

### 07 LSTM & GRU
- [ ] 阅读 notes + `lstm_cell.py` / `gru_cell.py`
- [ ] 理解门控如何控制信息流
- [ ] 运行 copy problem 实验
- [ ] 阅读 report

### 08 Attention & Transformer
- [ ] 阅读 notes + `attention.py` / `transformer_block.py`
- [ ] 理解 scaled dot-product attention
- [ ] 运行 `experiments/attention_viz.py`
- [ ] 阅读 report

---

## Track E — NLP

### 11 NLP Pretraining
- [ ] 阅读 notes + `skipgram.py`
- [ ] 理解负采样与词向量几何
- [ ] 运行 embedding 可视化实验
- [ ] 阅读 report

### 12 NLP Applications
- [ ] 阅读 notes + `bow_classifier.py`
- [ ] 运行情感分类与 error analysis
- [ ] 阅读 report

### 13 NLP Fine-Tuning
- [ ] 阅读 notes + `finetune_loop.py`
- [ ] 理解 freeze / full fine-tune 边界
- [ ] 运行 `freeze_vs_finetune.py`
- [ ] 阅读 report

---

## Track F — 推荐与强化学习

### 14 Recommender Systems
- [ ] 阅读 notes + `matrix_factorization.py`
- [ ] 理解 MF 的 SGD 更新与稀疏 mask
- [ ] 运行 `baseline_comparison.py`
- [ ] 阅读 report

### 15 Reinforcement Learning
- [ ] 阅读 notes + `gridworld.py` / `q_learning.py`
- [ ] 能手写 Q-learning 更新式
- [ ] 运行 `epsilon_decay.py`
- [ ] 阅读 report

---

## 结业自检（15 模块后）

- [ ] 能画出从 01 到 13 的知识依赖简图
- [ ] 能向他人用 10 分钟讲清 autograd 与 backprop
- [ ] 能解释 Transformer 中 attention 在算什么
- [ ] 能说明何时用迁移学习、何时用全量微调
- [ ] 能区分协同过滤与 Q-learning 各自解决的问题

完成以上清单，即具备系统复习与面试口述的基础。
