# 报告：Tiny MLP Trainer

## 目标

实现 2 层 MLP 的手写反向传播，与标量 autograd 对照，比较激活函数和梯度流行为。

## 实现范围

- `from_scratch/mlp_numpy.py` — 矩阵版 forward/backward，支持 ReLU/sigmoid/tanh
- `from_scratch/mlp_autograd.py` — 单样本标量图梯度验证
- `reproduce/mlp_pytorch.py` — PyTorch 基线
- `experiments/activation_comparison.py` — 激活函数对比
- `experiments/gradient_flow.py` — 各层梯度范数追踪

## 实验结果

### Spiral 分类

- Scratch MLP (ReLU, H=32): ~91% accuracy
- PyTorch 基线: 相当
- 线性模型（Module 02）在同一数据上无法有效分类

### 激活对比

| 激活 | 最终准确率 | 观察 |
|------|-----------|------|
| ReLU | ~0.91 | 收敛快，稳定 |
| Sigmoid | ~0.85 | 较慢，早期梯度小 |
| Tanh | ~0.88 | 介于两者之间 |

假设「ReLU 收敛更快」得到支持。

### 梯度流

Sigmoid 网络 W1 层梯度范数明显小于 ReLU，印证饱和导致的梯度缩小。

### 双路径验证

`mlp_autograd.py` 在 2-2-3 小网络上，所有参数梯度与 numpy backprop 一致。

## 深度检查点

- [x] numpy 与 autograd 梯度一致
- [x] MLP 在 spiral 上优于线性模型
- [x] 激活对比有先验假设并验证
- [x] 笔记含张量形状和计算图

## 核心知识点回顾

- 隐藏层激活是非线性的来源；无激活则深层等价单层线性
- backprop = 缓存前向中间量 + 从 loss 逐层反传，隐藏层必乘 `activation'(z)`
- ReLU 通常比 sigmoid 更适合默认选择；饱和激活会压缩浅层梯度
- 标量 autograd 与矩阵 backprop 应给出相同梯度 — 这是理解「框架在做什么」的锚点

## 推荐复习命令

```bash
# 主训练脚本
python modules/03_mlp/from_scratch/mlp_numpy.py

# 梯度一致性（应打印 all gradients match）
python modules/03_mlp/from_scratch/mlp_autograd.py

# PyTorch 对照
python modules/03_mlp/reproduce/mlp_pytorch.py

# 激活对比 + 曲线图
python modules/03_mlp/experiments/activation_comparison.py

# W1/W2 梯度范数随 epoch 变化
python modules/03_mlp/experiments/gradient_flow.py
```

## 失败模式

- **死 ReLU**：大学习率 + 差初始化可能导致部分神经元永不激活
- **Sigmoid 饱和**：深层网络中梯度接近 0，训练停滞

## 下一步

Module 04 将引入卷积的局部连接和参数共享，在图像数据上进一步验证表征学习能力。
