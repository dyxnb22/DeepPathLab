# 笔记：多层感知机与反向传播

## 为什么需要非线性

线性层的堆叠仍是线性变换：`W2(W1x) = (W2W1)x`。必须插入非线性激活函数，网络才能逼近非线性函数（通用近似定理）。

## 网络结构与张量形状

以 2 层 MLP 为例（batch=N, input=D, hidden=H, output=C）：

```text
X:      (N, D)
W1,b1:  (D, H), (H,)
z1:     (N, H)   = X @ W1 + b1
a1:     (N, H)   = activation(z1)
z2:     (N, C)   = a1 @ W2 + b2
```

## 反向传播逐层公式

设损失对 logits 的梯度为 `dL/dz2`：

```text
dL/dW2 = a1^T @ dL/dz2
dL/da1 = dL/dz2 @ W2^T
dL/dz1 = dL/da1 * activation'(z1)    # 逐元素乘
dL/dW1 = X^T @ dL/dz1
```

关键是**链式法则的逐层传递**：每层的梯度依赖下一层的梯度和局部导数。

## 激活函数对比

| 激活 | 优点 | 缺点 |
|------|------|------|
| ReLU | 计算快，缓解梯度消失 | 可能「死 ReLU」（梯度恒为 0） |
| Sigmoid | 输出有界 | 两端饱和，梯度接近 0 |
| Tanh | 零中心 | 同样饱和问题 |

假设：在 spiral 数据上，ReLU 收敛更快；sigmoid 因饱和导致早期梯度更小。

## 计算图（ASCII）

```text
Input(2) --> [W1,b1] --> ReLU --> [W2,b2] --> Softmax --> CE Loss
     ^            |                    |
     |            v                    v
     +-------- backward ------------+
```

## 双路径验证

- **矩阵 backprop**（`mlp_numpy.py`）：高效，用于实际训练
- **标量 autograd**（`mlp_autograd.py`）：对单样本构建完整标量图，验证矩阵梯度正确性

两条路径梯度一致，说明理解了 backprop 的本质就是链式法则的系统化应用。
