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

前向时 `cache` 保存 `X, z1, a1`，反向需要 `z1` 计算激活导数，需要 `a1` 计算 `dW2`。

## 反向传播逐层公式

设损失对 logits 的梯度为 `dL/dz2`（本仓库中为 `(probs - one_hot) / n`）：

```text
dL/dW2 = a1^T @ dL/dz2
dL/da1 = dL/dz2 @ W2^T
dL/dz1 = dL/da1 * activation'(z1)    # 逐元素乘（Hadamard）
dL/dW1 = X^T @ dL/dz1
```

关键是**链式法则的逐层传递**：每层的梯度依赖下一层的梯度和局部导数。`activation'(z1)` 在 ReLU 下是 `(z1 > 0)` 掩码，在 sigmoid 下是 `σ(z1)(1-σ(z1))`。

## 激活函数对比

| 激活 | 优点 | 缺点 |
|------|------|------|
| ReLU | 计算快，缓解梯度消失 | 可能「死 ReLU」（梯度恒为 0） |
| Sigmoid | 输出有界 | 两端饱和，梯度接近 0 |
| Tanh | 零中心 | 同样饱和问题 |

假设：在 spiral 数据上，ReLU 收敛更快；sigmoid 因饱和导致早期梯度更小。见 `activation_comparison.py` 与 `gradient_flow.py` 验证。

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

两条路径在 MSE-on-logits 损失下梯度应一致，说明 backprop 的本质就是链式法则的系统化应用。

## 常见踩坑

1. **忘记乘激活导数**：`grad_z1 = grad_a1 * relu_deriv(z1)` 漏写会导致隐藏层梯度全错
2. **矩阵维度转置**：`dW2 = a1.T @ grad_logits` 不是 `grad_logits.T @ a1`；batch 维在左
3. **对 logits 用了 softmax 梯度却忘了 softmax 前向**：交叉熵+softmax 组合梯度针对的是 logits，不是 probs
4. **死 ReLU**：大 lr + 差初始化使 `z1` 长期为负，该神经元梯度恒 0
5. **autograd 对照时损失不一致**：`mlp_autograd.py` 用 MSE-on-logits 而非 CE，以便标量图实现；对照时需两边用同一损失

## 自检问题

1. 写出 `dL/dW2` 的形状，并解释为何是 `a1.T @ dL/dz2` 而不是逐样本外积循环？
2. ReLU 网络里，若某个隐藏单元在整批数据上 `z1 <= 0`，它对损失的梯度贡献是多少？
3. 为何 spiral 数据上线性模型（Module 02）失败，而 2 层 MLP 可以成功？从「决策边界形状」回答。
