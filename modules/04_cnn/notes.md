# 笔记：卷积神经网络

## 卷积 vs 全连接

全连接层每个输出连接所有输入，参数量 O(H×W×C_in×C_out)。卷积利用两个关键假设：

1. **局部连接**：每个输出只依赖输入的局部区域（感受野）
2. **参数共享**：同一卷积核在整张图上滑动，大幅减少参数量

## 输出尺寸公式

对于边长 `L`、卷积核 `K`、步幅 `S`、填充 `P`：

\[
L_{out} = \lfloor (L + 2P - K) / S \rfloor + 1
\]

例：28×28 输入，3×3 kernel，stride=1，padding=1 → 输出 28×28。

## 池化

Max pooling 取局部窗口最大值，降低空间分辨率、增加平移不变性。典型 2×2 pool stride 2 将尺寸减半。

## 特征图

第一层卷积核常学到边缘、纹理等低级特征。堆叠多层后，感受野增大，可捕获更复杂的模式。

## LeNet 架构

```text
Input(1,28,28)
  -> Conv(6,5x5) -> ReLU -> MaxPool
  -> Conv(16,5x5) -> ReLU -> MaxPool
  -> Flatten -> FC(120) -> FC(84) -> FC(10)
```

## 卷积反向传播（概念）

完整 conv backward 较复杂，核心思想：

- 对输入的梯度：用翻转的 kernel 对 grad_output 做「全卷积」
- 对 kernel 的梯度：输入 patch 与 grad_output 的外积

实际训练使用 PyTorch autograd；`from_scratch/conv2d.py` 验证前向正确性。

## im2col 直觉

将每个感受野展开为一列，卷积变为矩阵乘法 `Y = X_col @ W`。GPU 上这是高效实现的基础。
