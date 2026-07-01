# 笔记：卷积神经网络

## 卷积 vs 全连接

全连接层每个输出连接所有输入，参数量 O(H×W×C_in×C_out)。卷积利用两个关键假设：

1. **局部连接**：每个输出只依赖输入的局部区域（感受野）
2. **参数共享**：同一卷积核在整张图上滑动，大幅减少参数量

直觉：识别「边缘」的规则在图左上与右下应相同，无需为每个位置学独立权重。

## 输出尺寸公式

对于边长 `L`、卷积核 `K`、步幅 `S`、填充 `P`：

\[
L_{out} = \lfloor (L + 2P - K) / S \rfloor + 1
\]

例：28×28 输入，3×3 kernel，stride=1，padding=1 → 输出 28×28（same 卷积）。

手算练习：LeNet 第一层 `Conv(1→6, 5×5, pad=2)` 在 28×28 上输出仍为 28×28，再经 2×2 pool stride 2 → 14×14。

## 池化

Max pooling 取局部窗口最大值，降低空间分辨率、增加平移不变性。典型 2×2 pool stride 2 将 H、W 各减半。反向时梯度只回传到窗口内最大值位置（本模块未手写 pool backward）。

## 特征图

第一层卷积核常学到边缘、纹理等低级特征。堆叠多层后，感受野增大，可捕获更复杂的模式。`feature_maps.py` 可视化训练后第一层响应，便于与「边缘检测」直觉对照。

## LeNet 架构

```text
Input(1,28,28)
  -> Conv(6,5x5) -> ReLU -> MaxPool
  -> Conv(16,5x5) -> ReLU -> MaxPool
  -> Flatten -> FC(120) -> FC(84) -> FC(10)
```

第二段卷积无 padding，空间尺寸会缩小；接 FC 前需 `Flatten`，参数量集中在全连接层。

## 卷积反向传播（概念）

完整 conv backward 较复杂，核心思想：

- 对输入的梯度：用翻转的 kernel 对 grad_output 做「全卷积」（full convolution）
- 对 kernel 的梯度：输入 patch 与 grad_output 的外积累加

实际训练使用 PyTorch autograd；`from_scratch/conv2d.py` 验证前向正确性。详见 `conv_backward.py`。

## im2col 直觉

将每个感受野展开为一列，卷积变为矩阵乘法 `Y = X_col @ W`。GPU 上这是高效实现的基础；naive 四重循环便于学习，不适合大规模训练。

## 常见踩坑

1. **通道维顺序**：PyTorch 为 NCHW；本模块 `conv2d_multi` 用 `(C, H, W)`，对照时注意维度假设
2. **padding 与输出尺寸**：漏 pad 会导致特征图过快缩小，后续 FC 输入维度对不上
3. **未训练就跑 feature_maps**：脚本会警告并用随机权重，特征图无语义；应先跑 `lenet_fashion_mnist.py`
4. **混淆上衣类别**：Shirt / Pullover / Coat 纹理相似，是数据与模型容量共同导致，不一定是实现 bug
5. **把 conv 当「小全连接」心算**：应用输出尺寸公式，不要凭感觉猜 flatten 后的长度

## 自检问题

1. 28×28、kernel 5×5、stride 1、padding 0，输出边长是多少？若 padding=2 呢？
2. 为何多通道卷积要对每个输入通道分别卷积再求和，而不是一个大核？
3. Fashion-MNIST 上 LeNet 约 87%（5 epoch）时，哪些类别对最容易混淆？从「视觉相似性」解释一条。

## 与 Module 03 的对比

同等参数量下，CNN 利用空间结构先验，在图像任务上远优于展平后的 MLP。本模块未实现 MLP 对照实验，但 Module 03 的 MLP 在图像上表现差是预期结论。
