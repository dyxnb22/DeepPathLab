# 笔记：现代 CNN 架构

## 核心问题

为什么 ResNet 的 skip connection 能缓解深层网络退化问题？

## 网络退化 vs 过拟合

「退化」指：**更深的 plain 网络训练误差反而更高**，不是过拟合（训练误差也高）。这说明优化器难以学习恒等映射，深层堆叠反而破坏了已有特征。

| 现象 | 训练误差 | 测试误差 | 典型原因 |
|------|----------|----------|----------|
| 过拟合 | 低 | 高 | 容量过大、正则不足 |
| 退化 | 高 | 高 | 优化困难、梯度传播受阻 |

## Skip Connection 的直觉

残差块学习 **F(x) = H(x) - x**，输出为 **y = F(x) + x**。

形式化：

\[
y = \mathcal{F}(x, \{W_i\}) + x
\]

- 若最优映射接近恒等，只需令 \(\mathcal{F}(x) \approx 0\)，比直接学 \(H(x) = x\) 更容易（权重可初始化为接近零）
- 反向传播时：

\[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left( \frac{\partial \mathcal{F}}{\partial x} + \mathbf{1} \right)
\]

这就是「梯度高速公路」：即使 \(\partial \mathcal{F}/\partial x\) 很小，梯度仍可通过 skip 回传。

### 与 Module 04 卷积的关系

残差块内部仍是 conv-bn-relu 堆叠；skip 改变的是**信息通路**，不是卷积算子本身。

## VGG 的设计

用小卷积核（3×3）堆叠替代大卷积核：

\[
\text{感受野}_{2 \times 3\times3} = \text{感受野}_{1 \times 5\times5} = 5
\]

但两层 3×3 的参数量 \(2 \times 3^2 C^2 = 18C^2\)，远小于一层 5×5 的 \(25C^2\)，且中间多一层 ReLU。

## AlexNet 的启示（简述）

更深、更宽 + ReLU + Dropout + GPU 训练，开启了深度学习在 ImageNet 上的突破。本模块用更小的 Fashion-MNIST 做架构对比实验。

## 架构对比维度

| 维度 | 说明 |
|------|------|
| 参数量 | 模型容量与过拟合风险 |
| 深度 | 卷积层数，影响感受野和优化难度 |
| 训练稳定性 | loss/acc 曲线是否平滑收敛 |
| 测试精度 | 最终泛化表现 |

## Plain vs Residual

同深度下：

- **PlainDeepCNN**：连续 conv-bn-relu 堆叠，无 skip
- **SmallResNet**：相同 stage 布局，用 ResidualBlock 替代

公平对比应控制：通道数、stage 数、训练超参一致。

## 从零实现要点

`from_scratch/residual_block.py` 用 numpy 演示：

```text
y_residual = relu(conv2(relu(conv1(x))) + x)
y_plain    = relu(conv2(relu(conv1(x))))
```

并展示梯度高速公路上「+1」项的作用。

## 常见误区

1. **「ResNet 一定比 plain 更准」** — 优势在**足够深**的网络才明显；浅层网络上 plain 可能持平甚至略优（本模块实验即如此）
2. **混淆退化与过拟合** — 退化看训练误差是否随深度恶化，不是测试集表现
3. **对比不公平** — 参数量或训练轮数不一致时，结论不可信
4. **忽视 BatchNorm** — 现代 CNN 训练稳定性部分来自 BN；本模块 PyTorch 模型含 BN，但 `from_scratch/` 未实现

## 局限

- 本模块网络规模远小于 ImageNet 级 ResNet-50
- BatchNorm 包含在 PyTorch 模型中，未从零实现
- Fashion-MNIST 较简单，退化现象可能不如 CIFAR/ImageNet 明显

## 自检问题

1. 写出残差块的输出公式，并解释为什么学习 \(\mathcal{F}(x) \approx 0\) 等价于近似恒等映射。
2. 在反向传播中，skip connection 为 \(\partial L/\partial x\) 贡献了哪一项？若该项不存在会发生什么？
3. 本模块实验中 plain 网络略优于 ResNet，你如何解释？在什么条件下预期 ResNet 会反超？
