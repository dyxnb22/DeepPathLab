# 笔记：现代 CNN 架构

## 核心问题

为什么 ResNet 的 skip connection 能缓解深层网络退化问题？

## 网络退化 vs 过拟合

「退化」指：**更深的 plain 网络训练误差反而更高**，不是过拟合（训练误差也高）。这说明优化器难以学习恒等映射，深层堆叠反而破坏了已有特征。

## Skip Connection 的直觉

残差块学习 **F(x) = H(x) - x**，输出为 **y = F(x) + x**。

- 若最优映射接近恒等，只需令 F(x) ≈ 0，比直接学 H(x) = x 更容易
- 反向传播时：∂L/∂x = ∂L/∂y · (∂F/∂x + **1**)，skip 路径提供常数 1 的梯度通道

这就是「梯度高速公路」：即使 ∂F/∂x 很小，梯度仍可通过 skip 回传。

## VGG 的设计

用小卷积核（3×3）堆叠替代大卷积核：

- 相同感受野，参数量更少
- 更多非线性层（更多 ReLU）

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

## 局限

- 本模块网络规模远小于 ImageNet 级 ResNet-50
- BatchNorm 包含在 PyTorch 模型中，未从零实现
- Fashion-MNIST 较简单，退化现象可能不如 CIFAR/ImageNet 明显
