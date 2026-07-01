# 04 卷积神经网络（CNN）

本模块从零实现 naive 2D 卷积与最大池化，与 PyTorch 数值对照；复现 LeNet 在 Fashion-MNIST 上的训练，并分析特征图与类别混淆模式。

## 核心知识点

- **局部连接与参数共享**：卷积核在整图滑动，参数量远小于同等感受野的全连接
- **输出尺寸**：\(L_{out} = \lfloor (L + 2P - K) / S \rfloor + 1\)
- **多通道卷积**：每个输出通道对所有输入通道卷积结果求和
- **最大池化**：降采样、增广平移不变性，典型 2×2 stride 2 使边长减半
- **特征层级**：浅层响应边缘/纹理，深层感受野更大、语义更强
- **LeNet 管线**：Conv → ReLU → Pool → Conv → Pool → FC 分类头
- **im2col 直觉**：感受野展开为列，卷积等价于矩阵乘 — GPU 高效实现的基础

## 项目产出

| 目录 | 文件 | 说明 |
|------|------|------|
| `from_scratch/` | `conv2d.py` | naive conv2d、max_pool2d、与 PyTorch 对照 |
| `from_scratch/` | `conv_backward.py` | 卷积反向传播概念说明（训练用 PyTorch autograd） |
| `reproduce/` | `lenet_fashion_mnist.py` | LeNet 训练，保存权重到 outputs |
| `experiments/` | `feature_maps.py` | 第一层卷积核与特征图可视化 |
| `experiments/` | `error_analysis.py` | 按类准确率与混淆样本 |
| 文档 | `notes.md`、`report.md` | 公式、架构与实验结论 |

## 如何运行

在仓库根目录 `/workspace` 下执行：

```bash
# 卷积前向 + PyTorch 数值对照
python modules/04_cnn/from_scratch/conv2d.py

# 卷积反向概念说明（打印笔记）
python modules/04_cnn/from_scratch/conv_backward.py

# 训练 LeNet（5 epoch，保存 lenet_weights.pt）
python modules/04_cnn/reproduce/lenet_fashion_mnist.py

# 实验：特征图可视化（建议先训练；无权重时用随机初始化 demo）
python modules/04_cnn/experiments/feature_maps.py

# 实验：错误分析（需先训练得到权重）
python modules/04_cnn/experiments/error_analysis.py
```

## 建议学习顺序

1. 阅读 `notes.md` — 手算 28×28、3×3、pad=1 的输出尺寸
2. 阅读 `from_scratch/conv2d.py` — 理解四重循环 naive 实现
3. 运行 `conv2d.py` — 确认与 `F.conv2d` 误差 < 1e-5
4. 阅读 `conv_backward.py` — 了解 input/kernel 梯度方向（不强制手写）
5. 运行 `lenet_fashion_mnist.py` — 获得 ~87% test acc（5 epoch）
6. 运行 `feature_maps.py` 与 `error_analysis.py` — 连接数值与可解释性
7. 阅读 `report.md` — 回顾 CNN vs MLP 与混淆类别

## 与前后模块的联系

- **承接 Module 03**：LeNet 后半段仍是 FC + ReLU；区别在前端用卷积提取空间特征
- **对比 MLP**：图像展平后喂 MLP 忽略邻域结构；CNN 显式编码平移等先验
- **后续 Module 05**：LeNet 是经典基线；现代 CNN（ResNet 等）在此基础上加深与残差连接
