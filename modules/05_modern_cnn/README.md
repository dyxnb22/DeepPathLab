# 05 Modern CNN

Track D — 理解现代 CNN 架构设计取舍（AlexNet / VGG / ResNet）。

## Core Question

为什么 ResNet 的 skip connection 能缓解深层网络退化问题？

## 核心知识点

- **网络退化 vs 过拟合**：更深的 plain 网络训练误差反而升高，是优化困难而非单纯过拟合
- **残差学习**：学习 \(F(x) = H(x) - x\)，输出 \(y = F(x) + x\)，恒等映射只需令 \(F \approx 0\)
- **梯度高速公路**：\(\partial L/\partial x = \partial L/\partial y \cdot (\partial F/\partial x + 1)\)，skip 路径提供常数 +1 梯度通道
- **VGG 思想**：用多个 3×3 卷积堆叠替代大卷积核，在相同感受野下减少参数、增加非线性
- **公平对比**：控制参数量、stage 布局、训练超参后再比较架构优劣

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | 退化现象、skip 直觉、架构对比维度 |
| `from_scratch/residual_block.py` | NumPy 残差块前向 + 梯度高速公路演示 |
| `models.py` | PlainDeepCNN、SmallResNet、VGGMini |
| `reproduce/benchmark.py` | 三架构 Fashion-MNIST 训练基线 |
| `experiments/plain_vs_residual.py` | 同深度 plain vs residual 对照 + 曲线图 |
| `experiments/architecture_comparison.py` | 参数量、卷积层数、精度汇总表 |
| `report.md` | 实验结论与设计取舍总结 |

## 如何运行

在仓库根目录执行（需安装 `torch`、`torchvision`）：

```bash
# 1. 从零理解残差块与梯度高速公路
python modules/05_modern_cnn/from_scratch/residual_block.py

# 2. 三架构训练基线（Plain / ResNet / VGG-mini）
python modules/05_modern_cnn/reproduce/benchmark.py

# 3. Plain vs Residual 对照实验（生成对比曲线图）
python modules/05_modern_cnn/experiments/plain_vs_residual.py

# 4. 架构汇总对比表
python modules/05_modern_cnn/experiments/architecture_comparison.py
```

输出图像保存在 `outputs/05_modern_cnn/`。

## 建议学习顺序

1. 阅读 [D2L Ch.8 Modern CNNs](https://d2l.ai/chapter_convolutional-modern/index.html) 中 ResNet / VGG 相关章节
2. 通读 `notes.md`，理解退化现象与 skip connection 直觉
3. 运行 `from_scratch/residual_block.py`，观察残差与 plain 输出差异及梯度演示
4. 阅读 `models.py` 中三种架构的 stage 布局
5. 运行 `experiments/plain_vs_residual.py`，对照训练曲线
6. 阅读 `report.md`，记录你自己的结论

## 与前后模块的联系

- **前置 — Module 04 CNN**：卷积、池化、LeNet 是理解现代 CNN 的基础；本模块在相同数据集上堆叠更深结构
- **后续 — Module 06+**：ResNet 的残差思想会再次出现于 Transformer 的 Add&Norm；优化技巧（BatchNorm、学习率）在 Module 09 系统讨论
- **横向**：本模块侧重**架构设计**；Module 04 侧重**卷积算子本身**

## Workflow

- Read [D2L Ch.8 Modern CNNs](https://d2l.ai/chapter_convolutional-modern/index.html)
- Write original notes in [notes.md](notes.md)
- Reproduce baselines in [reproduce/](reproduce/)
- Implement core ideas in [from_scratch/](from_scratch/)
- Run ablations in [experiments/](experiments/)
- Summarize in [report.md](report.md)

## Depth Checklist

- [x] Plain vs residual network training curves on same dataset
- [x] Notes explain skip connection gradient highway intuition
- [x] Architecture comparison table (params, depth, accuracy)
- [x] Report records which design choices mattered most
