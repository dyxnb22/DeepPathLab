# 报告：Modern CNN Benchmark

## 目标

对比 Plain 深层 CNN、Small ResNet 和 VGG-mini，理解 skip connection 与现代架构设计。

## 实现范围

- `from_scratch/residual_block.py` — numpy 残差块前向 + 梯度高速公路演示
- `models.py` — PlainDeepCNN、SmallResNet、VGGMini
- `experiments/plain_vs_residual.py` — 同深度对照训练
- `reproduce/benchmark.py` — 三架构训练脚本

## 实验结果

### Plain vs Residual（5 epochs, Fashion-MNIST）

| 模型 | 参数量 | 最终 test acc |
|------|--------|---------------|
| PlainDeepCNN | 61,194 | **0.9088** |
| SmallResNet | 61,802 | 0.8856 |

### 观察

- 在本模块的网络深度（约 5 个 block）和 Fashion-MNIST 上，**两者均能收敛**，未出现严重的「深度退化」
- Plain 网络在此规模下甚至略高，说明 ResNet 的优势在**更深网络**（如 ResNet-50 on ImageNet）上更明显
- ResNet 的训练曲线在 epoch 2 时 test acc 有短暂下探，后期恢复，整体与 plain 接近

### 梯度高速公路（numpy 演示）

残差路径使 ∂L/∂x 包含常数 +1 项，当 ∂F/∂x 很小时仍可回传梯度。这是 ResNet 能训练极深网络的理论基础。

## 深度检查点

- [x] Plain vs residual 训练曲线已记录
- [x] 笔记解释 skip connection 直觉
- [x] 架构参数量对比
- [x] 报告诚实记录实验结论（包括 plain 略优的情况）

## 收获

- 架构优势是**有条件的**：深度、数据集、超参都会影响结论
- 公平对比需要匹配参数量、stage 布局和训练设置
- VGG 的 3×3 堆叠与 ResNet 的 skip 解决的是不同问题（参数效率 vs 优化难度）

## 核心知识点回顾

- **退化**：更深 plain 网络训练误差升高，是优化问题而非单纯过拟合
- **残差学习**：\(y = \mathcal{F}(x) + x\)，恒等映射对应 \(\mathcal{F} \approx 0\)
- **梯度高速公路**：skip 路径为 \(\partial L/\partial x\) 提供 +1 项，使极深网络可训练
- **VGG**：3×3 堆叠在相同感受野下参数更少、非线性更多
- **架构对比需公平**：参数量、深度、超参一致才有意义

## 推荐复习命令

```bash
python modules/05_modern_cnn/from_scratch/residual_block.py
python modules/05_modern_cnn/experiments/plain_vs_residual.py
```

## 下一步

Module 06 RNN：理解序列递推与 BPTT 梯度消失。
