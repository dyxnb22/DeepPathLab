# 报告：LeNet-style CNN Lab

## 目标

从零实现卷积和池化，复现 LeNet 在 Fashion-MNIST 上的分类，分析特征图和错误模式。

## 实现范围

- `from_scratch/conv2d.py` — naive conv2d + max_pool2d，与 PyTorch 数值对照
- `from_scratch/conv_backward.py` — 反向传播概念笔记
- `reproduce/lenet_fashion_mnist.py` — LeNet 训练
- `experiments/feature_maps.py` — 卷积核与特征图可视化
- `experiments/error_analysis.py` — 按类别准确率与混淆分析

## 实验结果

### Conv 正确性

`conv2d_multi` 与 `torch.nn.functional.conv2d` 在随机输入上最大误差 < 1e-5。

### LeNet 训练

5 epoch Adam 训练后 test accuracy 达到 **87.5%**（5 epochs）。

### 错误分析

- Shirt / Pullover / Coat 等上衣类别容易混淆（纹理相似）
- 鞋类（Sandal vs Sneaker vs Ankle boot）偶有混淆
- 特征图显示第一层主要响应边缘和局部纹理

## 深度检查点

- [x] conv2d 与 PyTorch 一致
- [x] 能手算输出尺寸（见 notes 公式）
- [x] LeNet 达到合理 baseline
- [x] 错误分析回答类别混淆问题

## 核心知识点回顾

- 卷积 = 局部连接 + 权重共享；输出尺寸由 L、K、S、P 决定
- 多通道输出 = 对各输入通道卷积后沿通道求和
- MaxPool 降采样；LeNet = 两段 Conv-Pool + FC 头
- 前向可手写验证；完整 conv backward 复杂，生产训练交给 autograd
- 错误模式反映类间视觉相似性，而不仅是「准确率一个数」

## 推荐复习命令

```bash
# 前向数值对照（应 PASSED）
python modules/04_cnn/from_scratch/conv2d.py

# 反向概念速览
python modules/04_cnn/from_scratch/conv_backward.py

# 训练并保存权重（后续实验依赖）
python modules/04_cnn/reproduce/lenet_fashion_mnist.py

# 第一层核与特征图
python modules/04_cnn/experiments/feature_maps.py

# 按类准确率与混淆样本
python modules/04_cnn/experiments/error_analysis.py
```

## CNN vs MLP

同等参数量下，CNN 利用空间结构先验，在图像任务上远优于展平后的 MLP。本模块未实现 MLP 对照实验，但 Module 03 的 MLP 在图像上表现差是预期结论。

## 局限

- 未手写 conv backward（进阶项）
- LeNet 是经典基线，现代 CNN（ResNet 等）在 Module 05 继续

## 下一步

Module 05 Modern CNN：对比 AlexNet/VGG/ResNet 设计取舍。
