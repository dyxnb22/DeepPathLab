# 10 Computer Vision Applications

Track D — 迁移学习与视觉下游任务。

## 核心问题

预训练 CNN 特征如何迁移到新任务？

## 核心知识点

- **迁移学习**：利用源域预训练权重加速目标域学习
- **特征提取**：冻结 backbone，只训练分类头；适合小数据集、低算力
- **微调（Fine-tuning）**：解冻部分或全部层，用小学习率继续训练
- **数据增强**：随机翻转、裁剪、颜色抖动等，训练时扩增多样性、减轻过拟合
- **域偏移**：预训练域与目标域越接近，迁移收益越大
- **增强强度**：过强增强在小数据集上可能伤害性能

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | 迁移策略、增强公式直觉、陷阱、自检问题 |
| `from_scratch/augmentation.py` | numpy 版翻转 / 裁剪 / 归一化流水线 |
| `reproduce/transfer_learning.py` | Fashion-MNIST 微调 vs 从头训练 |
| `experiments/augmentation_ablation.py` | 有无增强的对照实验 |
| `report.md` | 实验结论与复习命令 |

## 如何运行

```bash
# 从零实现：增强流水线 sanity check
python modules/10_computer_vision_applications/from_scratch/augmentation.py

# 迁移学习：冻结特征 + 微调分类头 vs 从头训练
python modules/10_computer_vision_applications/reproduce/transfer_learning.py

# 数据增强 ablation（3k 子集）
python modules/10_computer_vision_applications/experiments/augmentation_ablation.py
```

依赖：`numpy`（from_scratch）、`torch` + `torchvision`（reproduce / experiments）。首次运行会下载 Fashion-MNIST。图表输出到 `outputs/10_computer_vision_applications/`。

## 建议学习顺序

1. 阅读 `notes.md`，理解特征提取 vs 微调 vs 从头训练
2. 运行 `from_scratch/augmentation.py`，观察增强前后张量形状与数值范围
3. 阅读 `augmentation.py` 源码，理解各变换的几何含义
4. 运行 `transfer_learning.py`，对比两种策略的 test acc
5. 运行 `augmentation_ablation.py`，思考增强在本实验中的效果
6. 阅读 `report.md`，总结小数据场景下的实践建议

## 模块联系

- **← Module 04**：LeNet 是本模块迁移的 backbone 来源
- **← Module 05**：更深 CNN（ResNet 等）是工业界常用预训练骨干
- **← Module 09**：微调时通常用小 lr + Adam/SGD，优化器选择影响收敛
- **→ Module 11–12**：NLP 中的 fine-tuning 思想与视觉迁移平行
- **→ Module 13**：大模型 NLP fine-tuning 是同一范式的延伸
