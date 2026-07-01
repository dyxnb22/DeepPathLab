# 报告：Transfer Learning & Augmentation

## 目标

理解预训练特征迁移与数据增强对小样本图像分类的影响。

## 实验结果

### 迁移学习（5k 子集，6 epochs）

| 策略 | 最终 test acc |
|------|--------------|
| 从头训练 | 83.0% |
| 冻结特征 + 微调分类头 | **89.2%** |

预训练 LeNet 特征带来约 6 个百分点提升。

### 数据增强（3k 子集，5 epochs）

| 设置 | 最终 test acc |
|------|--------------|
| 无增强 | **71.9%** |
| 翻转 + 随机裁剪 | 69.3% |

小子集上增强未带来提升（可能因数据太少、增强过强）。更大规模训练中增强通常更有益。

## 深度检查点

- [x] 微调 vs 从头训练对照
- [x] 数据增强 ablation
- [x] 下游任务报告

## 收获

迁移学习在小数据场景价值明显；增强效果依赖数据规模和强度。工程上应先验证预训练迁移收益，再逐步加入增强并做 ablation。

## 核心知识点回顾

1. **迁移学习**：复用源域预训练特征，加速目标域收敛、提升小数据泛化
2. **特征提取 vs 微调**：冻结 backbone 只训头 vs 联合小 lr 更新
3. **数据增强**：训练时随机变换扩增多样性，推理时不使用随机增强
4. **域偏移**：预训练域与目标域越近，迁移收益越大
5. **实验结论**：5k 子集上冻结 LeNet 特征 + 微调头优于从头训练约 6pp

## 推荐复习命令

```bash
# 增强流水线 sanity check
python modules/10_computer_vision_applications/from_scratch/augmentation.py

# 迁移学习对照（需 torchvision，首次下载 Fashion-MNIST）
python modules/10_computer_vision_applications/reproduce/transfer_learning.py

# 数据增强 ablation
python modules/10_computer_vision_applications/experiments/augmentation_ablation.py
```

复习时对照 `notes.md` 自检问题，思考若将子集扩大到 50k，增强策略应如何调整。
