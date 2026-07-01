# 报告：Optimizer Comparison Lab

## 目标

从零实现 SGD/Momentum/Adam 更新规则，在 spiral MLP 上对比优化器与学习率敏感性。

## 实验结果

### 优化器对比（100 epochs）

| 优化器 | 学习率 | 最终准确率 |
|--------|--------|-----------|
| SGD | 0.5 | 78.0% |
| Momentum | 0.1 | 91.0% |
| Adam | 0.01 | **99.3%** |

Adam 在默认 lr 下收敛最快；SGD 需要更大 lr 和 momentum 辅助。

### 学习率敏感性

- SGD 在 lr=0.001–0.01 几乎不学习（~33%）
- Adam 在 lr=0.01–0.1 均表现良好（>97%）
- Adam 在 lr=0.5 时退化（45%），说明自适应方法也并非对极大 lr 免疫

## 深度检查点

- [x] SGD、Momentum、Adam 从零实现
- [x] 同模型不同优化器对照
- [x] 学习率敏感性实验

## 收获

优化器选择是训练工程的核心决策之一，应结合任务、模型和调参预算。没有万能优化器：Adam 适合快速实验与默认起点；SGD+momentum 在部分视觉任务上仍值得最终精调。

## 核心知识点回顾

1. **SGD**：\(\theta \leftarrow \theta - \eta g\)，简单但对 lr 敏感
2. **Momentum**：梯度 EMA 赋予惯性，加速一致方向、抑制振荡
3. **Adam**：一阶矩 + 二阶矩 + 偏差修正，自适应 per-parameter 步长
4. **lr 敏感性**：不同优化器的有效 lr 区间差异大，需对照实验而非照搬默认值
5. **工程结论**：先用 Adam 快速验证 pipeline，再视任务考虑 SGD 精调

## 推荐复习命令

```bash
# 单步更新 sanity check（理解三种更新差异）
python modules/09_optimization/from_scratch/optimizers.py

# 三优化器 spiral 分类对照
python modules/09_optimization/experiments/optimizer_comparison.py

# 学习率扫描（观察有效区间）
python modules/09_optimization/experiments/lr_sensitivity.py
```

复习时对照 `notes.md` 自检问题，尝试不看代码手推 Adam 的 \(\hat{m}, \hat{v}\) 校正。
