# 09 Optimization

Track B — 跨模块优化器对比实验。

## 核心问题

Adam 相比 SGD 在什么场景下更有优势？为什么？

## 核心知识点

- **SGD**：沿负梯度方向更新参数，简单但对学习率敏感
- **Momentum**：指数移动平均梯度方向，加速一致更新、抑制振荡
- **Adam**：为每个参数维护一阶矩与二阶矩，自适应缩放步长
- **偏差修正**：Adam 在训练早期对 \(m, v\) 做 \(\hat{m}, \hat{v}\) 校正，避免初始偏置
- **学习率敏感性**：不同优化器对 lr 的容忍区间不同，需结合任务调参
- **泛化 vs 收敛速度**：Adam 常收敛更快；SGD+momentum 在部分 CV 任务上最终泛化更好

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | 优化器公式、常见陷阱、自检问题 |
| `from_scratch/optimizers.py` | numpy 版 SGD / Momentum / Adam |
| `reproduce/optimizer_benchmark.py` | PyTorch 优化器对照入口 |
| `experiments/optimizer_comparison.py` | spiral MLP 上三优化器对比 |
| `experiments/lr_sensitivity.py` | 学习率扫描实验 |
| `report.md` | 实验结论与复习命令 |

## 如何运行

```bash
# 从零实现：单步更新 sanity check
python modules/09_optimization/from_scratch/optimizers.py

# 优化器对比（spiral 分类，约 1–2 分钟）
python modules/09_optimization/experiments/optimizer_comparison.py

# 学习率敏感性扫描
python modules/09_optimization/experiments/lr_sensitivity.py

# reproduce 入口（转发到 optimizer_comparison）
python modules/09_optimization/reproduce/optimizer_benchmark.py
```

依赖：`numpy`（from_scratch）、`torch`（experiments）。图表输出到 `outputs/09_optimization/`。

## 建议学习顺序

1. 阅读 `notes.md`，手推 SGD → Momentum → Adam 更新式
2. 运行 `from_scratch/optimizers.py`，观察单步参数变化
3. 阅读 `from_scratch/optimizers.py` 源码，对照公式
4. 运行 `optimizer_comparison.py`，记录各优化器最终准确率
5. 运行 `lr_sensitivity.py`，理解 lr 区间差异
6. 阅读 `report.md`，总结「何时选 Adam、何时选 SGD」

## 模块联系

- **← Module 02**：线性模型中已观察学习率对收敛的影响
- **← Module 03**：MLP 训练依赖梯度下降，本模块系统对比更新规则
- **→ Module 04–10**：所有后续训练实验都需选择优化器与 lr
- **→ Module 11–12**：NLP 预训练与下游任务默认常用 Adam / AdamW
