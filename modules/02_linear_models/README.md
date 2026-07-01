# 02 线性模型

本模块在合成数据上实现线性回归与 Softmax 多类分类，对比解析解、手写梯度下降与 PyTorch 基线，理解损失函数、优化与学习率敏感性。

## 核心知识点

- **线性回归**：\( \hat{y} = Xw + b \)，用 MSE 衡量连续预测误差
- **正规方程**：\(\hat{\theta} = (X^T X)^{-1} X^T y\) 给出闭式最优解，特征不多时精确且快
- **梯度下降**：迭代更新 \(w \leftarrow w - \eta \nabla_w L\)，可推广到无闭式解的问题
- **Softmax 回归**：logits \(z = XW + b\) 经 softmax 得类别概率，损失为交叉熵
- **数值稳定 softmax**：减去 \(\max(z)\) 再取 exp，不改变结果但避免溢出
- **交叉熵 + softmax 梯度**：对 logits 的梯度简化为 \((p - y_{\text{one-hot}}) / n\)
- **学习率**：线性模型上最敏感的超参；过大发散，过小收敛慢

## 项目产出

| 目录 | 文件 | 说明 |
|------|------|------|
| `from_scratch/` | `linear_regression.py` | 正规方程 + 全批量 GD |
| `from_scratch/` | `softmax_regression.py` | 稳定 softmax + 交叉熵分类 |
| `reproduce/` | `baseline.py` | PyTorch `nn.Linear` 对照 |
| `experiments/` | `optimization_sweep.py` | 学习率扫描，输出 loss 曲线图 |
| `experiments/` | `decision_boundary.py` | 三分类决策边界可视化 |
| 文档 | `notes.md`、`report.md` | 概念笔记与实验总结 |

## 如何运行

在仓库根目录 `/workspace` 下执行：

```bash
# 线性回归：闭式解 vs GD
python modules/02_linear_models/from_scratch/linear_regression.py

# Softmax 分类
python modules/02_linear_models/from_scratch/softmax_regression.py

# 复现对照：scratch vs PyTorch
python modules/02_linear_models/reproduce/baseline.py

# 实验：学习率扫描（生成 outputs/02_linear_models/lr_sweep.png）
python modules/02_linear_models/experiments/optimization_sweep.py

# 实验：决策边界图（生成 outputs/02_linear_models/decision_boundary.png）
python modules/02_linear_models/experiments/decision_boundary.py
```

## 建议学习顺序

1. 阅读 `notes.md` — 回归 vs 分类、正规方程、softmax 稳定化
2. 阅读并运行 `from_scratch/linear_regression.py` — 对照 MSE 梯度公式
3. 阅读并运行 `from_scratch/softmax_regression.py` — 理解 \((p-y)/n\) 从何而来
4. 运行 `reproduce/baseline.py` — 确认与 PyTorch 对齐
5. 运行 `experiments/optimization_sweep.py` — 观察 lr=1.0 发散、0.01–0.1 稳定
6. 运行 `experiments/decision_boundary.py` — 看线性分类器的决策边界形状
7. 阅读 `report.md` — 汇总数值结论

## 与前后模块的联系

- **承接 Module 01**：GD 的每一步都是链式法则；可用标量 `Value` 对单样本 MSE 验证手写梯度
- **铺垫 Module 03**：线性模型在 spiral 等非线性可分数据上会失败，说明需要隐藏层与非线性激活
- **后续 CNN/NLP**：softmax + 交叉熵是分类头的标准组合，本模块是其最简形态
