# 报告：Linear Model Playground

## 目标

实现线性回归和 softmax 分类，对比解析解、手写 GD 与 PyTorch 基线，理解损失函数和优化行为。

## 实现范围

- `from_scratch/linear_regression.py` — 正规方程 + 全批量 GD
- `from_scratch/softmax_regression.py` — 数值稳定 softmax + 交叉熵
- `reproduce/baseline.py` — PyTorch `nn.Linear` 对照
- `experiments/optimization_sweep.py` — 学习率扫描
- `experiments/decision_boundary.py` — 三分类决策边界可视化

## 实验结果

### 线性回归

- 闭式解与 GD（lr=0.1, 300 epochs）在合成数据上 MSE 一致（权重差 < 1e-3）
- PyTorch 基线 MSE 与 scratch GD 对齐

### 学习率扫描

| 学习率 | 行为 |
|--------|------|
| 0.001 | 收敛慢，300 epoch 后 loss 仍较高 |
| 0.01–0.1 | 稳定收敛 |
| 0.5 | 收敛但可能有振荡 |
| 1.0 | 发散，loss 爆炸 |

### Softmax 分类

三分类 blob 数据上准确率 > 95%。决策边界近似线性分割，符合模型容量预期。

## 深度检查点

- [x] 解析解与 GD 误差 < 阈值
- [x] scratch 与 PyTorch 对齐
- [x] 学习率实验有明确观察结论
- [x] 笔记解释了 softmax 数值稳定

## 核心知识点回顾

- 线性回归：MSE + 闭式解或 GD；梯度与残差成正比
- Softmax 回归：logits → 稳定 softmax → 交叉熵；logits 梯度为 \((p-y)/n\)
- 学习率是首个需要系统扫描的超参
- 线性决策边界无法刻画非线性可分数据（为 MLP 铺垫）

## 推荐复习命令

```bash
# 闭式解 vs GD 数值对齐
python modules/02_linear_models/from_scratch/linear_regression.py

# Softmax 分类终点指标
python modules/02_linear_models/from_scratch/softmax_regression.py

# PyTorch 对照
python modules/02_linear_models/reproduce/baseline.py

# 学习率敏感性（建议先看终端输出再看 png）
python modules/02_linear_models/experiments/optimization_sweep.py

# 决策边界可视化
python modules/02_linear_models/experiments/decision_boundary.py
```

## 收获

- 交叉熵 + softmax 的梯度形式 `(p - y)/n` 极大简化了实现
- 学习率是线性模型上最敏感的 hyperparameter
- 线性模型无法处理非线性可分数据（为 Module 03 铺垫）

## 下一步

在 spiral 数据上测试线性模型失败、MLP 成功的对比（见 Module 03）。
