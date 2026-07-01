# 报告：Mini Autograd Engine

## 目标

实现一个标量自动微分引擎，理解计算图、链式法则、逆拓扑序和梯度累积。

## 实现范围

- **从零实现**：`from_scratch/value.py` — `Value` 类，支持 `+`, `*`, 幂、ReLU、tanh
- **复现对照**：`reproduce/pytorch_autograd_check.py` — 与 PyTorch 标量梯度对比
- **实验**：`experiments/gradient_check.py`（数值梯度检查）、`experiments/graph_viz.py`（图结构打印）

## 实验结果

### 梯度检查

在三个表达式上运行有限差分验证，相对误差均 < 1e-5：

1. `relu(a*b + c)` — 基本链式传播
2. `a*b + b*c` — 共享变量 `b` 的梯度累积
3. `relu(x**2)` — 复合运算

### PyTorch 对照

相同表达式下，scratch 与 PyTorch 梯度一致。

### 图可视化

`graph_viz.py` 打印了 forward/backward 拓扑序。观察到 `b` 在 `a*b` 和 `f*b` 两条路径上各贡献一次梯度，最终 `b.grad` 为两路之和。

## 深度检查点

- [x] 共享变量梯度正确累加
- [x] gradient_check 相对误差 < 1e-5
- [x] 能 trace 3 层表达式的 backward 顺序
- [x] 笔记解释了逆拓扑序的必要性

## 收获

- `_backward` 闭包必须捕获正确的局部变量（如 `other.data`），否则梯度会引用错误的值
- 梯度累积用 `+=` 是关键细节，共享节点场景下容易遗漏
- 有限差分检查是验证实现的必要步骤，不能跳过

## 未解问题

- 如何扩展到张量而不爆炸式增加代码复杂度？
- 内存效率：当前保留整个图，训练时需要 `zero_grad()` 和重新构图

## 下一步

Module 02 将用 NumPy 实现线性模型训练；可选地用本引擎做单样本 backward 对照。
