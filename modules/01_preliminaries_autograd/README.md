# 01 预备知识与自动微分

本模块从标量计算图入手，理解自动微分（autograd）如何记录运算、沿链式法则反向传播梯度，并用数值检查验证实现正确性。

## 核心知识点

- **计算图**：每个运算产生节点，保存前向值 `data`、累积梯度 `grad`、子节点 `_prev` 与局部反向规则 `_backward`
- **链式法则**：\(\partial f / \partial w = (\partial f / \partial x) \cdot (\partial x / \partial w)\)，梯度沿图从输出向输入传递
- **逆拓扑序**：反向传播必须先处理下游节点，保证子节点收到完整上游梯度后再调用其 `_backward`
- **梯度累积**：同一变量经多条路径贡献梯度时，必须用 `+=` 累加，不能覆盖
- **局部导数规则**：加减乘、幂、ReLU、tanh 各有固定的反向公式，`_backward` 闭包需捕获前向时的操作数
- **数值梯度检查**：有限差分 \(\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}\) 是发现手写反向 bug 的可靠手段
- **标量 vs 张量**：本模块刻意限制为标量，为后续模块的矩阵 backprop 和框架 autograd 打直觉基础

## 项目产出

| 目录 | 文件 | 说明 |
|------|------|------|
| `from_scratch/` | `value.py` | 标量 autograd 引擎：`Value` 类，支持 `+`、`*`、幂、ReLU、tanh |
| `reproduce/` | `pytorch_autograd_check.py` | 与 PyTorch 标量梯度逐表达式对照 |
| `experiments/` | `gradient_check.py` | 有限差分验证三个典型表达式 |
| `experiments/` | `graph_viz.py` | 打印 forward/backward 拓扑序，观察共享节点 |
| 文档 | `notes.md`、`report.md` | 概念笔记与实验总结 |

## 如何运行

在仓库根目录 `/workspace` 下执行：

```bash
# 从零实现：快速 smoke test
python -c "import sys; sys.path.insert(0,'modules/01_preliminaries_autograd/from_scratch'); from value import Value; a,b=Value(2.),Value(3.); (a*b).relu().backward(); print('a.grad=',a.grad,'b.grad=',b.grad)"

# 复现对照：scratch vs PyTorch
python modules/01_preliminaries_autograd/reproduce/pytorch_autograd_check.py

# 实验：数值梯度检查
python modules/01_preliminaries_autograd/experiments/gradient_check.py

# 实验：计算图拓扑序可视化
python modules/01_preliminaries_autograd/experiments/graph_viz.py
```

## 建议学习顺序

1. 阅读 `notes.md` — 理解计算图、链式法则、逆拓扑序与梯度累积
2. 阅读 `from_scratch/value.py` — 对照笔记看 `_backward` 闭包如何写
3. 运行 `experiments/gradient_check.py` — 确认有限差分与 autograd 一致
4. 运行 `experiments/graph_viz.py` — 观察共享变量 `b` 的两路梯度如何相加
5. 运行 `reproduce/pytorch_autograd_check.py` — 与工业级实现对照
6. 阅读 `report.md` — 回顾检查点与常见踩坑

## 与前后模块的联系

- **前置**：无硬性依赖；熟悉 Python 与基本微积分即可
- **承接 Module 02**：线性模型的 MSE / 交叉熵梯度，本质是链式法则在矩阵形式下的批量应用
- **承接 Module 03**：`mlp_autograd.py` 用本模块的 `Value` 对单样本 MLP 构图，与手写矩阵 backprop 交叉验证
- **后续模块**：真实训练依赖 PyTorch autograd；本模块的价值是「看见」框架在背后做了什么
