# 03 多层感知机（MLP）

本模块在 spiral 等非线性可分数据上实现两层 MLP，手写矩阵反向传播，并与 Module 01 的标量 autograd 交叉验证，比较 ReLU / sigmoid / tanh 的训练动态。

## 核心知识点

- **非线性必要性**：纯线性层堆叠仍是线性变换，必须插入激活函数才能拟合复杂边界
- **两层 MLP 形状**：\(X(N,D) \to z_1(N,H) \to a_1 \to z_2(N,C)\)，注意 batch 维始终保留
- **反向传播**：从 `dL/dz2` 出发，逐层链式传递；隐藏层需乘激活导数 `activation'(z1)`
- **激活函数权衡**：ReLU 计算快、缓解饱和；sigmoid/tanh 易在两端梯度趋零
- **双路径验证**：矩阵 backprop 高效训练；标量 autograd 对单样本展开图，核对每个权重梯度
- **梯度流**：饱和激活会使浅层 `W1` 梯度范数明显小于 ReLU，训练变慢
- **失败模式**：死 ReLU、过大学习率、差初始化导致部分神经元永不激活

## 项目产出

| 目录 | 文件 | 说明 |
|------|------|------|
| `from_scratch/` | `mlp_numpy.py` | 2 层 MLP + 手写 backprop，支持 ReLU/sigmoid/tanh |
| `from_scratch/` | `mlp_autograd.py` | 单样本标量图 vs numpy 梯度一致性检查 |
| `reproduce/` | `mlp_pytorch.py` | PyTorch MLP 基线对照 |
| `experiments/` | `activation_comparison.py` | 三种激活的准确率曲线 |
| `experiments/` | `gradient_flow.py` | 训练过程中各层梯度范数追踪 |
| 文档 | `notes.md`、`report.md` | 张量形状、计算图与实验结论 |

## 如何运行

在仓库根目录 `/workspace` 下执行：

```bash
# 矩阵版 MLP 训练（spiral 三分类）
python modules/03_mlp/from_scratch/mlp_numpy.py

# 标量 autograd vs numpy backprop 梯度对照
python modules/03_mlp/from_scratch/mlp_autograd.py

# PyTorch 基线对照
python modules/03_mlp/reproduce/mlp_pytorch.py

# 实验：激活函数对比（生成 activation_comparison.png）
python modules/03_mlp/experiments/activation_comparison.py

# 实验：梯度流追踪（生成 gradient_flow.png）
python modules/03_mlp/experiments/gradient_flow.py
```

## 建议学习顺序

1. 阅读 `notes.md` — 张量形状表与反向公式
2. 阅读 `from_scratch/mlp_numpy.py` 的 `forward` / `backward` — 对照笔记手推一遍
3. 运行 `mlp_numpy.py` — 确认 spiral 上 ~90% 准确率
4. 运行 `mlp_autograd.py` — 理解「两种实现，同一套链式法则」
5. 运行 `reproduce/mlp_pytorch.py` — 与框架对齐
6. 运行 `activation_comparison.py` 与 `gradient_flow.py` — 验证 ReLU vs sigmoid 假设
7. 阅读 `report.md` — 回顾失败模式与检查点

## 与前后模块的联系

- **承接 Module 02**：复用 softmax + 交叉熵头；spiral 数据暴露线性模型容量不足
- **承接 Module 01**：`mlp_autograd.py` 直接 import `Value`，是 autograd 引擎的「实战测验」
- **铺垫 Module 04**：图像若展平喂给 MLP 会丢失空间结构；CNN 用卷积保留局部性
