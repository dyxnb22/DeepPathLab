# 笔记：优化算法

## 核心问题

Adam 相比 SGD 在什么场景下更有优势？为什么？

## SGD（随机梯度下降）

批量大小为 \(B\) 时，用 mini-batch 梯度近似全量梯度：

\[
g_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla_\theta L(x_i, y_i)
\]
\[
\theta_{t+1} = \theta_t - \eta \, g_t
\]

**特点**：实现简单、内存占用低；在非凸地形上可能振荡，对学习率 \(\eta\) 非常敏感。

## Momentum

将梯度做指数移动平均，赋予更新「惯性」：

\[
v_t = \mu \, v_{t-1} + g_t
\]
\[
\theta_{t+1} = \theta_t - \eta \, v_t
\]

常见 \(\mu = 0.9\)。一致方向上的梯度会累积，反方向分量会部分抵消，从而加速收敛并抑制振荡。

Nesterov 动量变体先「预见」一步再计算梯度，本模块未实现，但思想类似。

## Adam

结合动量与自适应学习率（类似 RMSprop 的二阶矩）：

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{（一阶矩，动量）}
\]
\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{（二阶矩，梯度平方 EMA）}
\]

**偏差修正**（训练初期 \(m, v\) 偏向 0）：

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

\[
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

默认 \(\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}\)。

## 学习率与调度

- **固定 lr**：最简单；SGD 通常需手动搜索
- **学习率衰减**：\(\eta_t = \eta_0 / (1 + \alpha t)\) 或 step decay
- **Warmup**：大模型训练中先小 lr 再增大，本模块未涉及

## 实践观察

| 优化器 | 优点 | 缺点 |
|--------|------|------|
| SGD | 最终泛化有时更好；理论成熟 | 需仔细调 lr；收敛慢 |
| Momentum | 加速、减振 | 仍依赖全局 lr |
| Adam | 默认 lr 宽容；早期收敛快 | 极大 lr 仍会发散；部分任务泛化略差 |

本模块 spiral MLP 实验：Adam (lr=0.01) 约 99%，SGD (lr=0.5) 约 78%，Momentum (lr=0.1) 约 91%。

## 常见陷阱

1. **忘记偏差修正**：Adam 前几步若不做 \(\hat{m}, \hat{v}\) 校正，更新步长会偏小
2. **lr 与优化器不匹配**：对 SGD 有效的 lr=0.5 直接用于 Adam 可能发散
3. **混淆 weight decay 与 L2 正则**：Adam 中二者不等价；大模型常用 AdamW 解耦
4. **只看训练 loss**：优化器影响收敛路径，最终 test 指标才是选型依据
5. **batch size 改变未调 lr**：线性缩放规则 \(\eta \propto B\) 是经验法则，非恒成立

## 与前面模块的联系

- Module 02 已观察 lr 对线性模型的影响
- Module 03 MLP 反向传播产出梯度，本模块决定如何用梯度更新权重
- Module 04+ 所有 CNN / NLP 训练都依赖此处选择的优化器

## 自检问题

1. 写出 SGD、Momentum、Adam 的单步更新公式，并说明每个符号含义。
2. Adam 的 \(\hat{m}, \hat{v}\) 偏差修正在第 1 步、第 100 步分别有多大影响？
3. 为什么 Momentum 能抑制「之字形」振荡？
4. 若 loss 完全不下降，你会先检查 lr 过大还是过小？判断依据是什么？
5. 在什么场景下你会优先选 SGD+momentum 而非 Adam？
6. `from_scratch/optimizers.py` 中 `zero_state()` 何时必须调用？
