# 笔记：优化算法

## 核心问题

Adam 相比 SGD 在什么场景下更有优势？

## SGD

\[
\theta \leftarrow \theta - \eta \nabla L
\]

简单直接，但对学习率敏感，在非凸地形上可能振荡。

## Momentum

累积历史梯度方向，加速一致方向上的更新，抑制振荡：

\[
v \leftarrow \mu v + \nabla L,\quad \theta \leftarrow \theta - \eta v
\]

## Adam

自适应学习率：为每个参数维护一阶矩（动量）和二阶矩（梯度平方的移动平均）：

\[
m \leftarrow \beta_1 m + (1-\beta_1)g,\quad v \leftarrow \beta_2 v + (1-\beta_2)g^2
\]
\[
\hat{m} = m/(1-\beta_1^t),\quad \hat{v} = v/(1-\beta_2^t),\quad \theta \leftarrow \theta - \eta \hat{m}/(\sqrt{\hat{v}}+\epsilon)
\]

## 实践观察

- **SGD**：需要仔细调 lr；momentum 通常帮助收敛
- **Adam**：默认 lr 往往更宽容，早期收敛快
- 没有万能优化器：最终泛化有时 SGD+momentum 更好（尤其 CV）

## 与前面模块的联系

Module 02 已观察 lr 对线性模型的影响；本模块在 MLP 上系统对比优化器。
