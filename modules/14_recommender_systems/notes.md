# 笔记：推荐系统

## 核心问题

矩阵分解如何预测缺失的 user-item 评分？

## 协同过滤直觉

用户与物品由低维**隐因子**表示。用户 u 对物品 i 的评分近似为：

\[
\hat{r}_{ui} = P_u \cdot Q_i
\]

其中 \(P_u\) 是用户向量，\(Q_i\) 是物品向量。

## 训练

在观测到的评分上最小化平方误差 + L2 正则化，用 SGD 逐条更新 \(P\) 和 \(Q\)：

\[
P_u \leftarrow P_u + \eta \left( e_{ui} Q_i - \lambda P_u \right)
\]
\[
Q_i \leftarrow Q_i + \eta \left( e_{ui} P_u - \lambda Q_i \right)
\]

其中 \(e_{ui} = r_{ui} - P_u \cdot Q_i\) 是预测误差。

## 基线

**Popularity / 全局均值**：预测所有缺失值为训练集平均分。简单但忽略个性化。

## 本模块脚本

| 路径 | 作用 |
|------|------|
| `from_scratch/matrix_factorization.py` | SGD 矩阵分解核心 |
| `rating_data.py` | 合成评分矩阵生成 |
| `experiments/baseline_comparison.py` | 全局均值 vs MF 对照 |

```bash
python3 modules/14_recommender_systems/experiments/baseline_comparison.py
```

## 局限

- 本模块用合成数据演示机制
- 真实系统需处理冷启动、隐式反馈、大规模稀疏矩阵
- 小稀疏矩阵上 MF 未必优于简单基线，需更大规模数据验证
