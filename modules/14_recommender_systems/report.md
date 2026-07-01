# 报告：Recommender Systems

## 目标

从零实现矩阵分解，对比全局均值基线。

## 实验结果

合成评分矩阵（50 users × 40 items，25% 密度）：

| 方法 | Test MSE |
|------|----------|
| Popularity (global mean) | 0.781 |
| Matrix Factorization | 0.928 |

MF 训练 loss 持续下降，但在此稀疏小数据集上未超越简单基线。更大规模数据和调参后 MF 通常能捕获个性化偏好。

## 深度检查点

- [x] MF SGD 从零实现
- [x] 基线对照
- [x] 训练曲线记录

## 收获

推荐系统的核心是 user/item 隐向量分解；工程上还需处理冷启动与隐式反馈。
