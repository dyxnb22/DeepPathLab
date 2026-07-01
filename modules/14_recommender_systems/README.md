# 14 推荐系统

Track F — 协同过滤与矩阵分解。

## 核心问题

矩阵分解如何预测缺失的 user-item 评分？

## 核心知识点

- **协同过滤**：利用用户-物品交互矩阵中的模式做个性化推荐
- **隐因子模型**：\(\hat{r}_{ui} = P_u \cdot Q_i\)，用户与物品共享低维表示
- **SGD 更新**：仅在观测评分上更新 \(P\)、\(Q\)，带 L2 正则防过拟合
- **流行度基线**：全局均值预测；强基线但无个性化
- **训练/测试划分**：从观测条目中 hold-out 评估泛化 MSE
- **Top-K 推荐**：用 \(P_u \cdot Q_i^T\) 排序未交互物品

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | MF 直觉、训练目标、局限、自检问题 |
| `from_scratch/matrix_factorization.py` | numpy SGD 矩阵分解 + Top-K |
| `rating_data.py` | 合成评分数据生成 |
| `reproduce/mf_baseline.py` | 实验入口 |
| `experiments/baseline_comparison.py` | 流行度 vs MF 的 test MSE |
| `report.md` | 实验结论与复习命令 |

## 如何运行

```bash
# 从零实现：MF smoke test
python modules/14_recommender_systems/from_scratch/matrix_factorization.py

# 核心实验：流行度基线 vs MF（含训练曲线图）
python modules/14_recommender_systems/experiments/baseline_comparison.py
```

## 建议学习顺序

1. 阅读 `notes.md` — 理解 \(P\)、\(Q\) 与预测公式
2. 阅读 `rating_data.py` — 看合成数据如何生成
3. 阅读 `from_scratch/matrix_factorization.py` — 对照 SGD 双循环更新
4. 运行 `matrix_factorization.py` — 确认 MSE 下降与 Top-K 输出
5. 运行 `baseline_comparison.py` — 对比 test MSE
6. 阅读 `report.md` — 思考小稀疏数据上基线为何仍强

## 与前后模块的联系

- **前置 Module 02**：线性模型与 MSE 损失
- **前置 Module 09**：SGD 与正则化
- **延伸**：神经协同过滤、双塔召回等是 MF 的深度学习扩展

## Depth Checklist

- [x] MF from scratch
- [x] Popularity vs MF comparison
- [x] Training loss and test MSE report
