# 报告：Reinforcement Learning

## 目标

在 4×4 GridWorld 上实现 tabular Q-learning，比较探索策略。

## 实现范围

- `from_scratch/gridworld.py` — 网格环境（动作、奖励、终止）
- `from_scratch/q_learning.py` — ε-greedy + TD 更新
- `experiments/epsilon_decay.py` — 探索 schedule 对照

## 运行

```bash
python3 modules/15_reinforcement_learning/experiments/epsilon_decay.py
```

## 实验结果

### Q-learning（600 episodes）

- 最终平滑 return：**0.94**（goal reward=1，步惩罚=-0.01）
- 起始状态最优动作：向下/向右（朝向 goal）

### Epsilon schedule（20-ep 平均 return）

| 策略 | Return |
|------|--------|
| const 0.2 | 0.931 |
| decay 0.5→0.05 | 0.945 |
| const 0.05 | **0.948** |

本环境中衰减探索与固定低 ε 均表现良好。

## 深度检查点

- [x] GridWorld + Q-learning 实现
- [x] 学习曲线
- [x] ε-greedy 对照

## 收获

RL 的核心是 trial-and-error + bootstrap 更新；表格法是小规模理解 Q-learning 的最佳起点。

## 路线图完成

Module 01–15 覆盖从 autograd 到 NLP 微调、推荐、RL 的完整学习路径。
