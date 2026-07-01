# 15 强化学习

Track F — 表格型 Q-learning 入门。

## 核心问题

Q-learning 如何在不知环境模型的情况下学策略？

## 核心知识点

- **MDP 五元组**：状态、动作、转移、奖励、折扣 \(\gamma\)
- **Q 函数**：\(Q(s,a)\) = 从 \((s,a)\) 出发按最优策略的期望回报
- **贝尔曼最优**：\(Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s',a')]\)
- **Q-learning 更新**：用 TD 目标 bootstrap，离策略（max over actions）
- **\(\epsilon\)-greedy**：平衡探索与利用；衰减 \(\epsilon\) 的常见策略
- **表格法局限**：仅适用于小离散状态空间；大状态需 DQN 等函数逼近

## 项目产出

| 目录 | 内容 |
|------|------|
| `notes.md` | MDP、Q-learning 公式、探索策略、自检问题 |
| `from_scratch/gridworld.py` | 4×4 网格环境（墙、目标、步惩罚） |
| `from_scratch/q_learning.py` | 表格 Q-learning 训练循环 |
| `reproduce/q_learning_demo.py` | 学习曲线可视化 |
| `experiments/epsilon_decay.py` | 不同 \(\epsilon\) 策略对照 |
| `report.md` | 实验结论与复习命令 |

## 如何运行

```bash
# 从零实现：Q-learning smoke test
python modules/15_reinforcement_learning/from_scratch/q_learning.py

# 复现：学习曲线
python modules/15_reinforcement_learning/reproduce/q_learning_demo.py

# 实验：epsilon 策略对照
python modules/15_reinforcement_learning/experiments/epsilon_decay.py
```

## 建议学习顺序

1. 阅读 `notes.md` — 理解 MDP 与 Q-learning 更新式
2. 阅读 `gridworld.py` — 弄清状态编码、动作、奖励
3. 阅读 `q_learning.py` — 对照 TD bootstrap 一行更新
4. 运行 `q_learning.py` — 看最终 return 与起始状态 Q 值
5. 运行 `q_learning_demo.py` — 观察学习曲线
6. 运行 `epsilon_decay.py` — 比较探索策略
7. 阅读 `report.md` — 回顾表格 RL 的适用边界

## 与前后模块的联系

- **独立入门**：不硬性依赖神经网络模块；概率与期望即可
- **与 Module 09 类比**：都是「用数据迭代改进策略」，但 RL 无监督标签、靠奖励信号
- **延伸**：DQN、Policy Gradient、PPO 等将 Q 或策略函数参数化

## Depth Checklist

- [x] Gridworld + tabular Q-learning
- [x] Learning curve visualization
- [x] Epsilon schedule comparison
