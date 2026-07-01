# 笔记：强化学习

## 核心问题

Q-learning 如何在不知环境模型的情况下学策略？

## MDP 要素

- **状态** \(s\)：环境描述
- **动作** \(a\)：智能体选择
- **奖励** \(r\)：即时反馈
- **策略** \(\pi(a|s)\)：状态到动作的映射

## Q 函数

\(Q(s,a)\) = 从状态 s 执行 a 后按最优策略的期望累积回报。

## Q-learning 更新

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
\]

这是**离策略**时序差分学习：用 max over next actions 作为 bootstrap 目标（贪心策略）。

`(not done)` 项确保终止状态不再 bootstrap 未来价值。

## 探索 vs 利用

\(\epsilon\)-greedy：以 \(\epsilon\) 随机探索，否则选当前最优动作。衰减 \(\epsilon\) 可在早期多探索、后期多利用。

## GridWorld 环境

`from_scratch/gridworld.py` 提供 4×4 网格：从左上角出发，到达右下角 goal 获 +1，每步 -0.01，中间有一堵墙。

动作编码：0=上, 1=右, 2=下, 3=左。

## 本模块脚本

| 路径 | 作用 |
|------|------|
| `from_scratch/gridworld.py` | 简易网格环境 |
| `from_scratch/q_learning.py` | 表格 Q-learning |
| `experiments/epsilon_decay.py` | 探索策略对照 |

```bash
python3 modules/15_reinforcement_learning/experiments/epsilon_decay.py
```

## 局限

- 本模块为**表格型** Q-learning，仅适用于小状态空间
- 大状态空间需函数逼近（DQN 等）
