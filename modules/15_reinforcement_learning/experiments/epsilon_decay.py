#!/usr/bin/env python3
"""Compare epsilon schedules for exploration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/15_reinforcement_learning/from_scratch"))

from lib.plotting import module_output_dir, save_metric_curves
from gridworld import GridWorld
from q_learning import epsilon_greedy


def q_learning_eps(n_episodes: int, epsilon_fn) -> list[float]:
    env = GridWorld(size=4)
    rng = np.random.default_rng(0)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for ep in range(n_episodes):
        eps = epsilon_fn(ep)
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            action = epsilon_greedy(Q, state, eps, rng)
            ns, reward, done = env.step(action)
            Q[state, action] += 0.5 * (reward + 0.99 * Q[ns].max() * (not done) - Q[state, action])
            state = ns
            total += reward
        returns.append(total)
    return returns


def main() -> None:
    n = 400
    curves = {
        "const_0.2": q_learning_eps(n, lambda _: 0.2),
        "decay": q_learning_eps(n, lambda ep: max(0.05, 0.5 * (0.995 ** ep))),
        "const_0.05": q_learning_eps(n, lambda _: 0.05),
    }
    finals = {k: np.mean(v[-20:]) for k, v in curves.items()}
    print("Final 20-ep avg return:")
    for k, v in finals.items():
        print(f"  {k}: {v:.3f}")

    out = module_output_dir("15_reinforcement_learning") / "epsilon_schedules.png"
    save_metric_curves(curves, out, title="Epsilon Schedule Comparison", xlabel="Episode")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
