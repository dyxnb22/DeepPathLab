"""Tabular Q-learning from scratch."""

from __future__ import annotations

import numpy as np

from gridworld import GridWorld


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    return int(Q[state].argmax())


def q_learning(
    n_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    env = GridWorld(size=4)
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done = env.step(action)
            best_next = Q[next_state].max()
            Q[state, action] += alpha * (reward + gamma * best_next * (not done) - Q[state, action])
            state = next_state
            total_reward += reward
        returns.append(total_reward)
    return Q, returns


if __name__ == "__main__":
    Q, returns = q_learning()
    print(f"Final 10-episode avg return: {np.mean(returns[-10:]):.3f}")
    goal_state = GridWorld(size=4).goal
    print(f"Best action at start state 0: {Q[0].argmax()}")
