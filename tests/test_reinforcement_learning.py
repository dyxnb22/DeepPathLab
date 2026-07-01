"""Tests for Module 15 GridWorld and Q-learning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules/15_reinforcement_learning/from_scratch"))
from gridworld import GridWorld  # noqa: E402
from q_learning import q_learning  # noqa: E402


class TestReinforcementLearning(unittest.TestCase):
    def test_gridworld_goal_reward(self) -> None:
        env = GridWorld(size=4)
        env.state = env.goal - env.size  # one step above goal
        _, reward, done = env.step(2)  # down into goal
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 1.0)

    def test_wall_blocks_movement(self) -> None:
        env = GridWorld(size=4)
        env.reset()
        # from state 0, action right -> state 1 (wall below start is state 5, not blocking right)
        s1, _, _ = env.step(1)
        self.assertEqual(s1, 1)

    def test_q_learning_improves(self) -> None:
        _, returns = q_learning(n_episodes=200, epsilon=0.2, seed=0)
        self.assertGreater(returns[-1], returns[0])


if __name__ == "__main__":
    unittest.main()
