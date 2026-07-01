"""Gridworld environment for tabular RL.

A minimal MDP: agent starts top-left, must reach bottom-right goal.
One interior cell is a wall that blocks movement.
"""

from __future__ import annotations

import numpy as np


class GridWorld:
    """Simple grid: agent reaches goal (G) avoiding wall (W)."""

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 0=up, 1=right, 2=down, 3=left
        self.goal = size * size - 1  # bottom-right corner
        self.walls = {size + 1} if size >= 3 else set()  # one cell below start
        self.reset()

    def reset(self) -> int:
        """Reset agent to state 0 (top-left)."""
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take action, return (next_state, reward, done)."""
        row, col = divmod(self.state, self.size)
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.size - 1, col + 1)
        elif action == 2:
            row = min(self.size - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)
        next_state = row * self.size + col
        if next_state in self.walls:
            next_state = self.state  # bump into wall — stay put
        self.state = next_state
        if self.state == self.goal:
            return self.state, 1.0, True
        return self.state, -0.01, False  # small step penalty encourages shortest path
