"""Gridworld environment for tabular RL."""

from __future__ import annotations

import numpy as np


class GridWorld:
    """Simple grid: agent reaches goal (G) avoiding wall (W)."""

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, right, down, left
        self.goal = size * size - 1
        self.walls = {size + 1} if size >= 3 else set()
        self.reset()

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
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
            next_state = self.state
        self.state = next_state
        if self.state == self.goal:
            return self.state, 1.0, True
        return self.state, -0.01, False
