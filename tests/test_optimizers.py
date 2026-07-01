"""Tests for Module 09 optimizers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules/09_optimization/from_scratch"))
from optimizers import Adam, SGD, SGDMomentum  # noqa: E402


class TestOptimizers(unittest.TestCase):
    def test_sgd_moves_toward_minimum(self) -> None:
        params = {"w": np.array([5.0])}
        opt = SGD(lr=0.1)
        for _ in range(20):
            grads = {"w": 2 * params["w"]}  # grad of w^2
            opt.step(params, grads)
        self.assertLess(abs(params["w"][0]), 0.5)

    def test_momentum_accumulates_velocity(self) -> None:
        opt = SGDMomentum(lr=0.1, momentum=0.9)
        params = {"w": np.array([1.0])}
        opt.step(params, {"w": np.array([1.0])})
        self.assertIn("w", opt.velocity)
        self.assertGreater(abs(opt.velocity["w"][0]), 0.0)

    def test_adam_step(self) -> None:
        params = {"w": np.array([1.0])}
        opt = Adam(lr=0.1)
        opt.step(params, {"w": np.array([1.0])})
        self.assertNotEqual(params["w"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
