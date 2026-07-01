"""Tests for Module 02 linear regression."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules/02_linear_models/from_scratch"))
from linear_regression import closed_form_solution, mse_loss, predict, train_gd  # noqa: E402


class TestLinearRegression(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(50, 3))
        self.w_true = np.array([[1.0], [-0.5], [2.0]])
        self.b_true = 0.3
        self.y = self.X @ self.w_true + self.b_true + rng.normal(scale=0.01, size=(50, 1))

    def test_closed_form_low_mse(self) -> None:
        w, b = closed_form_solution(self.X, self.y)
        mse = mse_loss(predict(self.X, w, b), self.y)
        self.assertLess(mse, 1e-3)

    def test_gd_loss_decreases(self) -> None:
        _, _, losses = train_gd(self.X, self.y, lr=0.1, n_epochs=300)
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
