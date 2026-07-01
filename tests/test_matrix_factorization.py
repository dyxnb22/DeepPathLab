"""Tests for Module 14 matrix factorization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules/14_recommender_systems/from_scratch"))
from matrix_factorization import predict, train_matrix_factorization  # noqa: E402


class TestMatrixFactorization(unittest.TestCase):
    def test_training_loss_decreases(self) -> None:
        rng = np.random.default_rng(0)
        ratings = rng.uniform(1, 5, size=(20, 15))
        mask = rng.random((20, 15)) < 0.5
        ratings_masked = np.where(mask, ratings, 0.0)
        _, _, losses = train_matrix_factorization(
            ratings_masked, mask, n_factors=4, n_epochs=50, lr=0.05, reg=0.001
        )
        self.assertLess(losses[-1], losses[0])

    def test_predict_is_dot_product(self) -> None:
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q = np.array([[2.0, 3.0], [4.0, 5.0]])
        # user 0 · item 1 = [1,0] · [4,5] = 4
        self.assertAlmostEqual(predict(P, Q, 0, 1), 4.0)


if __name__ == "__main__":
    unittest.main()
