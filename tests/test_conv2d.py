"""Tests for Module 04 conv2d output shapes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules/04_cnn/from_scratch"))
from conv2d import conv2d, max_pool2d  # noqa: E402


class TestConv2d(unittest.TestCase):
    def test_conv_output_shape(self) -> None:
        x = np.ones((5, 5))
        k = np.ones((3, 3))
        out = conv2d(x, k, stride=1, padding=0)
        self.assertEqual(out.shape, (3, 3))

    def test_conv_with_padding(self) -> None:
        x = np.arange(16, dtype=float).reshape(4, 4)
        k = np.array([[1.0, 0.0], [0.0, -1.0]])
        out = conv2d(x, k, stride=1, padding=1)
        self.assertEqual(out.shape, (5, 5))

    def test_max_pool_halves_spatial(self) -> None:
        x = np.arange(16, dtype=float).reshape(4, 4)
        out = max_pool2d(x, pool_size=2, stride=2)
        self.assertEqual(out.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
