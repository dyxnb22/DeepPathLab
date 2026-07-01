"""Tests for Module 01 scalar autograd."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules/01_preliminaries_autograd/from_scratch"))
from value import Value  # noqa: E402


class TestAutograd(unittest.TestCase):
    def test_mul_chain(self) -> None:
        a, b = Value(2.0), Value(3.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(a.grad, 3.0)
        self.assertAlmostEqual(b.grad, 2.0)

    def test_shared_variable(self) -> None:
        a, b, c = Value(2.0), Value(3.0), Value(4.0)
        out = a * b + b * c
        out.backward()
        # ∂/∂b = a + c = 6
        self.assertAlmostEqual(b.grad, 6.0)

    def test_relu_gate(self) -> None:
        x_pos, x_neg = Value(1.5), Value(-2.0)
        y_pos, y_neg = x_pos.relu(), x_neg.relu()
        (y_pos + y_neg).backward()
        self.assertAlmostEqual(x_pos.grad, 1.0)
        self.assertAlmostEqual(x_neg.grad, 0.0)

    def test_pow_and_tanh(self) -> None:
        x = Value(0.5)
        y = (x ** 2).tanh()
        y.backward()
        t = math.tanh(0.25)
        expected = 2 * 0.5 * (1 - t * t)
        self.assertAlmostEqual(x.grad, expected, places=5)


if __name__ == "__main__":
    unittest.main()
